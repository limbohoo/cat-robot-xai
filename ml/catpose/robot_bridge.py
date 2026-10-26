"""Robot-car bridge for a test session: one clock, one log, one remote stop.

Runs on the laptop the team uses during the test. Every robot action passes
through here and is written to robot_log.csv in the format catpose.session
reads (time,event,detail). Times come from THIS computer's clock only, so the
car's own clock never matters.

Two ways for the car to be connected (both can be used at once):

  A. Control through the bridge. Open http://<laptop-ip>:8800 on a phone; each
     button logs the action and forwards it to the car with --car-url, e.g.
       --car-url "http://192.168.4.1/cmd?c={cmd}"
  B. Keep the team's own remote. The car firmware calls the bridge every time
     it executes a command:
       GET http://<laptop-ip>:8800/event?e=slow_prey&d=round1
     and may poll GET /status, which returns {"stop": true} after STOP.

The red STOP button always logs a safety_stop and forwards "stop" to the car.
The SYNC button logs a sync marker: press it and clap once in view of all
cameras, then put the clap's video time in session.json ("clap") so the robot
log is aligned to the videos exactly (see catpose.session).

    python -m catpose.robot_bridge SESSION_FOLDER [--car-url URL] [--port 8800]
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import threading
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# button id -> (label on the page, event name in the log)
ACTIONS = [
    ("slow_prey", "慢速逗猫（逃跑的猎物）"),
    ("light", "灯光"),
    ("sound", "声音"),
    ("approach", "慢慢靠近"),
    ("retreat", "后退离开"),
    ("round_end", "本轮结束"),
]

MARKERS = {"bridge_start", "bridge_end", "sync_clap", "round_end", "car_unreachable"}

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>小车记录</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#fbfaf7;color:#1d2130}
h1{font-size:20px;margin:0 0 4px}p{margin:4px 0 12px;color:#4a5263}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{font-size:17px;padding:16px 8px;border-radius:12px;border:1px solid #c9c6be;background:#fff}
input{width:100%;box-sizing:border-box;font-size:16px;padding:10px;margin:8px 0;border-radius:10px;border:1px solid #c9c6be}
#stop{width:100%;font-size:26px;font-weight:700;padding:26px;margin:12px 0;background:#c62828;color:#fff;border:0}
#sync{width:100%;margin-top:10px}
#log{font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap;margin-top:12px}
.warn{color:#c62828}
</style></head><body>
<h1>小车记录 · __NAME__</h1><p id="st">__CAR__</p>
<button id="stop" onclick="send('safety_stop')">停 止</button>
<input id="note" placeholder="备注（可选，例如：第 1 轮 / 猫躲开了）">
<div class="row">__BUTTONS__</div>
<button id="sync" onclick="send('sync_clap')">同步：按下后马上拍一次手</button>
<div id="log"></div>
<script>
async function send(e){
  const d=document.getElementById('note').value;
  const r=await fetch('/event?src=page&e='+encodeURIComponent(e)+'&d='+encodeURIComponent(d));
  const j=await r.json(); document.getElementById('note').value='';
  show(j);
}
function show(j){
  const el=document.getElementById('log');
  el.innerHTML=j.recent.map(x=>x.join('  ')).join('\\n');
  if(j.car_error){el.innerHTML='<span class="warn">小车没有回应：'+j.car_error+'</span>\\n'+el.innerHTML;}
}
setInterval(async()=>show(await (await fetch('/status')).json()),3000);
</script></body></html>"""


class Bridge:
    def __init__(self, folder: Path, car_url: str | None, timeout: float):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "robot_log.csv"
        self.car_url, self.timeout = car_url, timeout
        self.lock = threading.Lock()
        self.stop = False
        self.recent: list[list[str]] = []
        self.car_error = ""
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.f = self.path.open("a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "event", "detail", "source"])
        self.log("bridge_start", "", "bridge")

    def log(self, event: str, detail: str, source: str) -> str:
        now = datetime.now().astimezone().isoformat(timespec="milliseconds")
        with self.lock:
            self.w.writerow([now, event, detail, source])
            self.f.flush()  # a crash or power cut keeps everything logged so far
            self.recent = ([[now[11:23], event, detail, source]] + self.recent)[:12]
            if "stop" in event:
                self.stop = True
            elif source in ("page", "car") and event not in MARKERS:
                self.stop = False  # only a new action from the team resumes after a stop
        print(f"{now}  {event:12s} {detail}  ({source})", flush=True)
        return now

    def forward(self, cmd: str) -> None:
        if not self.car_url:
            return
        url = self.car_url.replace("{cmd}", urllib.parse.quote(cmd))
        try:
            urllib.request.urlopen(url, timeout=self.timeout).read()
            self.car_error = ""
        except OSError as e:
            self.car_error = str(e)
            self.log("car_unreachable", f"{cmd}: {e}", "bridge")

    def status(self) -> dict:
        return {"stop": self.stop, "recent": self.recent, "car_error": self.car_error}


def make_handler(bridge: Bridge, name: str):
    buttons = "".join(f'<button onclick="send(\'{k}\')">{v}</button>' for k, v in ACTIONS)
    car = f"转发到小车：{bridge.car_url}" if bridge.car_url else "只记录，不转发（小车用自己的遥控，并调用 /event）"
    page = PAGE.replace("__NAME__", name).replace("__BUTTONS__", buttons).replace("__CAR__", car)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            if u.path == "/":
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/event" and q.get("e"):
                event = q["e"].strip()[:40]
                src = q.get("src", "car")
                bridge.log(event, q.get("d", "")[:200], src)
                if src == "page" and event != "sync_clap":
                    bridge.forward("stop" if "stop" in event else event)
                self._json(bridge.status())
            elif u.path == "/stop":
                bridge.log("safety_stop", q.get("d", ""), q.get("src", "remote"))
                bridge.forward("stop")
                self._json(bridge.status())
            elif u.path == "/status":
                self._json(bridge.status())
            else:
                self.send_error(404)

        do_POST = do_GET  # firmware that can only POST works too

        def log_message(self, *a):  # keep the console for the event lines
            pass

    return Handler


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Log robot-car actions on this computer's clock, with a remote stop")
    p.add_argument("session", help="session folder (robot_log.csv is written here)")
    p.add_argument("--car-url", default=None, help='forward commands, e.g. "http://192.168.4.1/cmd?c={cmd}"')
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--timeout", type=float, default=1.5, help="seconds to wait for the car")
    args = p.parse_args()
    folder = Path(args.session)
    bridge = Bridge(folder, args.car_url, args.timeout)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(bridge, folder.name))
    print(f"control page: http://{_lan_ip()}:{args.port}   log: {bridge.path}", flush=True)
    if args.car_url:
        bridge.forward("ping")
        print("car: " + (f"NOT reachable ({bridge.car_error})" if bridge.car_error else "reachable"), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        bridge.log("bridge_end", "", "bridge")


if __name__ == "__main__":
    main()
