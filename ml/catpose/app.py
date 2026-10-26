"""Owner app: a phone-style web app over the analysed reports, with feedback stored on the server.

    python -m catpose.app reports sessions --port 8000

Every folder under the given roots that holds a report.json (written by
catpose.realtime --report-dir or catpose.session) shows up as one entry. Open
http://<this-computer>:8000 on a phone in the same Wi-Fi, or on this computer.

The owner's answers ("right" / "wrong, it was ..." / "not sure") are saved to
feedback.json inside that report's folder, and /api/reports/<id>/labels.csv
turns them into the labels file catpose.interpret learns from.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import socket
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

UI = Path(__file__).with_name("app_ui.html")
STATES = {"tense", "curious", "playful", "attentive", "calm"}
VERDICTS = {"right", "wrong", "unsure", "label"}  # label: owner names it where the system gave no guess


class Feedback(BaseModel):
    moment: int
    verdict: str  # right / wrong / unsure / label
    actual: str = ""  # with "wrong" or "label": what the owner thinks it was (one of STATES)
    note: str = ""


class Library:
    """Finds report folders under the roots; ids are short hashes of their paths."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = [r.resolve() for r in roots]
        self.lock = threading.Lock()

    def folders(self) -> dict[str, Path]:
        found = {}
        for root in self.roots:
            for f in root.rglob("report.json"):
                if "_work" in f.parts:  # per-camera intermediates of a session
                    continue
                found[hashlib.sha1(str(f.parent).encode()).hexdigest()[:10]] = f.parent
        return found

    def folder(self, rid: str) -> Path:
        f = self.folders().get(rid)
        if f is None:
            raise HTTPException(404, "report not found")
        return f

    def feedback(self, folder: Path) -> dict:
        p = folder / "feedback.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def save_feedback(self, folder: Path, fb: Feedback) -> dict:
        with self.lock:
            store = self.feedback(folder)
            store[str(fb.moment)] = {"verdict": fb.verdict, "actual": fb.actual, "note": fb.note[:500],
                                     "at": datetime.now().astimezone().isoformat(timespec="seconds")}
            (folder / "feedback.json").write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
        return store


def make_app(lib: Library) -> FastAPI:
    app = FastAPI(title="CatPose owner app")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI.read_text(encoding="utf-8")

    @app.get("/api/reports")
    def reports() -> list[dict]:
        out = []
        for rid, folder in lib.folders().items():
            d = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            cover = next((m["after_img"] for m in d["moments"] if m["show"]), None) or \
                next((m["after_img"] for m in d["moments"]), None)
            out.append({"id": rid, "title": d["title"], "period": d["period"], "generated": d["generated"],
                        "summary": d["summary"], "n_moments": sum(m["show"] for m in d["moments"]),
                        "n_alerts": len(d["alerts"]), "cover": cover,
                        "n_feedback": len(lib.feedback(folder))})
        return sorted(out, key=lambda r: r["generated"], reverse=True)

    @app.get("/api/reports/{rid}")
    def report(rid: str) -> dict:
        folder = lib.folder(rid)
        d = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        return {**d, "id": rid, "feedback": lib.feedback(folder)}

    @app.post("/api/reports/{rid}/feedback")
    def feedback(rid: str, fb: Feedback) -> dict:
        if fb.verdict not in VERDICTS or (fb.actual and fb.actual not in STATES) or (fb.verdict == "label" and not fb.actual):
            raise HTTPException(400, "verdict must be right / wrong / unsure / label; actual one of " + ", ".join(sorted(STATES)))
        folder = lib.folder(rid)
        n = len(json.loads((folder / "report.json").read_text(encoding="utf-8"))["moments"])
        if not 0 <= fb.moment < n:
            raise HTTPException(400, "no such moment")
        return lib.save_feedback(folder, fb)

    @app.get("/api/reports/{rid}/labels.csv")
    def labels(rid: str) -> Response:
        """Owner feedback as labels for `python -m catpose.interpret --labels`."""
        folder = lib.folder(rid)
        d = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["event_id", "your_label", "system_guess", "owner_feedback", "note"])
        for k, f in sorted(lib.feedback(folder).items(), key=lambda kv: int(kv[0])):
            m = d["moments"][int(k)]
            label = (m["guess"] if f["verdict"] == "right" and m["show"]
                     else f["actual"] if f["verdict"] in ("wrong", "label") and f["actual"] else "unsure")
            w.writerow([k, label, m["guess"] if m["show"] else "", f["verdict"], f["note"]])
        return Response("﻿" + buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="labels_{rid}.csv"'})

    @app.get("/media/{rid}/{path:path}")
    def media(rid: str, path: str) -> FileResponse:
        folder = lib.folder(rid)
        target = (folder / path).resolve()
        if folder not in target.parents or not target.is_file():  # no escaping the report folder
            raise HTTPException(404)
        return FileResponse(target)

    return app


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main(argv: Optional[list[str]] = None) -> None:
    import uvicorn
    p = argparse.ArgumentParser(description="Owner app over analysed cat reports")
    p.add_argument("roots", nargs="*", default=["reports", "sessions"],
                   help="folders to search for report.json (default: reports sessions)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from phones on this Wi-Fi")
    args = p.parse_args(argv)
    roots = [Path(r) for r in args.roots if Path(r).exists()]
    if not roots:
        raise SystemExit(f"none of these folders exist: {args.roots}")
    lib = Library(roots)
    print(f"{len(lib.folders())} report(s) found.  Open http://{_lan_ip()}:{args.port}  (or http://localhost:{args.port})")
    uvicorn.run(make_app(lib), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
