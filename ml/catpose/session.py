"""One test session -> one owner report: several videos and the robot log on one timeline.

A session folder holds the videos of one test (e.g. a room camera and the
robot's own camera), the robot's log, and optionally a session.json:

    session/
      session.json      optional, see `--init`
      room.mp4
      robot_cam.mp4
      robot_log.csv     time,event,detail

Steps:
1. Session clock. Each video's start time comes from session.json ("start" as
   ISO time, or "offset" in seconds), else from the file's creation_time
   metadata. Time zero is the earliest start (or session.json "start").
2. Audio sync (default on). Recorded sound is shared by all cameras in a room,
   so each video's offset is refined by cross-correlating its loudness envelope
   with the first video's. The peak correlation is reported as the sync
   confidence; a weak peak keeps the metadata offset and says so.
3. Robot log (written by catpose.robot_bridge, or by hand). Times are session
   seconds, epoch seconds, or ISO times. Events containing "stop" are safety
   stops; bridge_*/sync_* lines are markers; the others are robot actions. If
   session.json has "clap": {"video": "room.mp4", "s": 12.3} (the second of
   the clap in that video) and the log has a sync_clap, the whole log is
   shifted so the two coincide, removing any clock difference.
4. Each video is analysed (catpose.realtime) with its offset, so all events
   share the session clock. Moments seen by two cameras within 2 s are merged:
   the clearer view leads and the other becomes "see from another angle".
5. One owner report (report.html + clips/), events.json and labels_template.csv
   for labelling, timeline.csv, and sync.json describing how times were aligned.

    python -m catpose.session --init path/to/session     # write templates
    python -m catpose.session path/to/session --dry-run  # check sync only
    python -m catpose.session path/to/session            # analyse + report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .events import Event, EventRecorder
from .interpret import DEFAULT_CONFIG, load_config

STATE_TEXT = DEFAULT_CONFIG["state_text"]
from .report_ui import render_owner_report

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ENV_RATE = 50  # loudness envelope samples per second, used for audio sync
MERGE_S = 2.0  # moments from different cameras this close together are the same moment
BAND_RANK = {"高": 2, "中": 1, "低": 0}
MIN_COVERAGE = 0.5  # below this share of the session with the cat in view, the report warns up front


# ---------- clock helpers ----------
def _parse_time(value: str) -> Optional[datetime]:
    """ISO time -> aware datetime (no timezone given = this computer's local time)."""
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.astimezone()


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=creation_time",
                          "-show_entries", "stream=codec_type", "-of", "json", str(path)],
                         capture_output=True, text=True)
    d = json.loads(out.stdout or "{}")
    fmt = d.get("format", {})
    return {"duration": float(fmt.get("duration", 0) or 0),
            "creation_time": (fmt.get("tags") or {}).get("creation_time"),
            "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def _envelope(path: Path) -> Optional[np.ndarray]:
    """Log-loudness envelope at ENV_RATE Hz, or None without audio."""
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000",
                              "-f", "s16le", "-"], capture_output=True, check=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = 8000 // ENV_RATE
    n = len(x) // hop
    if n < ENV_RATE * 5:
        return None
    env = np.log(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-8)
    return (env - env.mean()) / (env.std() + 1e-8)


def _audio_lag(ref: np.ndarray, other: np.ndarray, guess_s: float, max_lag_s: float) -> tuple[float, float]:
    """Offset (s) of `other` relative to `ref` that best aligns their sound, and the correlation there.

    A sound at session time T is at ref-time T - off_ref and other-time T - off_other,
    so ref[n] ≈ other[n - k] with k = (off_other - off_ref) * rate.
    """
    n = len(ref) + len(other) - 1
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(other, size)), size)
    lags = np.arange(size)
    lags[lags >= size // 2] -= size  # circular -> signed lags
    lo, hi = (guess_s - max_lag_s) * ENV_RATE, (guess_s + max_lag_s) * ENV_RATE
    ok = (lags >= lo) & (lags <= hi)
    if not ok.any():
        return guess_s, 0.0
    k = int(lags[ok][np.argmax(corr[ok])])
    a = ref[max(0, k): min(len(ref), len(other) + k)]
    b = other[max(0, -k): max(0, -k) + len(a)]
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > ENV_RATE * 5 else 0.0
    return k / ENV_RATE, r


# ---------- session setup ----------
def load_session(folder: Path, audio_sync: bool, max_lag_s: float) -> dict:
    cfg = json.loads((folder / "session.json").read_text()) if (folder / "session.json").exists() else {}
    entries = cfg.get("videos") or [{"path": p.name} for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not entries:
        raise SystemExit(f"no videos found in {folder}")
    videos = []
    for i, v in enumerate(entries):
        path = (folder / v["path"]).resolve()
        info = _probe(path)
        start, how = None, "未知（按 0 处理）"
        if "start" in v and _parse_time(v["start"]):
            start, how = _parse_time(v["start"]), "session.json 中填写的开始时间"
        elif info["creation_time"] and _parse_time(info["creation_time"]):
            start, how = _parse_time(info["creation_time"]), "视频文件里的录制时间"
        videos.append({"id": f"cam{i}", "name": v.get("name", path.stem), "path": path, "start": start,
                       "moving": bool(v.get("moving", False)),
                       "offset": v.get("offset"), "how": how, **info})
    known = [v["start"] for v in videos if v["start"] is not None]
    t0 = _parse_time(cfg["start"]) if cfg.get("start") else (min(known) if known else None)
    for v in videos:
        if v["offset"] is not None:
            v["offset"], v["how"] = float(v["offset"]), "session.json 中填写的偏移"
        else:
            v["offset"] = (v["start"] - t0).total_seconds() if (v["start"] is not None and t0) else 0.0
    # audio refinement against the first video that has sound
    ref = next((v for v in videos if v["has_audio"]), None)
    if audio_sync and ref is not None and len(videos) > 1:
        ref_env = _envelope(ref["path"])
        for v in videos:
            if v is ref or not v["has_audio"] or ref_env is None or "session.json 中填写的偏移" == v["how"]:
                continue
            env = _envelope(v["path"])
            if env is None:
                continue
            guess = v["offset"] - ref["offset"]
            lag, r = _audio_lag(ref_env, env, guess, max_lag_s)
            v["audio_r"] = round(r, 2)
            if r >= 0.3:
                v["offset"], v["how"] = ref["offset"] + lag, f"按声音对齐（相关系数 {r:.2f}）"
            else:
                v["how"] += f"（声音对齐不可靠，相关系数 {r:.2f}，未采用）"
    shift = -min(v["offset"] for v in videos)  # session time never negative
    for v in videos:
        v["offset"] += shift
    if t0 is not None:
        t0 = t0 - timedelta(seconds=shift)
    clap_s = None
    if isinstance(cfg.get("clap"), dict) and cfg["clap"].get("s") is not None:
        v = next((v for v in videos if v["path"].name == cfg["clap"].get("video")), None)
        clap_s = v["offset"] + float(cfg["clap"]["s"]) if v else None
    return {"name": cfg.get("name", folder.name), "t0": t0, "videos": videos, "clap_s": clap_s,
            "robot_log": cfg.get("robot_log") or next((p.name for p in sorted(folder.glob("robot*"))
                                                       if p.suffix.lower() in {".csv", ".jsonl"}), None)}


def load_robot_log(path: Optional[Path], t0: Optional[datetime],
                   clap_s: Optional[float] = None) -> tuple[list, list, list]:
    """-> (actions [(t, name)], stops [(t, detail)], warnings). Times become session seconds.

    clap_s: session time of the sync clap seen in a video; the log is shifted so its
    last sync_clap lands there.
    """
    if path is None or not path.exists():
        return [], [], []
    rows = ([json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.suffix == ".jsonl"
            else list(csv.DictReader(path.open(newline="", encoding="utf-8-sig"))))
    actions, stops, warn, claps = [], [], [], []
    for r in rows:
        raw = str(r.get("time", r.get("time_s", r.get("timestamp", "")))).strip()
        name = str(r.get("event", r.get("action", ""))).strip()
        detail = str(r.get("detail", "") or "").strip()
        t = None
        try:
            x = float(raw)
            if x > 1e9:  # epoch seconds
                t = (datetime.fromtimestamp(x, timezone.utc) - t0).total_seconds() if t0 else None
            else:  # already session seconds
                t = x
        except ValueError:
            dt = _parse_time(raw)
            t = (dt - t0).total_seconds() if (dt and t0) else None
        if t is None:
            warn.append(f"无法换算时间：{raw}")
            continue
        low = name.lower()
        if low.startswith("sync"):
            claps.append(t)
        elif low == "car_unreachable":
            warn.append(f"小车在 {t:.0f} 秒时没有回应命令：{detail}")
        elif "stop" in low:
            stops.append((t, detail or name))
        elif not (low.endswith("end") or low.startswith("bridge")):
            actions.append((t, name))
    if clap_s is not None:
        if claps:
            shift = clap_s - claps[-1]
            actions = [(t + shift, a) for t, a in actions]
            stops = [(t + shift, d) for t, d in stops]
            warn.append(f"小车记录按拍手对齐，整体移动了 {shift:+.2f} 秒")
        else:
            warn.append("session.json 写了拍手时间，但小车记录里没有 sync_clap，未对齐")
    return sorted(actions), sorted(stops), warn


# ---------- analysis and merge ----------
def _analyse(v: dict, work: Path, robot_csv: Path, args) -> Optional[Path]:
    """Analyse one video into work/<id>; None if it could not be read (the session goes on without it)."""
    out = work / v["id"]
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-m", "catpose.realtime", "--source", str(v["path"]), "--detect", "--no-display",
           "--no-print", "--calib-frames", str(args.calib_frames), "--report-dir", str(out),
           "--robot-log", str(robot_csv), "--time-offset", f"{v['offset']:.3f}"]
    if v["moving"]:
        cmd.append("--moving-camera")
    if getattr(args, "cats", None) and getattr(args, "cat", None):
        cmd += ["--cats", str(args.cats), "--cat", args.cat]  # follow one named cat, by coat colour
    if args.interp_config:
        cmd += ["--interp-config", args.interp_config]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print(f"[{v['id']}] analysing {v['path'].name} (offset {v['offset']:.2f} s) ...", flush=True)
    if subprocess.run(cmd, env=env).returncode != 0 or not (out / "track.csv").exists():
        print(f"[{v['id']}] could not analyse {v['path'].name}; continuing without it", flush=True)
        return None
    return out


def _load_track(folder: Path) -> list[tuple]:
    rows = []
    with open(folder / "track.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), r["present"] == "1", float(r["area"]), r["reaction"],
                         float(r["activity_z"]) if r["activity_z"] else float("nan"), r["face"] == "1"))
    return rows


def _merge_budget(tracks: dict) -> dict:
    """Time share per behaviour, using in each 0.5 s slot the camera that saw the cat best."""
    names = {"resting": "平静", "active": "活跃 / 探索", "alert": "警觉", "ears_back": "耳朵后压",
             "freeze": "僵住", "unknown": "看不清"}
    slots: dict[int, list] = {}
    for rows in tracks.values():
        for t, present, _, reaction, _, face in rows:
            if reaction != "calibrating":
                slots.setdefault(int(t * 2), []).append((present, face, reaction))
    counts: dict[str, int] = {}
    for obs in slots.values():
        best = max(obs, key=lambda o: (o[0], o[1], o[2] not in ("unknown",)))
        k = names.get(best[2], best[2]) if best[0] else "不在画面里"
        counts[k] = counts.get(k, 0) + 1
    n = max(sum(counts.values()), 1)
    return {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def _merge_rounds(cam_dirs: dict, tracks: dict, actions: list, moving: set) -> list:
    """Robot rounds judged from the fixed camera that saw the cat most in the 10 s after each action.

    Cameras on the robot are left out: their picture moves with the robot, so they cannot
    tell the cat approaching from the robot approaching.
    """
    tracks = {cid: tr for cid, tr in tracks.items() if cid not in moving}
    per_cam = {}
    for cid, d in cam_dirs.items():
        if cid in moving:
            continue
        rec = EventRecorder(str(d), 30.0, robot_log=actions)
        rec.track = tracks[cid]
        per_cam[cid] = {rt: (rt, a, r) for rt, a, r in rec.robot_rounds()}
    out = []
    for rt, a in actions:
        seen = {cid: sum(1 for r in tracks[cid] if rt < r[0] <= rt + 10 and r[1]) for cid in tracks}
        cid = max(seen, key=seen.get) if seen else None
        if cid is None or seen[cid] == 0 or rt not in per_cam[cid]:
            out.append((rt, a, "不在画面里"))
        else:
            out.append(per_cam[cid][rt])
    return out


def _merge_events(cam_dirs: dict, names: dict, report: Path) -> list:
    (report / "img").mkdir(parents=True, exist_ok=True)
    (report / "clips").mkdir(exist_ok=True)
    allowed = {f.name for f in fields(Event)}
    evs = []
    for cid, d in cam_dirs.items():
        for raw in json.loads((d / "events.json").read_text()):
            ev = Event(**{k: v for k, v in raw.items() if k in allowed})
            for attr in ("before_img", "after_img", "clip"):
                src = d / getattr(ev, attr)
                dst = Path(Path(getattr(ev, attr)).parent.name) / f"{cid}_{Path(getattr(ev, attr)).name}"
                if src.exists():
                    shutil.copy2(src, report / dst)
                setattr(ev, attr, str(dst))
            ev.camera, ev.alt_views = names[cid], []
            evs.append(ev)
    evs.sort(key=lambda e: e.start_s)

    def score(ev):
        g = ev.interp or {}
        return (bool(g.get("show")), BAND_RANK.get(g.get("band"), -1), g.get("top_p", 0))

    merged, used = [], set()
    for i, ev in enumerate(evs):
        if i in used:
            continue
        group = [i] + [j for j in range(i + 1, len(evs)) if j not in used and evs[j].camera != ev.camera
                       and evs[j].start_s - ev.start_s <= MERGE_S]
        seen_cams, members = set(), []
        for j in group:  # at most one event per camera in a group
            if evs[j].camera not in seen_cams:
                seen_cams.add(evs[j].camera)
                members.append(j)
        used.update(members)
        best = max(members, key=lambda j: score(evs[j]))
        lead = evs[best]
        lead.alt_views = [(evs[j].camera, evs[j].clip) for j in members if j != best]
        g = lead.interp or {}
        for j in members:  # two views that disagree lower the confidence instead of hiding the conflict
            other = evs[j].interp or {}
            if j != best and g.get("show") and other.get("show") and other.get("top") != g.get("top"):
                text = (STATE_TEXT.get(other["top"], other["top"]))
                lead.notes.append(f"另一个视角（{evs[j].camera}）看起来更像“{text}”，两个视角不一致，把握已降低")
                g["band"] = {"高": "中", "中": "低"}.get(g["band"], "低")
                g["show"] = g["band"] != "低"
        merged.append(lead)
    return merged


def _checks(s: dict, tracks: dict, actions: list, duration: float) -> list[str]:
    """Foreseeable problems, written up front in the report so nobody over-reads it."""
    alerts = []
    slots = {int(t * 2) for rows in tracks.values() for t, present, *_ in rows if present}
    coverage = len(slots) / max(int(duration * 2), 1)
    if coverage < MIN_COVERAGE:
        alerts.append(f"这段时间里只有 {coverage:.0%} 的时间拍到了猫，其余时间它可能在镜头外。"
                      "时间占比和推测只代表拍到的部分。")
    for v in s["videos"]:
        if v["how"].startswith("未知"):
            alerts.append(f"无法确定“{v['name']}”的开始时间，它和其他视频、小车日志的时间可能对不上。"
                          "请在 session.json 里填写这段视频的 start 或 offset。")
        elif "不可靠" in v["how"]:
            alerts.append(f"“{v['name']}”的声音对齐不可靠，时间以文件记录为准，可能有几秒误差。")
    for t, a in actions:
        if not any(v["offset"] <= t <= v["offset"] + v["duration"] for v in s["videos"]):
            alerts.append(f"小车动作“{a}”发生时没有任何摄像头在录，这一轮无法判断猫的反应。")
    for v in s["videos"]:
        rows = tracks[v["id"]]
        calib_end = next((t for t, _, _, r, *_ in rows if r != "calibrating"), None)
        if calib_end is not None and any(v["offset"] <= t < calib_end for t, _ in actions):
            alerts.append(f"“{v['name']}”还在校准猫的平常状态时，小车已经动了，这个视角的基线可能不准。"
                          "测试时请先让猫安静地被拍几分钟，再启动小车。")
    return alerts


def run(args) -> Path:
    folder = Path(args.session).resolve()
    s = load_session(folder, not args.no_audio_sync, args.max_lag)
    actions, stops, warn = load_robot_log(folder / s["robot_log"] if s["robot_log"] else None, s["t0"],
                                          s["clap_s"])
    report = Path(args.out).resolve() if args.out else folder / "report"
    work = report / "_work"
    work.mkdir(parents=True, exist_ok=True)
    sync = {"session": s["name"], "t0": s["t0"].isoformat() if s["t0"] else None,
            "videos": [{"name": v["name"], "file": v["path"].name, "offset_s": round(v["offset"], 3),
                        "duration_s": round(v["duration"], 1), "how": v["how"]} for v in s["videos"]],
            "robot_actions": len(actions), "robot_stops": len(stops), "warnings": warn}
    (report / "sync.json").write_text(json.dumps(sync, ensure_ascii=False, indent=1))
    print(json.dumps(sync, ensure_ascii=False, indent=1))
    if args.dry_run:
        return report / "sync.json"
    robot_csv = work / "robot_session.csv"
    with open(robot_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "action"])
        w.writerows([(f"{t:.3f}", a) for t, a in actions])
    cam_dirs, failed = {}, []
    for v in s["videos"]:
        d = _analyse(v, work, robot_csv, args)
        if d is None:
            failed.append(v)
        else:
            cam_dirs[v["id"]] = d
    if not cam_dirs:
        raise SystemExit("none of the videos could be analysed; check that the files open in a video player")
    s["videos"] = [v for v in s["videos"] if v["id"] in cam_dirs]
    names = {v["id"]: v["name"] for v in s["videos"]}
    tracks = {cid: _load_track(d) for cid, d in cam_dirs.items()}
    events = _merge_events(cam_dirs, names, report)
    budget = _merge_budget(tracks)
    rounds = _merge_rounds(cam_dirs, tracks, actions, {v["id"] for v in s["videos"] if v["moving"]})
    summaries = [json.loads((d / "summary.json").read_text()) for d in cam_dirs.values()]
    duration = max(v["offset"] + v["duration"] for v in s["videos"])
    alerts = [f"“{v['name']}”（{v['path'].name}）无法读取，已跳过，这份报告只基于其余视频。" for v in failed]
    alerts += _checks(s, tracks, actions, duration)
    info = [f"{v['name']}（{v['path'].name}）：从 {v['offset']:.1f} 秒开始，时长 {v['duration']:.0f} 秒；时间来源：{v['how']}"
            for v in s["videos"]] + warn
    cfg = load_config(args.interp_config)
    html_text = render_owner_report(events, s["name"], duration, report, budget, rounds, cfg, {},
                                    sum(x["n_brief"] for x in summaries), any(x["has_audio"] for x in summaries),
                                    clock0=s["t0"], session_info=info, stops=stops, alerts=alerts)
    (report / "report.html").write_text(html_text, encoding="utf-8")
    clock = (lambda t: (s["t0"] + timedelta(seconds=t)).strftime("%H:%M:%S")) if s["t0"] else (lambda t: f"{t:.1f}s")
    (report / "events.json").write_text(json.dumps([{**e.__dict__} for e in events], ensure_ascii=False, indent=1))
    with open(report / "labels_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "clock", "camera", "reaction", "system_guess", "confidence", "your_label", "notes"])
        for i, e in enumerate(events):
            g = e.interp or {}
            w.writerow([i, clock(e.start_s), e.camera, e.reaction, g.get("top", ""), g.get("band", ""), "", ""])
    with open(report / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_s", "clock", "source", "what"])
        rows = [(e.start_s, e.camera, f"{e.reaction} → {(e.interp or {}).get('top', '')}") for e in events]
        rows += [(t, "robot", a) for t, a in actions] + [(t, "robot", f"STOP {d}") for t, d in stops]
        for t, src, what in sorted(rows):
            w.writerow([f"{t:.2f}", clock(t), src, what])
    print(f"\nreport: {report / 'report.html'}  ({len(events)} moments from {len(cam_dirs)} video(s))")
    return report / "report.html"


SESSION_TEMPLATE = {
    "name": "用户测试 1 · 主人不在家",
    "start": "2026-10-20T14:00:00",
    "videos": [
        {"path": "room.mp4", "name": "房间摄像头"},
        {"path": "robot_cam.mp4", "name": "小车摄像头", "moving": True},
    ],
    "robot_log": "robot_log.csv",
    "clap": {"video": "room.mp4", "s": None},  # fill in the second of the clap after watching the video
}
ROBOT_TEMPLATE = """time,event,detail
2026-10-20T14:15:00,slow_prey,第 1 轮
2026-10-20T14:15:40,round_end,
2026-10-20T14:30:00,light,第 2 轮
2026-10-20T14:30:12,safety_stop,猫耳朵压平，远程停止
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse one test session (videos + robot log) into one owner report")
    p.add_argument("session", help="session folder")
    p.add_argument("--init", action="store_true",
                   help="write session.json and example_robot_log.csv templates and exit")
    p.add_argument("--dry-run", action="store_true", help="only work out the time alignment (sync.json)")
    p.add_argument("--out", default=None, help="report folder (default: <session>/report)")
    p.add_argument("--calib-frames", type=int, default=300)
    p.add_argument("--interp-config", default=None)
    p.add_argument("--cats", default=None, help="cats.json (catpose.identity discover): tell cats apart by colour")
    p.add_argument("--cat", default=None, help="with --cats: build the report for this cat only")
    p.add_argument("--no-audio-sync", action="store_true", help="trust file times / session.json only")
    p.add_argument("--max-lag", type=float, default=120.0, help="audio sync search range around the file times (s)")
    args = p.parse_args()
    if args.init:
        folder = Path(args.session)
        folder.mkdir(parents=True, exist_ok=True)
        # The robot log example gets its own name: robot_bridge appends to robot_log.csv, and
        # example rows left in there would be counted as real robot actions.
        for name, text in (("session.json", json.dumps(SESSION_TEMPLATE, ensure_ascii=False, indent=1)),
                           ("example_robot_log.csv", ROBOT_TEMPLATE)):
            if not (folder / name).exists():
                (folder / name).write_text(text, encoding="utf-8")
        print(f"templates written to {folder}")
        return
    run(args)


if __name__ == "__main__":
    main()
