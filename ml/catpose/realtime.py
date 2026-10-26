"""Real-time cat face + body landmark demo over webcam or a video file.

Two OpenCV windows by default:
- `catpose · camera`  — the live video with landmarks overlaid.
- `catpose · data`    — a black panel with detection stats, timings, and the
                        first few keypoint coordinates. Toggle with `--no-data`.

Optional YOLO auto-detection re-locates the cat every N frames (`--detect`
enables it; `--detect-every` sets the cadence). Without `--detect`, the whole
frame (or a static `--bbox`) is used as the crop.

Examples:
    # Default: camera 0, both models, no auto-detect
    python -m catpose.realtime

    # Recommended: auto-detect cat + data window
    python -m catpose.realtime --detect

    # Video file with annotated output, headless
    python -m catpose.realtime --source clip.mp4 --output out.mp4 --no-display

Geometric features (ear height, ear angle, eye aperture, muzzle ratio, head
drop, activity; see catpose/features.py) are shown in the data window. The
first `--calib-frames` frames with a visible cat become this cat's baseline,
so keep the cat calm then; press `b` to recalibrate. They describe the face
and body, they are not an emotion classifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .crop import face_bbox_from_body_keypoints, face_bbox_from_face_landmarks
from .smoothing import OneEuro
from .draw import overlay_cv2
from .events import EventRecorder, audio_loudness
from .interpret import load_config as load_interp_config
from .reactions import COLORS as REACTION_COLORS, LABELS as REACTION_LABELS, ReactionCoder
from .features import EAR_BASE_L, EAR_BASE_R, EAR_TIP_L, EAR_TIP_R, EYE_L, EYE_R, FEATURE_NAMES, RELIABILITY, \
    FeatureTracker
from .predictor import LandmarkPredictor
from .schema import ANIMALPOSE_KEYPOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_H = 1000


def _parse_bbox(raw: Optional[str]):
    if not raw:
        return None
    parts = [float(x) for x in raw.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x0,y0,x1,y1")
    return tuple(parts)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_cat(cats: list, prev_bbox) -> Optional[object]:
    """Stay on the cat we were following: the detection overlapping the last box wins;
    only without overlap (first frame, cat lost) fall back to the most confident one."""
    if not cats:
        return None
    if prev_bbox is not None:
        best = max(cats, key=lambda d: _iou(d.bbox, prev_bbox))
        if _iou(best.bbox, prev_bbox) >= 0.1:
            return best
    return max(cats, key=lambda d: d.confidence)


def is_live(source: str) -> bool:
    """Camera index or network stream (phone app, RTSP); anything else is a video file."""
    return source.isdigit() or "://" in source


class LatestFrame:
    """Reads a live source on a thread and hands out only the newest frame.

    Without this, frames queue up whenever the models are slower than the camera
    and the picture falls further and further behind reality.
    """

    def __init__(self, cap: cv2.VideoCapture, record: Optional[str] = None) -> None:
        self.cap, self.frame, self.ok, self.seq = cap, None, True, 0
        self.running = True  # cleared by release(); `ok` is the stream's own state
        self.cond = threading.Condition()
        self.record, self.writer, self.written, self.t0 = record, None, 0, 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.rec_fps = fps if 5 <= fps <= 60 else 30.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _save(self, frame: np.ndarray) -> None:
        """Write every camera frame at a fixed rate, repeating or skipping frames to follow
        the wall clock, so the recording keeps real time even when the stream stutters."""
        now = time.time()
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), self.rec_fps, (w, h))
            self.t0 = now
        due = int((now - self.t0) * self.rec_fps) + 1
        while self.written < due:
            self.writer.write(frame)
            self.written += 1

    def _run(self) -> None:
        while self.running and self.ok:
            ok, frame = self.cap.read()
            if ok and self.record:
                self._save(frame)
            with self.cond:
                self.ok, self.frame = ok, frame
                self.seq += 1
                self.cond.notify_all()
        if self.writer is not None:
            self.writer.release()

    def read(self, seen: list) -> tuple[bool, Optional[np.ndarray]]:
        """Wait for a frame newer than seen[0] (10 s timeout = stream lost)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq != seen[0] or not self.ok, timeout=10):
                return False, None
            seen[0] = self.seq
            return self.ok, self.frame

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=3)  # the reader closes the recording on its way out
        self.cap.release()


def list_cameras() -> None:
    """Print the cameras macOS sees (iPhone Continuity Camera shows up here) with their indices."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    video = out.split("video devices:")[-1].split("audio devices:")[0]
    for line in video.splitlines():
        if "] [" in line:
            print("  --source " + line.split("] [", 1)[1].replace("] ", "   "))


def open_source(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        hint = ("  For a phone stream, check that the phone and this computer are on the same Wi-Fi "
                "and that the address opens in a browser." if "://" in source else
                "  Run with --list-cameras to see the camera numbers." if source.isdigit() else "")
        raise SystemExit(f"cannot open video source: {source}{hint}")
    return cap


def _fmt_time(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(seconds))


def _render_data_panel(
    width: int,
    height: int,
    detected: dict,
    face_pts,
    body_pts,
    timings_ms: dict,
    ema_fps: float,
    frame_count: int,
    bbox_stale: bool,
    no_cat_streak: int,
    face_conf=None,
    body_conf=None,
    min_conf: float = 0.25,
    features: Optional[dict] = None,
    baseline=None,
    coder=None,
) -> np.ndarray:
    """Draw a text-only OpenCV panel that summarizes the last frame.

    All text is ASCII to stay within FONT_HERSHEY_SIMPLEX's glyph set —
    unicode characters like the middle dot render as boxes / question marks.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (22, 25, 35)

    def put(text, y, size=0.55, weight=1, color=(230, 230, 230), x=16):
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)

    put("CatPose | live data", 32, size=0.65, weight=2, color=(240, 240, 245))
    put(f"time {_fmt_time(time.time())}  frame #{frame_count}", 56, size=0.5, color=(150, 160, 175))
    cv2.line(panel, (16, 66), (width - 16, 66), (60, 68, 90), 1)

    label = detected.get("label") if detected else "n/a"
    conf = detected.get("confidence") if detected else None
    conf_txt = f"{conf * 100:.1f}%" if conf is not None else "n/a"

    put("DETECTED", 92, size=0.45, color=(150, 160, 175))
    label_color = (245, 158, 11) if bbox_stale else (56, 189, 248)
    put(str(label), 118, size=0.85, weight=2, color=label_color if detected else (100, 110, 125))
    put("CONFIDENCE", 92, size=0.45, color=(150, 160, 175), x=width // 2 + 8)
    put(conf_txt, 118, size=0.85, weight=2,
        color=(16, 185, 129) if detected else (100, 110, 125), x=width // 2 + 8)

    put("SERVER TIMING", 152, size=0.45, color=(150, 160, 175))
    put(f"detect      {timings_ms.get('detector', 0):5.0f} ms", 178, size=0.55)
    put(f"face+body   {timings_ms.get('landmarks', 0):5.0f} ms", 202, size=0.55)
    put(f"total       {timings_ms.get('total', 0):5.0f} ms    fps {ema_fps:5.1f}",
        226, size=0.55, color=(245, 158, 11))

    put("DETECTION", 262, size=0.45, color=(150, 160, 175))
    if detected:
        bbox = detected.get("bbox", [])
        bbox_txt = "[" + ", ".join(f"{v:.0f}" for v in bbox) + "]"
        if bbox_stale:
            bbox_txt += "  (stale)"
    else:
        bbox_txt = "no cat found"
    put(f"bbox  {bbox_txt}", 286, size=0.5, color=(200, 210, 220))
    def shown(pts, conf, total):
        if pts is None:
            return f"0 / {total}"
        if conf is None:
            return f"{total} / {total}"
        return f"{int((conf >= min_conf).sum())} / {total}   mean conf {conf.mean() * 100:.0f}%"

    put(f"face shown   {shown(face_pts, face_conf, 48)}", 308, size=0.5, color=(200, 210, 220))
    put(f"body shown   {shown(body_pts, body_conf, 20)}", 328, size=0.5, color=(200, 210, 220))
    put(f"no-cat streak    {no_cat_streak}", 348, size=0.5,
        color=(245, 158, 11) if no_cat_streak > 0 else (200, 210, 220))

    put("BODY KEYPOINTS (first 8)          x      y   conf", 382, size=0.45, color=(150, 160, 175))
    y0 = 406
    if body_pts is not None:
        for i, (x, y) in enumerate(body_pts[:8]):
            name = ANIMALPOSE_KEYPOINT_NAMES[i] if i < len(ANIMALPOSE_KEYPOINT_NAMES) else str(i)
            c = None if body_conf is None else float(body_conf[i])
            color = (200, 210, 220) if c is None else (
                (16, 185, 129) if c >= 0.5 else (11, 158, 245) if c >= min_conf else (110, 115, 125))
            c_txt = "  -" if c is None else f"{c * 100:3.0f}%"
            put(f"{i + 1:2d}  {name:<18} {x:6.0f} {y:6.0f}  {c_txt}", y0 + i * 22, size=0.45, color=color)
    else:
        put("body model idle (no cat detected)", y0, size=0.45, color=(150, 160, 175))

    put("GEOMETRIC FEATURES         value     z vs own baseline", 596, size=0.45, color=(150, 160, 175))
    if baseline is not None and baseline.collecting:
        put(f"calibrating baseline {baseline.progress * 100:3.0f}%  (keep the cat calm)", 620, size=0.45,
            color=(245, 158, 11))
    elif baseline is not None:
        put("baseline ready  (press b to recalibrate)", 620, size=0.45, color=(16, 185, 129))
    labels = {"ear_height": "ear tip height", "ear_angle": "ear angle (deg)", "eye_aperture": "eye aperture",
              "muzzle_ratio": "muzzle ratio", "head_drop": "head drop", "activity": "activity"}
    for i, k in enumerate(FEATURE_NAMES):
        y = 648 + i * 26
        val, z = (features or {}).get(k, (float("nan"), float("nan"), None))[:2]
        aux = RELIABILITY.get(k, 0) >= 0.3
        name = labels[k] + ("*" if aux else "")
        if not np.isfinite(val):
            put(f"{name:<17}      -", y, size=0.45, color=(110, 115, 125))
            continue
        put(f"{name:<17} {val:7.2f}", y, size=0.45, color=(200, 210, 220))
        if np.isfinite(z):
            color = (11, 158, 245) if abs(z) >= 2 else (200, 210, 220)
            put(f"{z:+5.1f}", y, size=0.45, color=color, x=268)
            cx, half = 380, 60  # bar: +-3 SD across 120 px
            cv2.line(panel, (cx - half, y - 5), (cx + half, y - 5), (60, 68, 90), 1)
            cv2.line(panel, (cx, y - 11), (cx, y + 1), (90, 98, 120), 1)
            end = int(cx + np.clip(z / 3, -1, 1) * half)
            cv2.rectangle(panel, (min(cx, end), y - 9), (max(cx, end), y - 2), color, -1)
    put("* noisier feature: supporting evidence only",
        648 + len(FEATURE_NAMES) * 26, size=0.4, color=(120, 130, 145))
    put("features describe the face and body; they are NOT an emotion label",
        668 + len(FEATURE_NAMES) * 26, size=0.4, color=(120, 130, 145))

    if coder is not None:
        y = 866
        cv2.line(panel, (16, y - 26), (width - 16, y - 26), (60, 68, 90), 1)
        put("REACTION  (rule-based, behaviour not emotion)", y, size=0.45, color=(150, 160, 175))
        r = coder.current
        put(r.text, y + 28, size=0.7, weight=2, color=REACTION_COLORS[r.key])
        put("; ".join(r.evidence)[:62], y + 50, size=0.4, color=(200, 210, 220))
        for i, (stamp, key, _) in enumerate(list(coder.events)[::-1][:3]):
            put(f"{stamp}  {REACTION_LABELS[key]}", y + 74 + i * 18, size=0.4, color=REACTION_COLORS[key])

    put("press q or Esc to quit", height - 18, size=0.45, color=(120, 130, 145))
    return panel


def _draw_feature_guides(out: np.ndarray, face_pts, face_conf, min_conf: float) -> None:
    """Draw what the features measure: ear axes (yellow) and eye openings (cyan)."""
    def ok(idx):
        return face_conf is None or bool((face_conf[idx] >= min_conf).all())

    def pt(p):
        return int(p[0]), int(p[1])

    for tip, base in ((EAR_TIP_L, EAR_BASE_L), (EAR_TIP_R, EAR_BASE_R)):
        if ok([tip] + base):
            cv2.line(out, pt(face_pts[base].mean(0)), pt(face_pts[tip]), (0, 215, 255), 2, cv2.LINE_AA)
    for eye in (EYE_L, EYE_R):
        if ok(eye):
            cv2.line(out, pt(face_pts[eye[2]]), pt(face_pts[eye[3]]), (255, 220, 0), 2, cv2.LINE_AA)


def _draw_reaction_banner(out: np.ndarray, reaction) -> None:
    """Top-left banner with the current rule-based reaction and its main evidence."""
    color = REACTION_COLORS[reaction.key]
    text = "reaction: " + reaction.text
    sub = reaction.evidence[0] if reaction.evidence else ""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = 12, 78  # below the FPS counter
    cv2.rectangle(out, (x - 6, y - th - 10), (x + max(tw, sw) + 10, y + 26), (22, 25, 35), -1)
    cv2.rectangle(out, (x - 6, y - th - 10), (x - 2, y + 26), color, -1)
    cv2.putText(out, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(out, sub, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 220), 1, cv2.LINE_AA)


def _print_status_line(fields: dict) -> None:
    """Print one line to the terminal, updating in place."""
    text = " | ".join(f"{k}={v}" for k, v in fields.items())
    sys.stdout.write("\r\033[K" + text)
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="camera index, video path, or phone stream URL (e.g. http://192.168.1.5:8080/video)")
    p.add_argument("--record", default=None,
                   help="live sources: save the raw camera video here (default: <report-dir>/raw.mp4)")
    p.add_argument("--moving-camera", action="store_true",
                   help="camera mounted on the robot: do not read approach / leave from the cat's size")
    p.add_argument("--list-cameras", action="store_true", help="list camera indices (incl. iPhone) and exit")
    p.add_argument("--output", default=None, help="write annotated video to this path")
    p.add_argument("--bbox", default=None, help="static bbox x0,y0,x1,y1 (overrides detector)")
    p.add_argument("--face-ckpt", default=str(REPO_ROOT / "ml" / "checkpoints" / "face.pt"))
    p.add_argument("--body-ckpt", default=str(REPO_ROOT / "ml" / "checkpoints" / "body.pt"))
    p.add_argument("--yolo-ckpt", default=str(REPO_ROOT / "ml" / "checkpoints" / "yolov8s.pt"))
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                   help="cuda (NVIDIA GPU, e.g. Colab), mps (Apple GPU) or cpu")
    p.add_argument("--no-smooth", action="store_true", help="disable One-Euro temporal smoothing")
    p.add_argument("--no-tta", action="store_true",
                   help="skip flip test-time augmentation (about 2x faster on CPU, slightly less accurate)")
    p.add_argument("--min-conf", type=float, default=0.25, help="hide keypoints below this confidence")
    p.add_argument("--face-passes", type=int, default=2,
                   help="face passes; pass 2+ re-crops on the previous pass's landmarks")
    p.add_argument("--detect", action="store_true", help="auto-detect cat bbox each frame")
    p.add_argument("--detect-every", type=int, default=2, help="run YOLO every N frames (default 2)")
    p.add_argument("--yolo-conf", type=float, default=0.20, help="YOLO confidence threshold")
    p.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO input size (bigger = better recall)")
    p.add_argument("--bbox-hold", type=int, default=15,
                   help="Reuse the last cat bbox this many frames when YOLO briefly misses")
    p.add_argument("--skip-face", action="store_true")
    p.add_argument("--skip-body", action="store_true")
    p.add_argument("--no-data", action="store_true", help="don't open the data-panel window")
    p.add_argument("--no-print", action="store_true", help="don't print status line to terminal")
    p.add_argument("--no-display", action="store_true", help="disable all GUI windows (headless)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    p.add_argument("--no-features", action="store_true", help="skip the geometric feature section")
    p.add_argument("--calib-frames", type=int, default=300,
                   help="frames with a visible cat used as this cat's calm baseline (default 300)")
    p.add_argument("--baseline-in", default=None, help="load a saved baseline JSON instead of calibrating")
    p.add_argument("--baseline-out", default=None, help="save the baseline JSON here once calibration finishes")
    p.add_argument("--features-csv", default=None, help="append per-frame features and z-scores to this CSV")
    p.add_argument("--robot-log", default=None,
                   help="CSV of robot actions (columns: time_s, action) to use as context in the report")
    p.add_argument("--interp-config", default=None,
                   help="JSON with interpretation priors/likelihoods (default: hand-set values in catpose/interpret.py)")
    p.add_argument("--time-offset", type=float, default=0.0,
                   help="seconds added to this video's clock so several videos share one session timeline")
    p.add_argument("--report-dir", default=None,
                   help="save state-change events and an image + text report.html in this folder")
    p.add_argument("--cats", default=None,
                   help="cats.json from `python -m catpose.identity discover`: tell cats apart by coat colour")
    p.add_argument("--cat", default=None,
                   help="with --cats: follow only this cat (a name in cats.json); the other cat is ignored")
    p.add_argument("--reaction-hold", type=float, default=0.3,
                   help="seconds a new reaction must dominate before it is shown (default 0.3)")
    args = p.parse_args()
    if args.list_cameras:
        list_cameras()
        return

    static_bbox = _parse_bbox(args.bbox)

    cat_book = None
    if args.cats or args.cat:
        if not (args.cats and args.cat):
            raise SystemExit("--cats and --cat go together: --cats cats.json --cat NAME")
        from .identity import CatBook

        cat_book = CatBook.load(args.cats)
        if args.cat not in cat_book.names:
            raise SystemExit(f"unknown cat '{args.cat}'; cats.json has: {', '.join(cat_book.names)}")
        print(f"[info] following the {args.cat} cat only (colour match; cats: {', '.join(cat_book.names)})")

    face_predictor = body_predictor = detector = None
    if not args.skip_face:
        if not Path(args.face_ckpt).exists():
            print(f"[warn] face checkpoint missing: {args.face_ckpt} — disabling face model")
        else:
            face_predictor = LandmarkPredictor("face", args.face_ckpt, device=args.device)
    if not args.skip_body:
        if not Path(args.body_ckpt).exists():
            print(f"[warn] body checkpoint missing: {args.body_ckpt} — disabling body model")
        else:
            body_predictor = LandmarkPredictor("body", args.body_ckpt, device=args.device)
    if face_predictor is None and body_predictor is None:
        raise SystemExit("no models loaded; train first or point --face-ckpt/--body-ckpt at valid files")
    for predictor in (face_predictor, body_predictor):
        if predictor is not None:
            predictor.default_tta = not args.no_tta

    if args.detect and static_bbox is None:
        if not Path(args.yolo_ckpt).exists():
            print(f"[warn] YOLO weights missing: {args.yolo_ckpt} — falling back to whole frame")
        else:
            from .detector import CatDetector

            detector = CatDetector(
                weights=args.yolo_ckpt,
                device=args.device,
                conf_threshold=args.yolo_conf,
                imgsz=args.yolo_imgsz,
            )
            print(
                f"[info] YOLO loaded ({args.yolo_ckpt})  "
                f"conf>={args.yolo_conf}  imgsz={args.yolo_imgsz}  hold={args.bbox_hold} frames"
            )

    cap = open_source(args.source)
    live = is_live(args.source)
    if live:
        record = args.record or (str(Path(args.report_dir) / "raw.mp4") if args.report_dir else None)
        if record:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            print(f"[info] recording the camera to {record} (re-analyse it later as a file)")
        cap = LatestFrame(cap, record)
        seen = [0]
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_source or 20.0, (width, height))

    display = not args.no_display and (os.environ.get("DISPLAY") or sys.platform == "darwin")
    show_data = display and not args.no_data
    print_status = not args.no_print

    if display:
        cv2.namedWindow("catpose | camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("catpose | camera", 720, int(720 * height / max(width, 1)))
    if show_data:
        cv2.namedWindow("catpose | data", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("catpose | data", 460, PANEL_H)

    ema_fps = 0.0
    frame_count = 0
    cached_bbox: Optional[tuple] = static_bbox
    cached_detection: dict = {}
    frames_since_detection = 10 ** 9  # how many frames since the last real YOLO hit
    no_cat_streak = 0  # how many recent YOLO passes returned nothing
    body_filter, face_filter = OneEuro(), OneEuro()
    t_prev = time.time()

    tracker = None if args.no_features else FeatureTracker(args.calib_frames, min_conf=args.min_conf)
    coder = None if tracker is None else ReactionCoder(hold_s=args.reaction_hold)
    is_file = not live
    recorder = None
    if coder is not None and args.report_dir:
        audio = audio_loudness(args.source, fps_source or 30.0) if is_file else None
        robot_log = None
        if args.robot_log:
            with open(args.robot_log, newline="", encoding="utf-8") as f:
                robot_log = [(float(r["time_s"]), r["action"]) for r in csv.DictReader(f)]
        recorder = EventRecorder(args.report_dir, fps_source or 30.0, audio, load_interp_config(args.interp_config),
                                 robot_log, moving_camera=args.moving_camera)
        print(f"[info] event report -> {args.report_dir}  (audio track: {'yes' if audio is not None else 'no'})")
    t_start, t_video = time.time(), 0.0
    if tracker is not None and args.baseline_in:
        tracker.baseline.load(Path(args.baseline_in))
        print(f"[info] baseline loaded from {args.baseline_in}")
    baseline_saved = False
    features: dict = {}
    csv_file = csv_writer = None
    if tracker is not None and args.features_csv:
        new_file = not Path(args.features_csv).exists()
        csv_file = open(args.features_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(["time", "frame"] + [f"{k}{suffix}" for k in FEATURE_NAMES for suffix in ("", "_z")]
                                + ["reaction"])

    try:
        while True:
            ok, frame_bgr = cap.read(seen) if live else cap.read()
            if not ok:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_size = (rgb.shape[1], rgb.shape[0])
            # Video files use their own clock so smoothing (and every report) is reproducible.
            t_src = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else None

            t_det_start = time.time()
            if detector is not None and static_bbox is None and (
                frame_count == 1 or frame_count % max(args.detect_every, 1) == 0
            ):
                dets = detector.detect(rgb)
                best = (pick_cat(dets, cached_bbox) if cat_book is None
                        else cat_book.pick(dets, frame_bgr, args.cat, cached_bbox))
                if best is not None:
                    cached_bbox = best.bbox
                    cached_detection = {
                        "label": best.label if cat_book is None else f"{best.label} ({args.cat})",
                        "confidence": best.confidence,
                        "bbox": [round(v, 1) for v in best.bbox],
                    }
                    frames_since_detection = 0
                    no_cat_streak = 0
                else:
                    no_cat_streak += 1
            t_det = (time.time() - t_det_start) * 1000

            frames_since_detection += 1
            bbox_stale = frames_since_detection > args.detect_every
            # After bbox_hold frames without a hit, drop the cached bbox.
            if cached_bbox is not None and static_bbox is None and detector is not None:
                if frames_since_detection > args.bbox_hold:
                    cached_bbox = None
                    cached_detection = {}

            bbox_to_use = static_bbox if static_bbox is not None else cached_bbox

            face_pts = body_pts = face_conf = body_conf = None
            face_bbox = None
            t_kp_start = time.time()
            # Only run the landmark models when we actually have a cat bbox to
            # crop into. Without one, the models were producing noise across
            # the whole frame (TV, cabinet, background) which read as broken.
            if bbox_to_use is not None:
                box_scale = max(bbox_to_use[2] - bbox_to_use[0], bbox_to_use[3] - bbox_to_use[1])
                if body_predictor is not None:
                    pred = body_predictor.predict(rgb, bbox=bbox_to_use)
                    body_pts, body_conf = pred.landmarks, pred.confidences
                    if not args.no_smooth:
                        body_pts = body_filter(body_pts, t=t_src, scale=box_scale)
                if face_predictor is not None:
                    face_input_bbox = bbox_to_use
                    if body_pts is not None:
                        derived = face_bbox_from_body_keypoints(body_pts, frame_size, conf=body_conf)
                        if derived is not None:
                            face_input_bbox = derived
                        elif body_conf is not None:
                            face_input_bbox = None  # head not confidently located: skip face
                    if face_input_bbox is not None:
                        pred = face_predictor.predict(rgb, bbox=face_input_bbox)
                        for _ in range(args.face_passes - 1):
                            refined = face_bbox_from_face_landmarks(pred.landmarks, pred.confidences)
                            if refined is None:
                                break
                            pred = face_predictor.predict(rgb, bbox=refined)
                        face_pts, face_conf, face_bbox = pred.landmarks, pred.confidences, pred.bbox
                        if not args.no_smooth:
                            face_pts = face_filter(face_pts, t=t_src, scale=face_bbox[2] - face_bbox[0])
                    else:
                        face_filter.reset()
            else:
                body_filter.reset()
                face_filter.reset()
            t_kp = (time.time() - t_kp_start) * 1000

            features = {}
            if tracker is not None and (face_pts is not None or body_pts is not None):
                scale = max(bbox_to_use[2] - bbox_to_use[0], bbox_to_use[3] - bbox_to_use[1])
                t_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else None  # reproducible on files
                features = tracker.update(face_pts, face_conf, body_pts, body_conf, scale, t=t_frame)
                if is_file:
                    sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    stamp = f"video {int(sec // 60):02d}:{sec % 60:04.1f}"
                else:
                    stamp = time.strftime("%H:%M:%S")
                coder.update(features, tracker.baseline.collecting, stamp, tracker.pose_note,
                             t=t_frame if t_frame is not None else time.time())
                if args.baseline_out and not tracker.baseline.collecting and not baseline_saved:
                    tracker.baseline.save(Path(args.baseline_out))
                    baseline_saved = True
                    print(f"\n[info] baseline saved to {args.baseline_out}")
                if csv_writer is not None:
                    csv_writer.writerow([f"{time.time():.3f}", frame_count] + [
                        "" if not np.isfinite(v) else f"{v:.4f}" for k in FEATURE_NAMES for v in features[k][:2]]
                        + [coder.current.key])
            elif tracker is not None:
                tracker.activity.reset()
                # Cat lost: let the label fall back to "unknown" instead of freezing on the last one.
                coder.update({}, tracker.baseline.collecting, None, "cat not in view",
                             t=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.time())

            t_now = time.time()
            dt = max(t_now - t_prev, 1e-3)
            fps = 1.0 / dt
            ema_fps = fps if frame_count == 1 else 0.9 * ema_fps + 0.1 * fps
            t_prev = t_now

            out = overlay_cv2(
                frame_bgr,
                face_pts=face_pts,
                body_pts=body_pts,
                face_bbox=face_bbox,
                fps=ema_fps,
                face_conf=face_conf,
                body_conf=body_conf,
                min_conf=args.min_conf,
            )
            if face_pts is not None and features:
                _draw_feature_guides(out, face_pts, face_conf, args.min_conf)
            if coder is not None and cached_detection:
                _draw_reaction_banner(out, coder.current)
            if cached_detection:
                x0, y0, x1, y1 = (int(v) for v in cached_detection["bbox"])
                color = (16, 185, 129) if not bbox_stale else (24, 132, 245)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
                tag = f"{cached_detection['label']} {cached_detection['confidence'] * 100:.0f}%"
                if bbox_stale:
                    tag += f" (stale {frames_since_detection})"
                cv2.putText(out, tag, (x0, max(y0 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2, cv2.LINE_AA)
            elif detector is not None:
                # No cat right now — say so on the camera view instead of
                # drawing garbage keypoints on the background.
                h, w = out.shape[:2]
                msg1 = "no cat detected"
                msg2 = "center the cat, or press q to quit"
                cv2.putText(out, msg1, (w // 2 - 180, h // 2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 158, 11), 3, cv2.LINE_AA)
                cv2.putText(out, msg2, (w // 2 - 240, h // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 210, 220), 2, cv2.LINE_AA)

            if recorder is not None:
                t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.time() - t_start
                if live and frame_count > 30:
                    recorder.fps = max(ema_fps, 1.0)  # clips play at the speed frames were really processed
                recorder.update(t_video + args.time_offset, frame_count, frame_bgr, out, bbox_to_use, features,
                                face_pts, face_conf,
                                coder.current.key, decision=coder.current)

            if writer is not None:
                writer.write(out)

            if display:
                cv2.imshow("catpose | camera", out)
            if show_data:
                panel = _render_data_panel(
                    width=460,
                    height=PANEL_H,
                    detected=cached_detection,
                    face_pts=face_pts,
                    body_pts=body_pts,
                    timings_ms={"detector": t_det, "landmarks": t_kp, "total": t_det + t_kp},
                    ema_fps=ema_fps,
                    frame_count=frame_count,
                    bbox_stale=bbox_stale and bool(cached_detection),
                    no_cat_streak=no_cat_streak,
                    face_conf=face_conf,
                    body_conf=body_conf,
                    min_conf=args.min_conf,
                    features=features,
                    baseline=None if tracker is None else tracker.baseline,
                    coder=coder,
                )
                cv2.imshow("catpose | data", panel)

            if print_status:
                fields = {
                    "frame": frame_count,
                    "fps": f"{ema_fps:.1f}",
                    "det": cached_detection.get("label", "-"),
                    "conf": (f"{cached_detection['confidence'] * 100:.0f}%"
                             if cached_detection else "-"),
                    "stale": frames_since_detection if cached_detection else "-",
                    "kp_face": 48 if face_pts is not None else 0,
                    "kp_body": 20 if body_pts is not None else 0,
                    "t_yolo": f"{t_det:.0f}ms",
                    "t_kp": f"{t_kp:.0f}ms",
                }
                if coder is not None:
                    fields["reaction"] = coder.current.key
                _print_status_line(fields)

            if display:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("b") and tracker is not None:
                    tracker.baseline.restart()
                    baseline_saved = False
                    print("\n[info] recalibrating baseline")
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if print_status:
            sys.stdout.write("\n")
            sys.stdout.flush()

    print(f"processed {frame_count} frames, avg fps ~ {ema_fps:.1f}")
    if recorder is not None:
        report = recorder.write_report(args.source, t_video + args.time_offset)
        print(f"report: {report}  ({len(recorder.events)} events)")


if __name__ == "__main__":
    main()
