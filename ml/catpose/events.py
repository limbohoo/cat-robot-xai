"""State-change events and the owner-facing image + text report.

An event starts when the rule-based reaction (catpose/reactions.py) moves from
"no clear change" into another reaction, and ends when it moves back. For each
event we keep:

- why it got its label: the conditions the rules found (`drivers`) in the
  values the label was decided on (`z`, recorded by ReactionCoder), checked by
  catpose/faithfulness.py and circled on the after-frame;
- what changed on the cat: before/after frames, and the features whose z-score
  moved by at least `MIN_DELTA` between a window before the change and the
  moment the new reaction was confirmed;
- what happened just before: sudden loudness jumps in the audio track and
  motion in the part of the frame outside the cat (later: robot actions).

Triggers are reported as "N s before", never as causes: one co-occurrence is
not evidence of cause. Counts across events are reported in the summary.
"""

from __future__ import annotations

import json
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .faithfulness import check_events
from .features import EAR_L, EAR_R, EYE_L, EYE_R
from .interpret import DEFAULT_CONFIG, interpret
from .reactions import Reaction
from .report_ui import render_owner_report

MIN_DELTA = 1.0  # SD change needed before a feature is listed as "changed"
BEFORE_WINDOW = (-2.5, -1.0)  # seconds relative to the confirmed change
LOOKBACK_S = 3.0  # how far back to search for triggers
SOUND_JUMP_DB = 10.0  # loudness above the preceding 5 s median
MOTION_JUMP = 3.0  # background motion above this multiple of its recent median
MIN_EVENT_S = 0.5  # the live label reacts fast; the report only keeps changes that lasted this long
MERGE_GAP_S = 2.0  # the same reaction coming back within this gap continues the previous event
FAST_Z = 2.5  # activity this far above baseline counts as fast movement

FEATURE_TEXT = {  # (name, text when z went down, text when z went up)
    "ear_height": ("耳朵", "放低了", "竖起来了"),
    "ear_angle": ("耳朵", "更朝前", "向外转了"),
    "eye_aperture": ("眼睛", "眯起来了", "睁大了"),
    "muzzle_ratio": ("口鼻", "变窄了", "变宽了"),
    "head_drop": ("头", "抬高了", "放低了"),
    "activity": ("身体", "静下来了", "动起来了"),
}
FEATURE_TEXT_EN = {"ear_height": "ears", "ear_angle": "ears", "eye_aperture": "eyes"}
FEATURE_POINTS = {"ear_height": EAR_L + EAR_R, "ear_angle": EAR_L + EAR_R, "eye_aperture": EYE_L + EYE_R}


def audio_loudness(path: str, fps: float) -> Optional[np.ndarray]:
    """Per-video-frame loudness in dB from the file's audio track, or None if there is none."""
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
            capture_output=True, check=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if len(x) == 0:
        return None
    hop = 16000 / fps
    n = int(len(x) / hop)
    rms = np.array([np.sqrt(np.mean(x[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-10) for i in range(n)])
    return 20 * np.log10(rms)


@dataclass
class Event:
    start_s: float
    reaction: str
    changes: list = field(default_factory=list)  # (feature, delta z)
    triggers: list = field(default_factory=list)  # (kind, seconds before, detail)
    notes: list = field(default_factory=list)
    end_s: Optional[float] = None
    before_img: str = ""
    after_img: str = ""
    clip: str = ""
    interp: Optional[dict] = None
    drivers: list = field(default_factory=list)  # conditions behind the label (see reactions.classify)
    z: dict = field(default_factory=dict)  # z per feature the label was decided on
    prev_reaction: str = ""  # label shown before (decides which thresholds applied)
    faith: Optional[dict] = None  # result of catpose.faithfulness.check_event


class EventRecorder:
    def __init__(self, out_dir: str, fps: float, audio_db: Optional[np.ndarray] = None,
                 interp_cfg: Optional[dict] = None, robot_log: Optional[list] = None,
                 moving_camera: bool = False) -> None:
        self.dir = Path(out_dir)
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        (self.dir / "clips").mkdir(exist_ok=True)
        self.cfg = interp_cfg or DEFAULT_CONFIG
        self.robot_log = sorted(robot_log or [])  # (t seconds, action)
        # A camera on the robot moves by itself, so the cat's size in the picture says
        # nothing about the cat approaching or leaving.
        self.moving_camera = moving_camera
        self.track: list[tuple] = []  # per frame: (t, present, area, reaction, activity z, face visible)
        self.clips: list[list] = []  # open clip writers: [writer, end_t, event]
        self.fps = max(fps, 1.0)
        self.audio = audio_db
        n = int(self.fps * (LOOKBACK_S + 3))
        self.buf: deque = deque(maxlen=n)  # (t, frame, features, face_pts, face_conf, reaction)
        self.bg: deque = deque(maxlen=n)  # (t, background motion)
        self.prev_gray: Optional[np.ndarray] = None
        self.motion_log: list[tuple[float, float]] = []
        self.last_reaction = "calibrating"
        self.events: list[Event] = []
        self.timeline: list[tuple[float, int]] = []  # (t, frame index) of every frame, for base rates
        self.open: Optional[Event] = None
        self.mp4v_clips: list[Path] = []  # browsers cannot play these; converted to H.264 at the end

    # ---------- per frame ----------
    def _background_motion(self, frame: np.ndarray, bbox) -> float:
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0
        if self.prev_gray is not None:
            diff = np.abs(small - self.prev_gray)
            if bbox is not None:  # ignore the cat itself
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = bbox
                diff[max(int(y0 / h * 90) - 4, 0):int(y1 / h * 90) + 4, max(int(x0 / w * 160) - 4, 0):int(x1 / w * 160) + 4] = 0
            motion = float(diff.mean())
        self.prev_gray = small
        return motion

    def update(self, t: float, frame_idx: int, frame: np.ndarray, annotated: np.ndarray, bbox, features: dict,
               face_pts, face_conf, reaction: str, decision: Optional[Reaction] = None) -> None:
        """`decision` is the coder's current Reaction; on a label switch it carries what the
        label was decided on (drivers, snapshot, previous label)."""
        self.bg.append((t, self._background_motion(frame, bbox), frame_idx))
        self.motion_log.append((t, self.bg[-1][1]))
        self.timeline.append((t, frame_idx))
        self.buf.append((t, annotated.copy(), dict(features), None if face_pts is None else face_pts.copy(),
                         None if face_conf is None else face_conf.copy(), reaction))
        h, w = frame.shape[:2]
        area = 0.0 if bbox is None else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(w * h)
        act = features.get("activity", (np.nan, np.nan))[1] if features else np.nan
        face = bool(features) and np.isfinite(features.get("ear_height", (np.nan, np.nan))[1])
        self.track.append((t, bbox is not None, area, reaction, act, face))
        for c in self.clips:  # keep writing open event clips
            if t <= c[1]:
                c[0].write(self._clip_frame(annotated))
        for c in [c for c in self.clips if t > c[1]]:
            c[0].release()
            self.clips.remove(c)
        if reaction != self.last_reaction:
            if self.open is not None:
                self.open.end_s = t
                self.open = None
            if reaction not in ("resting", "calibrating", "unknown") and self.last_reaction != "calibrating":
                last = self.events[-1] if self.events else None
                if (last is not None and last.reaction == reaction and last.end_s is not None
                        and t - last.end_s <= MERGE_GAP_S):
                    # same reaction back after a short gap: one event, not a string of fragments
                    last.end_s, self.open = None, last
                    for c in self.clips:
                        if c[2] is last:
                            c[1] = max(c[1], t + self.cfg["follow_s"])
                else:
                    self.open = self._start(t, frame_idx, reaction, decision)
                    self.events.append(self.open)
            self.last_reaction = reaction

    # ---------- event construction ----------
    def _start(self, t: float, frame_idx: int, reaction: str, decision: Optional[Reaction] = None) -> Event:
        ev = Event(start_s=t, reaction=reaction)
        if decision is not None and decision.key == reaction:
            ev.drivers, ev.z, ev.prev_reaction = list(decision.drivers), dict(decision.snapshot), decision.prev
        before = [b for b in self.buf if BEFORE_WINDOW[0] <= b[0] - t <= BEFORE_WINDOW[1]]
        now = self.buf[-1]
        # What changed: mean z before vs now.
        for k in FEATURE_TEXT:
            zb = [b[2][k][1] for b in before if k in b[2] and np.isfinite(b[2][k][1])]
            za = now[2].get(k, (np.nan, np.nan))[1]
            if zb and np.isfinite(za) and abs(za - np.mean(zb)) >= MIN_DELTA:
                ev.changes.append((k, float(za - np.mean(zb))))
        ev.changes.sort(key=lambda c: -abs(c[1]))
        if not any(np.isfinite(now[2].get(k, (np.nan, np.nan))[1]) for k in ("ear_height", "eye_aperture")):
            ev.notes.append("这一刻脸部看不清（被遮挡、侧脸或太远），只依据身体判断")
        # What happened before: sound jumps and background motion.
        hit = self._sound_jump(frame_idx)
        if hit is not None:
            ev.triggers.append(("sound", hit[0], "突然出现较大的声音"))
        hit = self._motion_jump(t, self.motion_log)
        if hit is not None:
            ev.triggers.append(("motion", hit, "猫以外的画面里有东西在动"))
        # Images: before (start of the before window) and now, changed regions circled.
        idx = len(self.events)
        b_frame = before[0][1] if before else self.buf[0][1]
        a_frame = now[1].copy()
        scale = max(a_frame.shape[1] / 640, 1.0)  # images are saved at 640 px wide
        if now[3] is not None:
            # circle what the label was decided on, not merely what moved (faithful by construction)
            marked = list(dict.fromkeys(d["feature"] for d in ev.drivers if d["feature"] in FEATURE_POINTS))
            for k in marked:
                pts = [now[3][i] for i in FEATURE_POINTS[k] if now[4] is None or now[4][i] >= 0.25]
                for group in ([p for p in pts[:len(pts) // 2]], [p for p in pts[len(pts) // 2:]]):
                    if group:  # one ring around each ear / each eye
                        c = np.mean(group, 0)
                        r = max(np.linalg.norm(np.array(group) - c, axis=1).max() + 12 * scale, 18 * scale)
                        cv2.circle(a_frame, (int(c[0]), int(c[1])), int(r), (60, 60, 255), int(3 * scale), cv2.LINE_AA)
            if marked:
                label = "basis: " + ", ".join(sorted({FEATURE_TEXT_EN[k] for k in marked}))
                y = a_frame.shape[0] - int(16 * scale)
                cv2.putText(a_frame, label, (int(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                            (0, 0, 0), int(5 * scale), cv2.LINE_AA)
                cv2.putText(a_frame, label, (int(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                            (60, 60, 255), int(2 * scale), cv2.LINE_AA)
        ev.clip = f"clips/e{idx:03d}.mp4"
        fh, fw = self._clip_frame(now[1]).shape[:2]
        writer = cv2.VideoWriter(str(self.dir / ev.clip), cv2.VideoWriter_fourcc(*"avc1"), self.fps, (fw, fh))
        if not writer.isOpened():  # fall back when the H.264 encoder is unavailable (e.g. pip OpenCV on Linux)
            writer = cv2.VideoWriter(str(self.dir / ev.clip), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (fw, fh))
            self.mp4v_clips.append(self.dir / ev.clip)
        for b in self.buf:  # about 2 s before the change ...
            if t - 2.0 <= b[0] <= t:
                writer.write(self._clip_frame(b[1]))
        self.clips.append([writer, t + self.cfg["follow_s"], ev])  # ... and the follow-up seconds after it
        ev.before_img, ev.after_img = f"img/e{idx:03d}_before.jpg", f"img/e{idx:03d}_after.jpg"
        for name, img in ((ev.before_img, b_frame), (ev.after_img, a_frame)):
            h, w = img.shape[:2]
            s = 640 / max(w, 1)
            cv2.imwrite(str(self.dir / name), cv2.resize(img, (640, int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return ev

    @staticmethod
    def _clip_frame(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (640, int(h * 640 / max(w, 1)) // 2 * 2))

    # ---------- follow-up cues and interpretation ----------
    def _cues(self, ev: Event) -> tuple[list[str], float]:
        t0, t1 = ev.start_s, ev.start_s + self.cfg["follow_s"]
        before = [r for r in self.track if t0 - 1.5 <= r[0] <= t0 and r[1]]
        after = [r for r in self.track if t0 < r[0] <= t1]
        if not after:
            return [], 0.0
        dt = (t1 - t0) / len(after)
        a0 = float(np.median([r[2] for r in before])) if before else None
        present = [r for r in after if r[1]]
        cues = []
        if self.robot_log and any(t0 - LOOKBACK_S <= rt <= t0 for rt, _ in self.robot_log):
            cues.append("robot_before")
        acts = [r[4] for r in present if np.isfinite(r[4])]
        fast = sum(1 for a in acts if a >= FAST_Z) * dt >= 0.5  # sustained, not one jumpy frame
        still = sum(1 for a in acts if a <= -1) * dt >= 2.0
        if not self.moving_camera:
            absent_frac = 1 - len(present) / len(after)
            end_area = float(np.median([r[2] for r in present[-max(3, len(present) // 4):]])) if present else 0.0
            # A cat that stays still but looks smaller is crouching, not leaving.
            away = absent_frac >= 0.3 or bool(a0 and present and end_area < 0.8 * a0 and not still)
            if away:
                # Moving fast while leaving is running off, not play.
                cues.append("flees" if fast else "moves_away")
            elif a0 and present and max(r[2] for r in present) >= 1.25 * a0:
                cues.append("approaches")
            if fast and not away:
                cues.append("plays")
        # On a moving camera, fast movement alone cannot tell play from running off: no cue.
        if sum(1 for r in after if r[3] == "resting") * dt >= 1.5:
            cues.append("settles")
        elif still:
            cues.append("stays_still")
        face = sum(1 for r in after if r[5]) / len(after)
        return cues, face

    def time_budget(self) -> dict:
        names = {"resting": "平静", "active": "活跃 / 探索", "alert": "警觉", "ears_back": "耳朵后压",
                 "freeze": "僵住", "unknown": "看不清"}
        rows = [r for r in self.track if r[3] != "calibrating"]
        if not rows:
            return {}
        out = {}
        for r in rows:
            k = names.get(r[3], r[3]) if r[1] else "不在画面里"
            out[k] = out.get(k, 0) + 1
        return {k: v / len(rows) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def robot_rounds(self) -> list[tuple]:
        """For each logged robot action: how the cat responded in the next 10 s."""
        if self.moving_camera:  # leaving / approaching cannot be read from a camera that moves
            return []
        out = []
        for rt, action in self.robot_log:
            win = [r for r in self.track if rt < r[0] <= rt + 10]
            pres = [r for r in win if r[1]]
            if not win:
                continue
            a0 = float(np.median([r[2] for r in self.track if rt - 2 <= r[0] <= rt and r[1]] or [0])) or None
            acts = [r[4] for r in pres if np.isfinite(r[4])]
            if (1 - len(pres) / len(win)) >= 0.5 or (a0 and pres and pres[-1][2] < 0.8 * a0):
                resp = "回避或离开"
            elif (a0 and pres and max(r[2] for r in pres) >= 1.25 * a0) or (acts and max(acts) >= FAST_Z):
                resp = "参与（靠近或扑玩）"
            else:
                resp = "没有明显反应"
            out.append((rt, action, resp))
        return out

    def _sound_jump(self, frame_idx: int):
        """(seconds before, dB jump) of the loudest jump in the LOOKBACK_S window ending at this frame."""
        if self.audio is None:
            return None
        i1 = min(frame_idx, len(self.audio) - 1)
        i0 = max(0, i1 - int(LOOKBACK_S * self.fps))
        ref = self.audio[max(0, i0 - int(5 * self.fps)):max(i0, 1)]
        if not len(ref):
            return None
        peak = i0 + int(np.argmax(self.audio[i0:i1 + 1]))
        jump = self.audio[peak] - float(np.median(ref))
        return ((i1 - peak) / self.fps, jump) if jump >= SOUND_JUMP_DB else None

    @staticmethod
    def _motion_jump(t: float, log):
        """Seconds before t of a background-motion spike in the LOOKBACK_S window, else None."""
        ref = [m for (tb, m) in log if t - LOOKBACK_S - 5 <= tb < t - LOOKBACK_S]
        recent = [(tb, m) for (tb, m) in log if t - LOOKBACK_S <= tb <= t]
        if not ref or not recent:
            return None
        tb, m = max(recent, key=lambda r: r[1])
        return t - tb if m >= MOTION_JUMP * max(float(np.median(ref)), 0.5) else None

    def base_rates(self) -> dict:
        """Share of all 3 s windows (sampled every second) that contain each kind of cue."""
        if not self.timeline:
            return {}
        samples = [self.timeline[i] for i in range(0, len(self.timeline), max(int(self.fps), 1))
                   if self.timeline[i][0] >= LOOKBACK_S + 5]
        if not samples:
            return {}
        rates = {"motion": float(np.mean([self._motion_jump(t, self.motion_log) is not None for t, _ in samples]))}
        if self.audio is not None:
            rates["sound"] = float(np.mean([self._sound_jump(i) is not None for _, i in samples]))
        return rates

    # ---------- report ----------
    def _to_h264(self) -> None:
        """Re-encode MPEG-4 fallback clips as H.264 with ffmpeg so the report and the app can play them."""
        for clip in self.mp4v_clips:
            tmp = clip.with_suffix(".h264.mp4")
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(tmp)], check=True, timeout=120)
                tmp.replace(clip)
            except (OSError, subprocess.SubprocessError):
                tmp.unlink(missing_ok=True)
                print(f"[warn] {clip.name} stays MPEG-4 (no ffmpeg with libx264); browsers may not play it")
        self.mp4v_clips = []

    def write_report(self, source: str, duration_s: float) -> Path:
        if self.open is not None and self.open.end_s is None:
            self.open.end_s = duration_s
        for c in self.clips:
            c[0].release()
        self.clips = []
        self._to_h264()
        brief = [e for e in self.events if e.end_s is not None and e.end_s - e.start_s < MIN_EVENT_S]
        self.events = [e for e in self.events if e not in brief]
        for ev in self.events:
            cues, face = self._cues(ev)
            ev.interp = interpret(ev.reaction, cues, face, self.cfg)
            if ev.interp is not None:
                ev.interp["face_visible"] = round(face, 2)  # kept so the guess can be re-checked
        faith = check_events(self.events, self.cfg)
        with open(self.dir / "labels_template.csv", "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            wr = _csv.writer(f)
            wr.writerow(["event_id", "time", "reaction", "system_guess", "confidence", "your_label", "notes"])
            for i, ev in enumerate(self.events):
                g = ev.interp or {}
                wr.writerow([i, _fmt(ev.start_s), ev.reaction, g.get("top", ""), g.get("band", ""), "", ""])
        (self.dir / "events.json").write_text(json.dumps([e.__dict__ for e in self.events], ensure_ascii=False, indent=1))
        (self.dir / "summary.json").write_text(json.dumps({"n_brief": len(brief), "duration_s": duration_s,
                                                           "has_audio": self.audio is not None,
                                                           "faithfulness": faith}, ensure_ascii=False))
        with open(self.dir / "track.csv", "w", newline="") as f:  # per-frame log, used to merge several cameras
            import csv as _csv
            wr = _csv.writer(f)
            wr.writerow(["t", "present", "area", "reaction", "activity_z", "face"])
            for t, present, area, reaction, act, face in self.track:
                wr.writerow([f"{t:.3f}", int(present), f"{area:.5f}", reaction,
                             "" if not np.isfinite(act) else f"{act:.3f}", int(face)])
        path = self.dir / "report.html"
        path.write_text(render_owner_report(self.events, source, duration_s, self.dir, self.time_budget(),
                                            self.robot_rounds(), self.cfg, self.base_rates(), len(brief),
                                            self.audio is not None), encoding="utf-8")
        return path


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"
