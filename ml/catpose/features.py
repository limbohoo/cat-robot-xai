"""Geometric features from face + body keypoints, relative to this cat's own baseline.

These are the features of the Colab tutorial (ml/colab, section 4): ear-tip
height, ear rotation, eye aperture, muzzle ratio, head drop and activity. They
describe what the face and body look like; they are NOT an emotion
classifier. Each value is also shown as a z-score against a baseline collected
while the cat is calm, because breeds differ (a Scottish Fold's ears always
sit low).

CatFLW indices were identified from the training-set mean face (the dataset
does not name them): ear contours 22-26 / 27-31 with tips 24 / 29, eye
outer/inner/upper/lower corners 4-7 / 8-11, muzzle sides 32 / 35, chin 0 / 2.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

EAR_L, EAR_R = [22, 23, 24, 25, 26], [27, 28, 29, 30, 31]
EAR_TIP_L, EAR_TIP_R = 24, 29
EAR_BASE_L, EAR_BASE_R = [22, 26], [27, 31]
EYE_L, EYE_R = [4, 5, 6, 7], [8, 9, 10, 11]  # outer, inner, upper, lower
MUZZLE = [32, 35]
CHIN = [0, 2]
NOSE = [12, 13]

# Head-pose gate. The features only cancel in-plane rotation; when the cat looks up, down or
# sideways the ears foreshorten and look "lower". Yaw: left/right eye-to-nose asymmetry
# (0 when frontal). Pitch: eye-to-nose drop / eye distance, 0.55 on the CatFLW mean face.
MAX_YAW = 0.2
MAX_PITCH_CHANGE = 0.3  # relative change vs the baseline pitch
DEFAULT_PITCH = 0.55
BODY_NOSE, BODY_WITHERS, BODY_TAIL = 2, 18, 19

# Model error / between-cat spread on held-out faces (tutorial section 4.7).
# Below 0.3 the feature reflects the cat, not model noise.
RELIABILITY = {"ear_height": 0.15, "ear_angle": 0.54, "eye_aperture": 0.41, "muzzle_ratio": 0.52}
# Mean absolute feature error caused by keypoint error on the same held-out faces. A baseline spread
# below this is measurement noise, so it floors the z-score denominator.
MODEL_ERROR = {"ear_height": 0.031, "ear_angle": 3.0, "eye_aperture": 0.056, "muzzle_ratio": 0.105,
               # Body features have no held-out estimate yet; these floors are about the jitter of a still cat.
               "head_drop": 0.1, "activity": 0.1}

MAX_SPEED = 3.0  # cat-sizes per second

FEATURE_NAMES = ["ear_height", "ear_angle", "eye_aperture", "muzzle_ratio", "head_drop", "activity"]


def _ok(conf: Optional[np.ndarray], idx, min_conf: float) -> bool:
    return conf is None or bool((conf[idx] >= min_conf).all())


def head_pose(pts: np.ndarray) -> tuple[float, float]:
    """(yaw asymmetry, pitch ratio) proxies from eyes and nose."""
    e_l, e_r, nose = pts[EYE_L].mean(0), pts[EYE_R].mean(0), pts[NOSE].mean(0)
    a, b = np.linalg.norm(e_l - nose), np.linalg.norm(e_r - nose)
    d = np.linalg.norm(e_r - e_l) + 1e-6
    xhat = (e_r - e_l) / d
    drop = abs(np.dot(nose - (e_l + e_r) / 2, np.array([-xhat[1], xhat[0]])))
    return float(abs(a - b) / (a + b + 1e-6)), float(drop / d)


def face_features(pts: np.ndarray, conf: Optional[np.ndarray] = None, min_conf: float = 0.25) -> dict:
    """Scale-, translation- and in-plane-rotation-invariant face features (NaN when points are unreliable)."""
    out = {k: float("nan") for k in ("ear_height", "ear_angle", "eye_aperture", "muzzle_ratio")}
    if not _ok(conf, EYE_L + EYE_R + CHIN + NOSE, min_conf):
        return out
    e_l, e_r = pts[EYE_L].mean(0), pts[EYE_R].mean(0)
    d = float(np.linalg.norm(e_r - e_l))
    if d < 1e-3:
        return out
    xhat = (e_r - e_l) / d
    up = np.array([xhat[1], -xhat[0]])
    mid = (e_l + e_r) / 2
    if np.dot(pts[CHIN].mean(0) - mid, up) > 0:  # point "up" away from the chin
        up = -up

    if _ok(conf, [EAR_TIP_L, EAR_TIP_R], min_conf):
        out["ear_height"] = float(np.mean([np.dot(pts[t] - mid, up) / d for t in (EAR_TIP_L, EAR_TIP_R)]))
    if _ok(conf, EAR_L + EAR_R, min_conf):
        angles = []
        for tip, base, side in ((EAR_TIP_L, EAR_BASE_L, -1), (EAR_TIP_R, EAR_BASE_R, 1)):
            axis = pts[tip] - pts[base].mean(0)
            angles.append(np.degrees(np.arctan2(np.dot(axis, xhat) * side, np.dot(axis, up))))
        out["ear_angle"] = float(np.mean(angles))
    widths = [np.linalg.norm(pts[e[0]] - pts[e[1]]) for e in (EYE_L, EYE_R)]
    if min(widths) > 1e-3:
        out["eye_aperture"] = float(np.mean(
            [np.linalg.norm(pts[e[2]] - pts[e[3]]) / w for e, w in zip((EYE_L, EYE_R), widths)]))
    if _ok(conf, MUZZLE, min_conf):
        out["muzzle_ratio"] = float(np.linalg.norm(pts[MUZZLE[0]] - pts[MUZZLE[1]]) / d)
    return out


def body_features(pts: np.ndarray, conf: Optional[np.ndarray] = None, min_conf: float = 0.25) -> dict:
    """Head drop: nose height below the withers, in body lengths (image y grows downward)."""
    if not _ok(conf, [BODY_NOSE, BODY_WITHERS, BODY_TAIL], min_conf):
        return {"head_drop": float("nan")}
    length = float(np.linalg.norm(pts[BODY_WITHERS] - pts[BODY_TAIL]))
    if length < 1e-3:
        return {"head_drop": float("nan")}
    return {"head_drop": float((pts[BODY_NOSE, 1] - pts[BODY_WITHERS, 1]) / length)}


class ActivityMeter:
    """Mean keypoint speed in cat-sizes per second, averaged over roughly `window_s` seconds."""

    def __init__(self, window_s: float = 0.5) -> None:
        self.window_s = window_s
        self.prev: Optional[np.ndarray] = None
        self.t: Optional[float] = None
        self.value = float("nan")

    def reset(self) -> None:
        self.prev, self.t, self.value = None, None, float("nan")

    def __call__(self, pts: np.ndarray, conf: Optional[np.ndarray], scale: float, min_conf: float = 0.25,
                 t: Optional[float] = None) -> float:
        t = time.time() if t is None else t
        if self.prev is not None and self.prev.shape == pts.shape and self.t is not None:
            dt = max(t - self.t, 1e-3)
            keep = np.ones(len(pts), bool) if conf is None else conf >= min_conf
            if keep.any():
                # Median over keypoints: a few points jumping at close range must not dominate.
                # Capped at MAX_SPEED cat-sizes per second, faster is keypoint error, not a cat.
                speed = float(np.median(np.linalg.norm(pts[keep] - self.prev[keep], axis=1)) / max(scale, 1.0) / dt)
                speed = min(speed, MAX_SPEED)
                a = min(1.0, dt / self.window_s)
                self.value = speed if np.isnan(self.value) else self.value + a * (speed - self.value)
        self.prev, self.t = pts.copy(), t
        return self.value


class FeatureBaseline:
    """Robust per-cat baseline: median and 1.4826 * MAD of frames collected while the cat is calm."""

    def __init__(self, calib_frames: int = 300, min_frames: int = 60) -> None:
        self.calib_frames, self.min_frames = calib_frames, min_frames
        self.buf: dict[str, deque] = {k: deque(maxlen=calib_frames) for k in FEATURE_NAMES}
        self.med: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        self.collecting = True
        self.n = 0  # frames that contributed at least one feature

    def restart(self) -> None:
        for b in self.buf.values():
            b.clear()
        self.med, self.scale, self.collecting, self.n = {}, {}, True, 0

    @property
    def progress(self) -> float:
        return min(1.0, self.n / self.calib_frames)

    def add(self, feats: dict) -> None:
        if not self.collecting:
            return
        seen = False
        for k, v in feats.items():
            if k in self.buf and np.isfinite(v):
                self.buf[k].append(v)
                seen = True
        self.n += seen
        if self.n >= self.calib_frames:
            self.finish()

    def finish(self) -> None:
        for k, b in self.buf.items():
            if len(b) >= self.min_frames:
                x = np.asarray(b)
                med = float(np.median(x))
                self.med[k] = med
                mad = float(1.4826 * np.median(np.abs(x - med)))
                self.scale[k] = max(mad, MODEL_ERROR.get(k, 0.0)) + 1e-6
        self.collecting = False

    def z(self, k: str, v: float) -> float:
        if k not in self.med or not np.isfinite(v):
            return float("nan")
        return (v - self.med[k]) / self.scale[k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"median": self.med, "scale": self.scale}, indent=1))

    def load(self, path: Path) -> None:
        d = json.loads(path.read_text())
        self.med, self.scale, self.collecting = d["median"], d["scale"], False


class FeatureTracker:
    """Per-frame features -> short EMA for display -> z-score against the baseline."""

    def __init__(self, calib_frames: int = 300, ema: float = 0.5, min_conf: float = 0.25) -> None:
        self.baseline = FeatureBaseline(calib_frames)
        self.activity = ActivityMeter()
        self.ema, self.min_conf = ema, min_conf
        self.smooth: dict[str, float] = {}
        self.pitches: deque = deque(maxlen=calib_frames)
        self.pose_note = ""  # why face features are paused, if they are

    def _frontal(self, face_pts) -> bool:
        yaw, pitch = head_pose(face_pts)
        if self.baseline.collecting:
            self.pitches.append(pitch)
        ref = float(np.median(self.pitches)) if len(self.pitches) >= 30 else DEFAULT_PITCH
        if yaw > MAX_YAW:
            self.pose_note = f"head turned sideways (yaw {yaw:.2f})"
            return False
        if abs(pitch / ref - 1) > MAX_PITCH_CHANGE:
            self.pose_note = f"head tilted up/down (pitch {pitch / ref - 1:+.0%} vs baseline)"
            return False
        self.pose_note = ""
        return True

    def update(self, face_pts, face_conf, body_pts, body_conf, body_scale: float, t: Optional[float] = None) -> dict:
        """`t` is the frame time in seconds (video position for files); defaults to the wall clock."""
        raw = {k: float("nan") for k in FEATURE_NAMES}
        self.pose_note = ""
        if face_pts is not None and self._frontal(face_pts):
            raw.update(face_features(face_pts, face_conf, self.min_conf))
        if body_pts is not None:
            raw.update(body_features(body_pts, body_conf, self.min_conf))
            raw["activity"] = self.activity(body_pts, body_conf, body_scale, self.min_conf, t=t)
        else:
            self.activity.reset()
        for k, v in raw.items():
            if np.isfinite(v):
                prev = self.smooth.get(k)
                self.smooth[k] = v if prev is None or not np.isfinite(prev) else prev + self.ema * (v - prev)
            else:
                self.smooth[k] = float("nan")
        self.baseline.add(raw)
        return {k: (self.smooth[k], self.baseline.z(k, self.smooth[k]), raw[k]) for k in FEATURE_NAMES}
