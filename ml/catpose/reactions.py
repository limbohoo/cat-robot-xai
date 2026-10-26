"""Rule-based reaction coding on top of the geometric features (catpose/features.py).

Labels describe observable behaviour relative to this cat's own baseline, in
the spirit of an ethogram; they are NOT emotions. Thresholds follow the
direction of the Feline Grimace Scale ear action unit (Evangelista et al. 2019)
and owner body-language guides (Cats Protection), but their values are
unvalidated guesses until labelled clips are collected.

A label must dominate the last `hold_s` seconds before it replaces the current
one, so single noisy frames do not flicker the display.

Every label carries the conditions that produced it (`Reaction.drivers`): which
feature, its deviation from this cat's baseline, and the threshold it crossed.
When the debounced label switches, the coder also keeps the feature values the
decision was made on (`snapshot`, the median over the frames that voted for
the label) and the label shown before (`prev`, which sets the thresholds in
use). The owner-facing explanation is built from these, and
catpose/faithfulness.py re-runs `classify` on them to check that the
explanation names the real reasons.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# z-score thresholds. ear_height is the reliable feature, so it gets a lower bar;
# the noisier face features need a larger deviation before they count.
Z_EAR_H = 1.5
Z_AUX = 2.0
Z_ACTIVE = 2.0
Z_STILL = -1.0
Z_EAR_CO = 0.5  # ear angle may open "ears back" only when the ear tip is also at least this much lower
# Hysteresis: a label already shown stays until its feature falls back inside this
# smaller deviation, so values hovering at a threshold do not toggle the label.
Z_EAR_H_STAY = 1.0
Z_ACTIVE_STAY = 1.5
Z_STILL_STAY = -0.5

USED_FEATURES = ("ear_height", "ear_angle", "eye_aperture", "activity")  # what classify reads

LABELS = {
    "calibrating": "calibrating baseline",
    "unknown": "unknown",
    "resting": "no clear change",
    "active": "active / exploring",
    "alert": "alert / orienting",
    "ears_back": "ears back / out",
    "freeze": "freeze (still + ears back)",
}

# BGR colours for the overlay
COLORS = {
    "calibrating": (150, 160, 175), "unknown": (150, 160, 175), "resting": (129, 185, 16),
    "active": (248, 189, 56), "alert": (11, 158, 245), "ears_back": (60, 76, 231), "freeze": (60, 60, 240),
}


@dataclass
class Reaction:
    key: str
    evidence: list[str] = field(default_factory=list)
    # conditions that produced the label: {"feature", "z", "test" ("<=" / ">="), "bar"}
    drivers: list[dict] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)  # z per feature the switch was decided on
    prev: str = ""  # label shown before this one (decides which thresholds applied)
    consistent: bool = True  # False if the snapshot alone does not give this label (see ReactionCoder)

    @property
    def text(self) -> str:
        return LABELS[self.key]


def _z(features: dict, k: str) -> float:
    return features.get(k, (np.nan, np.nan, np.nan))[1]


def _drv(feature: str, z: float, test: str, bar: float) -> dict:
    return {"feature": feature, "z": round(float(z), 3), "test": test, "bar": round(float(bar), 3)}


def as_features(z: dict) -> dict:
    """{feature: z} -> the (value, z, raw) layout classify reads."""
    return {k: (np.nan, v, np.nan) for k, v in z.items()}


def classify(features: dict, calibrating: bool, pose_note: str = "", prev: str = "") -> Reaction:
    """One frame -> one reaction, with the conditions that triggered it.

    `pose_note` says why face features are paused (head turned or tilted); the label then
    rests on the body only. `prev` is the label currently shown (for hysteresis).
    """
    if calibrating:
        return Reaction("calibrating")
    ear_h, ear_a, eye, act = (_z(features, k) for k in USED_FEATURES)
    have_face = np.isfinite(ear_h) or np.isfinite(ear_a) or np.isfinite(eye)
    if not have_face and not np.isfinite(act):
        return Reaction("unknown", [pose_note or "no reliable keypoints"])

    # Only ear tip height (reliable) can open a face label; the noisier ear angle and eye
    # aperture add supporting evidence, except that ear angle may open "ears back" when the
    # ear tip also points the same way.
    back_bar = Z_EAR_H_STAY if prev in ("ears_back", "freeze") else Z_EAR_H
    up_bar = Z_EAR_H_STAY if prev == "alert" else Z_EAR_H
    active_bar = Z_ACTIVE_STAY if prev == "active" else Z_ACTIVE
    still_bar = Z_STILL_STAY if prev == "freeze" else Z_STILL
    ears_back, back_drv = [], []
    if np.isfinite(ear_h) and ear_h <= -back_bar:
        ears_back.append(f"ear tip {ear_h:+.1f} SD (lower)")
        back_drv.append(_drv("ear_height", ear_h, "<=", -back_bar))
    if np.isfinite(ear_a) and ear_a >= Z_AUX and (ears_back or (np.isfinite(ear_h) and ear_h <= -Z_EAR_CO)):
        ears_back.append(f"ear angle {ear_a:+.1f} SD (outward)")
        back_drv.append(_drv("ear_angle", ear_a, ">=", Z_AUX))
        if not any(d["feature"] == "ear_height" for d in back_drv):  # the ear-tip co-condition
            back_drv.append(_drv("ear_height", ear_h, "<=", -Z_EAR_CO))
    alert, alert_drv = [], []
    if np.isfinite(ear_h) and ear_h >= up_bar:
        alert.append(f"ear tip {ear_h:+.1f} SD (higher)")
        alert_drv.append(_drv("ear_height", ear_h, ">=", up_bar))
        if np.isfinite(eye) and eye >= Z_AUX:
            alert.append(f"eyes {eye:+.1f} SD (wider)")
            alert_drv.append(_drv("eye_aperture", eye, ">=", Z_AUX))
    still = np.isfinite(act) and act <= still_bar
    active = np.isfinite(act) and act >= active_bar
    act_txt = f"activity {act:+.1f} SD" if np.isfinite(act) else "activity n/a"

    if ears_back and still:
        return Reaction("freeze", ears_back + [act_txt + " (still)"],
                        back_drv + [_drv("activity", act, "<=", still_bar)])
    if ears_back:
        return Reaction("ears_back", ears_back, back_drv)
    if alert and not active:
        return Reaction("alert", alert, alert_drv)
    if active:
        return Reaction("active", [act_txt + " (moving)"], [_drv("activity", act, ">=", active_bar)])
    if not have_face:  # body at rest is still observable; say the face was not used
        return Reaction("resting", [act_txt + "; body only", (pose_note or "face not visible") + ", face paused"])
    evidence = ["features within baseline"]
    return Reaction("resting", evidence + [act_txt])


def _median_snapshot(zs: list[dict]) -> dict:
    """Per-feature median of z over frames, ignoring frames where the feature was not measured."""
    keys = {k for z in zs for k in z}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        for k in sorted(keys):
            v = np.nanmedian([z.get(k, np.nan) for z in zs])
            if np.isfinite(v):
                out[k] = round(float(v), 3)
    return out


class ReactionCoder:
    """Debounces per-frame labels by time and keeps a short log of reaction events.

    A new label is shown once it makes up at least `majority` of the frames in the last
    `hold_s` seconds. Measuring in seconds keeps the delay the same at any frame rate, and a
    majority (rather than N consecutive frames) still switches when the raw label flickers
    between two neighbours such as "ears back" and "freeze".

    On a switch, the new Reaction records what the decision rested on: the median z of the
    frames that voted for it (`snapshot`), the label shown before (`prev`), and the
    conditions `classify` finds in that snapshot (`drivers`). If the snapshot on its own
    would give a different label (possible when voting frames met different conditions),
    the switching frame's own conditions are kept and `consistent` is False.
    """

    def __init__(self, hold_s: float = 0.3, majority: float = 0.6, log_size: int = 4) -> None:
        self.hold_s, self.majority = hold_s, majority
        self.current = Reaction("calibrating")
        self.window: deque = deque()  # (t, Reaction, {feature: z})
        self.since = time.time()
        self.events: deque = deque(maxlen=log_size)  # (stamp, label key, evidence)

    def update(self, features: dict, calibrating: bool, stamp: str | None = None, pose_note: str = "",
               t: float | None = None) -> Reaction:
        """`t` is the frame time in seconds (video position for files, wall clock otherwise);
        `stamp` labels logged events and defaults to the wall clock."""
        t = time.time() if t is None else t
        r = classify(features, calibrating, pose_note, prev=self.current.key)
        z = {k: float(v[1]) for k, v in features.items() if np.isfinite(v[1])}
        self.window.append((t, r, z))
        while self.window and self.window[0][0] < t - self.hold_s:
            self.window.popleft()
        if r.key == self.current.key:
            self.current = r  # same label, fresher evidence
            return self.current
        share = sum(1 for _, w, _ in self.window if w.key == r.key) / len(self.window)
        if share >= self.majority and t - self.window[0][0] >= 0.5 * self.hold_s:
            prev = self.current.key
            snap = _median_snapshot([zz for _, w, zz in self.window if w.key == r.key])
            decided = classify(as_features(snap), calibrating, pose_note, prev=prev)
            if decided.key == r.key:
                self.current = Reaction(r.key, decided.evidence, decided.drivers, snap, prev, True)
            else:
                self.current = Reaction(r.key, r.evidence, r.drivers, snap, prev, False)
            self.since = t
            if r.key not in ("calibrating", "unknown", "resting"):
                self.events.append((stamp or time.strftime("%H:%M:%S"), r.key, "; ".join(r.evidence)))
        return self.current
