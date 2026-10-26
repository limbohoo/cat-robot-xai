"""One-Euro filter for keypoint streams (Casiez et al., CHI 2012).

Adaptive low-pass: heavy smoothing when a point is nearly still (kills
jitter), light smoothing when it moves fast (keeps latency low).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None
        self.t: Optional[float] = None

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: Optional[float] = None, scale: float = 1.0) -> np.ndarray:
        """`scale` = object size in pixels, so the speed threshold is size-invariant."""
        t = time.time() if t is None else t
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True) / max(scale, 1.0)
        cutoff = self.min_cutoff + self.beta * speed * 100
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = self.x + a * (x - self.x)
        return self.x.copy()
