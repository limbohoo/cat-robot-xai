"""Bounding-box cropping helpers used in both training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropRect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def expand_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    margin: float = 0.15,
    square: bool = True,
) -> CropRect:
    """Expand a (x0, y0, x1, y1) box by `margin` on every side, optionally squared."""
    x0, y0, x1, y1 = bbox
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    if square:
        side = max(w, h)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        half = side * (1 + margin) / 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    else:
        pad_w, pad_h = w * margin, h * margin
        x0 -= pad_w
        y0 -= pad_h
        x1 += pad_w
        y1 += pad_h
    W, H = image_size
    x0 = float(max(0.0, min(x0, W - 1)))
    y0 = float(max(0.0, min(y0, H - 1)))
    x1 = float(max(0.0, min(x1, W - 1)))
    y1 = float(max(0.0, min(y1, H - 1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        # Degenerate — return whole image
        return CropRect(0.0, 0.0, float(W - 1), float(H - 1))
    return CropRect(x0, y0, x1, y1)


def normalize_points(
    points: np.ndarray, rect: CropRect
) -> np.ndarray:
    """Convert pixel-space (x, y) to normalized [0, 1] inside the crop."""
    nx = (points[..., 0] - rect.x0) / max(rect.width, 1e-6)
    ny = (points[..., 1] - rect.y0) / max(rect.height, 1e-6)
    out = np.stack([nx, ny], axis=-1).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def denormalize_points(
    norm_points: np.ndarray, rect: CropRect
) -> np.ndarray:
    """Convert normalized [0, 1] crop coordinates back to pixel space."""
    x = norm_points[..., 0] * rect.width + rect.x0
    y = norm_points[..., 1] * rect.height + rect.y0
    return np.stack([x, y], axis=-1)


def face_bbox_from_face_landmarks(
    face_pts: np.ndarray,
    conf: np.ndarray | None = None,
    min_conf: float = 0.25,
) -> tuple[float, float, float, float] | None:
    """Rebuild a CatFLW-style face box from 48 predicted landmarks.

    On CatFLW training labels the annotated box is, at the median, 1.36x the
    landmark extent in width and 1.30x in height, centred 3% lower. Used for a
    second, refined face pass after a first pass on a rough box.
    """
    pts = np.asarray(face_pts, dtype=np.float32)
    if conf is not None:
        pts = pts[np.asarray(conf) >= min_conf]
    if len(pts) < 24:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = float(x1 - x0), float(y1 - y0)
    if w < 4 or h < 4:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.034 * h
    hw, hh = 1.356 * w / 2.0, 1.30 * h / 2.0
    return (float(cx - hw), float(cy - hh), float(cx + hw), float(cy + hh))


def face_bbox_from_body_keypoints(
    body_pts: np.ndarray,
    image_size: tuple[int, int],
    expand: float = 0.55,
    min_side: float = 32.0,
    conf: np.ndarray | None = None,
    min_conf: float = 0.25,
) -> tuple[float, float, float, float] | None:
    """Estimate a face bbox from AnimalPose head keypoints.

    Uses body indices 0-4 (left_eye, right_eye, nose, left_ear, right_ear).
    With `conf`, only head points above `min_conf` count and at least 3 are
    required. Returns None if the enclosing box is degenerate.

    The face model was trained on tight face crops (CatFLW). Handing it the
    whole body bbox from YOLO produces scattered predictions because the
    receptive field is stretched across too much irrelevant background. This
    helper feeds the face model something close to what it saw at training
    time — a padded box around the head landmarks the body model just found.
    """
    if body_pts is None or len(body_pts) < 5:
        return None
    head = np.asarray(body_pts[:5], dtype=np.float32)
    if conf is not None:
        head = head[np.asarray(conf[:5]) >= min_conf]
        if len(head) < 3:
            return None
    x0, y0 = head.min(axis=0)
    x1, y1 = head.max(axis=0)
    w = float(x1 - x0)
    h = float(y1 - y0)
    if w < 1 and h < 1:
        return None
    side = max(w, h, min_side)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side * (1.0 + expand) / 2.0
    W, H = image_size
    fx0 = float(max(0.0, cx - half))
    fy0 = float(max(0.0, cy - half))
    fx1 = float(min(W - 1, cx + half))
    fy1 = float(min(H - 1, cy + half))
    if fx1 - fx0 < 4 or fy1 - fy0 < 4:
        return None
    return (fx0, fy0, fx1, fy1)
