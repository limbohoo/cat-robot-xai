"""Draw predictions on a PIL image (API) or BGR frame (OpenCV demo).

Points whose confidence is below `min_conf` are not drawn, and skeleton edges
are drawn only when both endpoints are confident. Legacy models return no
confidences, in which case everything is drawn.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .schema import ANIMALPOSE_SKELETON

MIN_CONF = 0.25


def _keep(n: int, conf: np.ndarray | None, min_conf: float) -> np.ndarray:
    return np.ones(n, bool) if conf is None else np.asarray(conf) >= min_conf


def overlay(
    image: Image.Image,
    face_pts: np.ndarray | None = None,
    body_pts: np.ndarray | None = None,
    face_bbox: tuple[float, float, float, float] | None = None,
    body_bbox: tuple[float, float, float, float] | None = None,
    face_conf: np.ndarray | None = None,
    body_conf: np.ndarray | None = None,
    min_conf: float = MIN_CONF,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    if body_bbox is not None:
        draw.rectangle(body_bbox, outline="#10B981", width=3)
    if body_pts is not None:
        keep = _keep(len(body_pts), body_conf, min_conf)
        for a, b in ANIMALPOSE_SKELETON:
            if keep[a] and keep[b]:
                draw.line((*body_pts[a], *body_pts[b]), fill="#F59E0B", width=3)
        for (x, y), k in zip(body_pts, keep):
            if k:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#F59E0B", outline="white", width=1)

    if face_bbox is not None:
        draw.rectangle(face_bbox, outline="#24A8D8", width=2)
    if face_pts is not None:
        keep = _keep(len(face_pts), face_conf, min_conf)
        for (x, y), k in zip(face_pts, keep):
            if k:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#FF3B30", outline="white", width=1)
    return out


def overlay_cv2(
    frame_bgr,
    face_pts: np.ndarray | None = None,
    body_pts: np.ndarray | None = None,
    face_bbox: tuple[float, float, float, float] | None = None,
    body_bbox: tuple[float, float, float, float] | None = None,
    fps: float | None = None,
    face_conf: np.ndarray | None = None,
    body_conf: np.ndarray | None = None,
    min_conf: float = MIN_CONF,
):
    import cv2

    frame = frame_bgr.copy()
    if body_bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in body_bbox)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 165, 255), 2)
    if body_pts is not None:
        keep = _keep(len(body_pts), body_conf, min_conf)
        for a, b in ANIMALPOSE_SKELETON:
            if keep[a] and keep[b]:
                cv2.line(frame, tuple(int(v) for v in body_pts[a]), tuple(int(v) for v in body_pts[b]),
                         (0, 165, 255), 2, cv2.LINE_AA)
        for (x, y), k in zip(body_pts, keep):
            if k:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    if face_bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in face_bbox)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (216, 168, 36), 1)
    if face_pts is not None:
        keep = _keep(len(face_pts), face_conf, min_conf)
        for (x, y), k in zip(face_pts, keep):
            if k:
                cv2.circle(frame, (int(x), int(y)), 2, (48, 59, 255), -1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame
