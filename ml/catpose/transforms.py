"""Affine crop + augmentation shared by training and inference.

A square region (center, side, rotation) is warped to an out x out patch.
Regions that extend past the frame are zero-padded instead of clipped, so
the animal is never squashed when it stands near an image edge.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def box_to_center_side(bbox, margin: float) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)
    side = max(x1 - x0, y1 - y0, 8.0) * (1.0 + margin)
    return center, side


def crop_matrix(center, side: float, rot_deg: float, out: int) -> np.ndarray:
    """2x3 matrix mapping image pixels -> patch pixels."""
    s = out / side
    a = math.radians(rot_deg)
    cos, sin = math.cos(a) * s, math.sin(a) * s
    cx, cy = float(center[0]), float(center[1])
    return np.array([
        [cos, sin, out / 2 - cos * cx - sin * cy],
        [-sin, cos, out / 2 + sin * cx - cos * cy],
    ], dtype=np.float32)


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m)


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def warp(image_rgb: np.ndarray, m: np.ndarray, out: int) -> np.ndarray:
    return cv2.warpAffine(image_rgb, m, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def to_tensor_array(patch_rgb: np.ndarray) -> np.ndarray:
    x = patch_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def color_jitter(patch: np.ndarray) -> np.ndarray:
    x = patch.astype(np.float32)
    x = x * random.uniform(0.7, 1.3) + random.uniform(-25, 25)  # contrast, brightness
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * random.uniform(0.6, 1.4)  # saturation
    if random.random() < 0.1:
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)
    if random.random() < 0.15:
        x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.5, 1.5))
    return np.clip(x, 0, 255).astype(np.uint8)


def augmented_crop(image_rgb, bbox, pts, vis, out: int, margin: float,
                   flip_map: list[int] | None, train: bool):
    """Returns (patch_rgb, pts_in_patch, vis, matrix). Points outside the patch lose visibility."""
    center, side = box_to_center_side(bbox, margin)
    rot = 0.0
    flip = False
    if train:
        side *= random.uniform(0.8, 1.25)
        center = center + np.random.uniform(-0.08, 0.08, 2).astype(np.float32) * side
        if random.random() < 0.6:
            rot = random.uniform(-30, 30)
        flip = flip_map is not None and random.random() < 0.5

    m = crop_matrix(center, side, rot, out)
    patch = warp(image_rgb, m, out)
    p = apply(m, pts)
    v = vis.copy()
    if flip:
        patch = np.ascontiguousarray(patch[:, ::-1])
        p[:, 0] = out - 1 - p[:, 0]
        p = p[flip_map]
        v = v[flip_map]
        f = np.array([[-1, 0, out - 1], [0, 1, 0]], dtype=np.float32)
        m = np.vstack([f, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]])
        m = m[:2]
    inside = (p[:, 0] >= 0) & (p[:, 0] < out) & (p[:, 1] >= 0) & (p[:, 1] < out)
    v = v * inside.astype(np.float32)
    if train:
        patch = color_jitter(patch)
    return patch, p.astype(np.float32), v, m
