"""CatPose inference for the UNO Q: onnxruntime + numpy + OpenCV, no PyTorch.

Mirrors ml/catpose (transforms.py, predictor.py, crop.py, smoothing.py) so the
board produces the same landmarks as the Mac pipeline. Frames are BGR, as
delivered by arduino.app_peripherals.camera.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent / "models"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COCO_CAT_ID = 15

BODY_NAMES = [
    "left_eye", "right_eye", "nose", "left_ear", "right_ear",
    "left_front_elbow", "right_front_elbow", "left_back_elbow", "right_back_elbow",
    "left_front_knee", "right_front_knee", "left_back_knee", "right_back_knee",
    "left_front_paw", "right_front_paw", "left_back_paw", "right_back_paw",
    "throat", "withers", "tailbase",
]
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 17), (18, 19),
    (5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14), (11, 15), (12, 16),
]


def _session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


# ---------- detector ----------

@dataclass
class Detection:
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy pixels


class CatDetector:
    """YOLOv8 (COCO) exported by ultralytics: output (1, 84, N) = cx, cy, w, h + 80 class scores."""

    def __init__(self, imgsz: int, conf: float = 0.25, threads: int = 4) -> None:
        self.sess = _session(MODELS / "yolo.onnx", threads)
        self.imgsz = imgsz
        self.conf = conf

    def best_cat(self, bgr: np.ndarray) -> Detection | None:
        h, w = bgr.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = round(h * s), round(w * s)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)  # letterbox, top-left aligned
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]  # (84, N)
        scores = out[4 + COCO_CAT_ID]
        i = int(np.argmax(scores))
        if scores[i] < self.conf:
            return None
        cx, cy, bw, bh = out[:4, i] / s
        box = (max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(w - 1.0, cx + bw / 2), min(h - 1.0, cy + bh / 2))
        return Detection(float(scores[i]), tuple(float(v) for v in box))


# ---------- landmark models ----------

def _crop_matrix(center, side: float, out: int) -> np.ndarray:
    s = out / side
    return np.array([[s, 0, out / 2 - s * center[0]], [0, s, out / 2 - s * center[1]]], np.float32)


class PosePredictor:
    def __init__(self, kind: str, meta: dict, threads: int = 4) -> None:
        self.kind = kind
        self.sess = _session(MODELS / f"{kind}.onnx", threads)
        self.img = int(meta["img"])
        self.margin = float(meta["margin"])
        self.flip_map = meta.get("flip_map")
        self.tta = False  # flip TTA doubles CPU time; off by default on the board

    def predict(self, rgb: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray, tuple]:
        x0, y0, x1, y1 = bbox
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        side = max(x1 - x0, y1 - y0, 8.0) * (1.0 + self.margin)
        m = _crop_matrix(center, side, self.img)
        patch = cv2.warpAffine(rgb, m, (self.img, self.img), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        views = [patch]
        use_flip = self.tta and self.flip_map is not None
        if use_flip:
            views.append(np.ascontiguousarray(patch[:, ::-1]))
        batch = np.stack([((v.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
                          for v in views])
        uv, conf = self.sess.run(None, {"image": batch})
        uv = uv * self.img
        pts, c = uv[0], conf[0]
        if use_flip:
            pf = uv[1].copy()
            pf[:, 0] = self.img - 1 - pf[:, 0]
            pts = (pts + pf[self.flip_map]) / 2
            c = (c + conf[1][self.flip_map]) / 2
        inv = cv2.invertAffineTransform(m)
        pts = pts @ inv[:, :2].T + inv[:, 2]
        half = side / 2
        return pts, c, (center[0] - half, center[1] - half, center[0] + half, center[1] + half)


def face_bbox_from_body(body_pts, conf, frame_wh, expand=0.55, min_side=32.0, min_conf=0.25):
    head = body_pts[:5][conf[:5] >= min_conf]
    if len(head) < 3:
        return None
    (x0, y0), (x1, y1) = head.min(0), head.max(0)
    side = max(x1 - x0, y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side * (1 + expand) / 2
    W, H = frame_wh
    box = (max(0.0, cx - half), max(0.0, cy - half), min(W - 1.0, cx + half), min(H - 1.0, cy + half))
    return box if box[2] - box[0] >= 4 and box[3] - box[1] >= 4 else None


def face_bbox_from_face(face_pts, conf, min_conf=0.25):
    pts = face_pts[conf >= min_conf]
    if len(pts) < 24:
        return None
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.034 * h
    return (cx - 1.356 * w / 2, cy - 1.30 * h / 2, cx + 1.356 * w / 2, cy + 1.30 * h / 2)


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x, scale=1.0):
        t = time.time()
        x = np.asarray(x, np.float64)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = 1.0 / (1.0 + (1.0 / (2 * math.pi * self.d_cutoff)) / dt)
        self.dx = self.dx + a_d * ((x - self.x) / dt - self.dx)
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True) / max(scale, 1.0)
        cutoff = self.min_cutoff + self.beta * speed * 100
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = self.x + a * (x - self.x)
        return self.x.copy()


# ---------- full pipeline ----------

@dataclass
class Result:
    cat: Detection | None = None
    stale: bool = False
    body: np.ndarray | None = None
    body_conf: np.ndarray | None = None
    face: np.ndarray | None = None
    face_conf: np.ndarray | None = None
    face_box: tuple | None = None
    ms: dict | None = None


class CatPose:
    def __init__(self, detect_every: int = 3, bbox_hold: int = 6, face_passes: int = 1, threads: int = 4) -> None:
        meta = json.loads((MODELS / "meta.json").read_text())
        self.detector = CatDetector(meta["yolo"]["imgsz"], threads=threads)
        self.body = PosePredictor("body", meta["body"], threads)
        self.face = PosePredictor("face", meta["face"], threads)
        self.detect_every, self.bbox_hold, self.face_passes = detect_every, bbox_hold, face_passes
        self.frame = 0
        self.since_hit = 10 ** 9
        self.cat: Detection | None = None
        self.body_filter, self.face_filter = OneEuro(), OneEuro()

    def __call__(self, bgr: np.ndarray) -> Result:
        self.frame += 1
        ms = {}
        t = time.perf_counter()
        if self.frame == 1 or self.frame % self.detect_every == 0:
            hit = self.detector.best_cat(bgr)
            if hit is not None:
                self.cat, self.since_hit = hit, 0
        ms["detect"] = (time.perf_counter() - t) * 1000
        self.since_hit += 1
        if self.since_hit > self.bbox_hold:
            self.cat = None

        r = Result(cat=self.cat, stale=self.since_hit > self.detect_every, ms=ms)
        if self.cat is None:
            self.body_filter.reset()
            self.face_filter.reset()
            return r

        t = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = self.cat.bbox
        scale = max(box[2] - box[0], box[3] - box[1])
        pts, conf, _ = self.body.predict(rgb, box)
        r.body, r.body_conf = self.body_filter(pts, scale=scale), conf
        ms["body"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fbox = face_bbox_from_body(r.body, conf, (rgb.shape[1], rgb.shape[0]))
        if fbox is not None:
            pts, conf, used = self.face.predict(rgb, fbox)
            for _ in range(self.face_passes - 1):
                refined = face_bbox_from_face(pts, conf)
                if refined is None:
                    break
                pts, conf, used = self.face.predict(rgb, refined)
            r.face, r.face_conf, r.face_box = self.face_filter(pts, scale=used[2] - used[0]), conf, used
        else:
            self.face_filter.reset()
        ms["face"] = (time.perf_counter() - t) * 1000
        return r


def draw(bgr: np.ndarray, r: Result, min_conf: float = 0.25) -> np.ndarray:
    out = bgr.copy()
    if r.cat is not None:
        x0, y0, x1, y1 = (int(v) for v in r.cat.bbox)
        color = (24, 132, 245) if r.stale else (129, 185, 16)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"cat {r.cat.confidence * 100:.0f}%", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if r.body is not None:
        keep = r.body_conf >= min_conf
        for a, b in BODY_SKELETON:
            if keep[a] and keep[b]:
                cv2.line(out, tuple(int(v) for v in r.body[a]), tuple(int(v) for v in r.body[b]),
                         (11, 158, 245), 2, cv2.LINE_AA)
        for (x, y), k in zip(r.body, keep):
            if k:
                cv2.circle(out, (int(x), int(y)), 3, (11, 158, 245), -1, cv2.LINE_AA)
    if r.face is not None:
        for (x, y), c in zip(r.face, r.face_conf):
            if c >= min_conf:
                cv2.circle(out, (int(x), int(y)), 2, (60, 60, 240), -1, cv2.LINE_AA)
    return out
