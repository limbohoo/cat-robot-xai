"""YOLOv8n-based cat detector.

Wraps Ultralytics YOLO to pick the highest-confidence "cat" instance from a
frame. Returns pixel-space bboxes in (x0, y0, x1, y1). Returns None when no
cat is detected — no fallback to other animal classes, so a dog in view will
not trigger cat-face landmarks.

The default checkpoint is `ml/checkpoints/yolov8s.pt` (COCO-pretrained,
downloaded automatically by ultralytics on first use). Cat is COCO class 15.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "ml" / "checkpoints" / "yolov8s.pt"

COCO_CAT_ID = 15

# Silence ultralytics chatty logger.
os.environ.setdefault("YOLO_VERBOSE", "False")
for name in ("ultralytics", "yolo"):
    logging.getLogger(name).setLevel(logging.ERROR)


@dataclass
class Detection:
    cls_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) pixels


class CatDetector:
    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        conf_threshold: float = 0.20,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO  # imported lazily to keep API light

        self.model = YOLO(str(weights))
        self.model.to(device)
        # Ultralytics uses its own device string handling; store for later.
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.names = self.model.names  # dict[int, str]

    def _to_bgr(self, image) -> np.ndarray:
        """Accepts a PIL image or an RGB numpy array; returns BGR for Ultralytics.

        Ultralytics treats numpy input as OpenCV-style BGR. Passing RGB swaps the
        red and blue channels, which on a test frame dropped cat confidence from
        0.85 to 0.14 (below threshold) and made the detector call it a couch.
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(image[:, :, ::-1])
        raise TypeError(f"unsupported image type: {type(image)}")

    def detect(
        self,
        image,
        target_classes: Iterable[int] = (COCO_CAT_ID,),
    ) -> List[Detection]:
        arr = self._to_bgr(image)
        results = self.model.predict(
            source=arr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for c, p, xy in zip(cls, conf, xyxy):
                if c in target_classes:
                    detections.append(
                        Detection(
                            cls_id=int(c),
                            label=self.names.get(int(c), str(c)),
                            confidence=float(p),
                            bbox=tuple(float(v) for v in xy),
                        )
                    )
        return detections

    def best_cat(self, image) -> Optional[Detection]:
        cats = self.detect(image, target_classes=(COCO_CAT_ID,))
        if not cats:
            return None
        return max(cats, key=lambda d: d.confidence)
