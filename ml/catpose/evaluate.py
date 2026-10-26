"""Evaluate a face or body checkpoint on the fixed validation split.

Crops come from the ground-truth bbox, so this measures the landmark model
alone (the YOLO detector is not involved).

    python -m catpose.evaluate --kind body --ckpt ml/checkpoints/body.pt
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .predictor import LandmarkPredictor
from .samples import Sample, body_split, face_split


class Metrics:
    """PCK@0.05 / PCK@0.1 (threshold = fraction of max bbox side) and NME (err / sqrt(w*h))."""

    def __init__(self) -> None:
        self.hits05 = self.hits10 = self.n = 0
        self.nme_sum = 0.0

    def add(self, pred: np.ndarray, sample: Sample) -> None:
        x0, y0, x1, y1 = sample.bbox
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        err = np.linalg.norm(pred - sample.pts, axis=1)[sample.vis > 0]
        side = max(w, h)
        self.hits05 += int((err < 0.05 * side).sum())
        self.hits10 += int((err < 0.10 * side).sum())
        self.nme_sum += float((err / np.sqrt(w * h)).sum())
        self.n += len(err)

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {
            "pck@0.05": round(100 * self.hits05 / n, 2),
            "pck@0.1": round(100 * self.hits10 / n, 2),
            "nme%": round(100 * self.nme_sum / n, 2),
            "points": self.n,
        }


def evaluate(predictor, samples: list[Sample], **predict_kwargs) -> dict:
    m = Metrics()
    for s in samples:
        img = cv2.cvtColor(cv2.imread(str(s.image)), cv2.COLOR_BGR2RGB)
        m.add(predictor.predict(img, bbox=tuple(float(v) for v in s.bbox), **predict_kwargs).landmarks, s)
    return m.summary()


def val_samples(kind: str, subset: str = "all") -> list[Sample]:
    if kind == "face":
        return face_split()[1]
    _, val = body_split("cat")
    if subset == "all":
        return val
    return [s for s in val if (s.image.name.startswith("ca")) == (subset == "part2")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["face", "body"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    pred = LandmarkPredictor(args.kind, args.ckpt, device=args.device)
    out = {}
    subsets = ["all"] if args.kind == "face" else ["all", "part2", "voc"]
    for sub in subsets:
        out[sub] = evaluate(pred, val_samples(args.kind, sub), flip_tta=not args.no_tta)
    print(json.dumps({"kind": args.kind, "ckpt": args.ckpt, **out}, indent=1))


if __name__ == "__main__":
    main()
