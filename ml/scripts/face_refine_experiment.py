#!/usr/bin/env python3
"""Does a second face pass on a landmark-derived box recover accuracy from sloppy boxes?

Simulates the live pipeline, where the face box is estimated from body head
keypoints, by perturbing CatFLW val boxes (scale 0.7-1.6x, shift up to 20%).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catpose.crop import face_bbox_from_face_landmarks  # noqa: E402
from catpose.evaluate import Metrics  # noqa: E402
from catpose.predictor import LandmarkPredictor  # noqa: E402
from catpose.samples import face_split  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ml/checkpoints/exp/face_hm.pt"
pred = LandmarkPredictor("face", ckpt, device="mps")
_, val = face_split()
rng = np.random.default_rng(0)
names = ["gt box, 1 pass", "sloppy box, 1 pass", "sloppy box, 2 passes", "sloppy box, 3 passes"]
m = {n: Metrics() for n in names}
for s in val:
    img = cv2.cvtColor(cv2.imread(str(s.image)), cv2.COLOR_BGR2RGB)
    m[names[0]].add(pred.predict(img, bbox=s.bbox).landmarks, s)
    x0, y0, x1, y1 = s.bbox
    side = max(x1 - x0, y1 - y0)
    k = rng.uniform(0.7, 1.6)
    cx, cy = (x0 + x1) / 2 + rng.uniform(-0.2, 0.2) * side, (y0 + y1) / 2 + rng.uniform(-0.2, 0.2) * side
    box = (cx - k * side / 2, cy - k * side / 2, cx + k * side / 2, cy + k * side / 2)
    r = pred.predict(img, bbox=box)
    m[names[1]].add(r.landmarks, s)
    for n in names[2:]:
        refined = face_bbox_from_face_landmarks(r.landmarks, r.confidences)
        if refined is not None:
            r = pred.predict(img, bbox=refined)
        m[n].add(r.landmarks, s)
for n in names:
    print(f"{n:<24} {m[n].summary()}")
