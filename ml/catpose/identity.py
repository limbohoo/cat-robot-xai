"""Tell cats apart by coat colour, so each report is about one named cat.

The pipeline follows one cat per run. With two cats in the room, `pick_cat` only
keeps the box that overlaps the previous one, so the box can jump to the other cat
and the baseline mixes the two. This module gives every cat a colour signature
(brightness and saturation histograms of the middle of its box) and lets
`realtime --cats cats.json --cat NAME` accept only detections whose colour is
closest to that cat. Run once per cat to get one report per cat.

    # 1. learn the colours from some videos (writes cats.json and cats_preview.jpg)
    python -m catpose.identity discover sessions/test1_vid --out sessions/cats.json

    # 2. look at cats_preview.jpg, rename the cats in cats.json if you like, then
    python -m catpose.realtime --source clip.mp4 --detect --cats sessions/cats.json --cat dark

Limits: this separates cats whose coats differ in brightness or hue (black vs
silver); it does not separate two similar cats, and a very different camera
exposure can move a cat's signature. When the nearest cat is not clearly closer
than the other and there is no overlap with the previous box, the detection is
rejected rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

V_BINS, S_BINS = 8, 4
CENTER = (0.2, 0.15)  # share cut from each side / top and bottom, to leave the floor and wall out
MIN_CROP_PX = 24
MAX_DIST = 0.55  # Hellinger distance above which a detection matches no known cat
MARGIN = 0.08  # how much closer the target must be than the runner-up to win on colour alone
KEEP_IOU = 0.3  # continuity with the previous box may break a colour tie


def signature(frame_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Coat colour of the cat inside `bbox`: sqrt of the brightness and saturation histograms (unit length)."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    bw, bh = x1 - x0, y1 - y0
    ax, ay = int(max(0, x0 + CENTER[0] * bw)), int(max(0, y0 + CENTER[1] * bh))
    bx, by = int(min(w, x1 - CENTER[0] * bw)), int(min(h, y1 - CENTER[1] * bh))
    if bx - ax < MIN_CROP_PX or by - ay < MIN_CROP_PX:
        return None
    crop = frame_bgr[ay:by, ax:bx]
    scale = 64.0 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    v_hist = np.histogram(hsv[:, 2], bins=V_BINS, range=(0, 256))[0].astype(np.float64)
    s_hist = np.histogram(hsv[:, 1], bins=S_BINS, range=(0, 256))[0].astype(np.float64)
    sig = np.concatenate([v_hist / max(v_hist.sum(), 1), 0.5 * s_hist / max(s_hist.sum(), 1)])
    sig = np.sqrt(sig)
    return sig / (np.linalg.norm(sig) + 1e-9)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hellinger-style distance between two signatures (0 identical, about 1.4 opposite)."""
    return float(np.linalg.norm(a - b))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CatBook:
    """The known cats and their colour signatures."""

    def __init__(self, sigs: dict[str, np.ndarray]) -> None:
        if len(sigs) < 1:
            raise ValueError("cats.json holds no cats")
        self.sigs = sigs

    @classmethod
    def load(cls, path: str | Path) -> "CatBook":
        d = json.loads(Path(path).read_text())
        return cls({name: np.asarray(c["signature"], dtype=np.float64) for name, c in d["cats"].items()})

    @property
    def names(self) -> list[str]:
        return list(self.sigs)

    def rank(self, frame_bgr: np.ndarray, bbox) -> Optional[list[tuple[str, float]]]:
        """Cats ordered from nearest to farthest colour, or None if the box is too small to read."""
        sig = signature(frame_bgr, bbox)
        if sig is None:
            return None
        return sorted(((n, distance(sig, s)) for n, s in self.sigs.items()), key=lambda x: x[1])

    def pick(self, dets: list, frame_bgr: np.ndarray, target: str, prev_bbox=None):
        """The detection that is `target`, or None (the target is not in view, or only the other cat is).

        A detection counts when the target is the nearest known cat and close enough; if the
        runner-up is almost as near, it also needs to overlap the previous box. Among the
        candidates the one overlapping the previous box wins, else the most confident.
        """
        if target not in self.sigs:
            raise KeyError(f"unknown cat '{target}'; known: {', '.join(self.names)}")
        cands = []
        for d in dets:
            ranked = self.rank(frame_bgr, d.bbox)
            if ranked is None:
                continue
            (first, d1), rest = ranked[0], ranked[1:]
            d_target = dict(ranked)[target]
            if d_target > MAX_DIST:
                continue
            gap = (rest[0][1] - d1) if rest else 1.0
            near_prev = prev_bbox is not None and _iou(d.bbox, prev_bbox) >= KEEP_IOU
            if first == target and (gap >= MARGIN or near_prev):
                cands.append(d)
            elif first != target and near_prev and d_target - d1 < MARGIN:
                cands.append(d)  # colour is a near tie but this box continues the one we followed
        if not cands:
            return None
        if prev_bbox is not None:
            best = max(cands, key=lambda d: _iou(d.bbox, prev_bbox))
            if _iou(best.bbox, prev_bbox) >= 0.1:
                return best
        return max(cands, key=lambda d: d.confidence)


# ---------- learning the colours ----------
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _videos(paths: Iterable[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    return out


def _kmeans2(x: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Two clusters, started from the darkest and the lightest sample (deterministic)."""
    v_mid = x[:, :V_BINS] @ (np.arange(V_BINS) + 0.5)  # brightness proxy
    centers = np.stack([x[np.argmin(v_mid)], x[np.argmax(v_mid)]])
    labels = np.zeros(len(x), int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centers[None], axis=2)
        labels = d.argmin(1)
        new = np.stack([x[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(2)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, labels


def discover(videos: list[Path], out: Path, every: int, min_conf: float, max_samples: int, weights: str,
             device: str) -> None:
    from .detector import CatDetector

    det = CatDetector(weights=weights, conf_threshold=min_conf, device=device)
    sigs, thumbs, srcs = [], [], []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n_before, i = len(sigs), 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            i += 1
            if i % every:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            for d in det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                sig = signature(frame, d.bbox)
                if sig is None:
                    continue
                x0, y0, x1, y1 = (int(max(0, v_)) for v_ in d.bbox)
                th = frame[y0:y1, x0:x1]
                if th.size:
                    sigs.append(sig)
                    thumbs.append(cv2.resize(th, (96, 96)))
                    srcs.append(v.name)
        cap.release()
        print(f"{v.name}: {len(sigs) - n_before} cat boxes", flush=True)
    if len(sigs) < 10:
        raise SystemExit(f"only {len(sigs)} cat boxes found; need more footage with the cats in view")
    x = np.stack(sigs)
    centers, labels = _kmeans2(x)
    sep = distance(centers[0], centers[1])
    v_mid = [float(c[:V_BINS] @ (np.arange(V_BINS) + 0.5)) for c in centers]
    order = np.argsort(v_mid)  # darker first
    names = {int(order[0]): "dark", int(order[1]): "light"}
    cats = {}
    for k in range(2):
        mine = x[labels == k]
        spread = float(np.mean([distance(s, centers[k]) for s in mine])) if len(mine) else 0.0
        cats[names[k]] = {"signature": [round(float(v), 5) for v in centers[k]], "n": int(len(mine)),
                          "spread": round(spread, 3)}
    out.write_text(json.dumps({"cats": cats, "separation": round(sep, 3),
                               "note": "rename the keys if you like; signatures are colour histograms"}, indent=1))
    rng = np.random.default_rng(0)
    rows = []
    for k in sorted(names, key=lambda k: names[k]):
        idx = np.where(labels == k)[0]
        pick = rng.choice(idx, min(max_samples, len(idx)), replace=False) if len(idx) else []
        row = np.hstack([thumbs[i] for i in pick]) if len(pick) else np.zeros((96, 96, 3), np.uint8)
        pad = np.zeros((96, max(0, 96 * max_samples - row.shape[1]), 3), np.uint8)
        row = np.hstack([row, pad])
        cv2.putText(row, f"{names[k]} (n={len(idx)})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        rows.append(row)
    preview = out.with_name(out.stem + "_preview.jpg")
    cv2.imwrite(str(preview), np.vstack(rows))
    print(f"wrote {out} and {preview}")
    summary = ", ".join("%s n=%d" % (n, c["n"]) for n, c in cats.items())
    print(f"cat colours: {summary}; separation {sep:.2f}")
    if sep < 0.25:
        print("[warn] the two colour groups are close; the cats may not be separable by colour")


def main() -> None:
    p = argparse.ArgumentParser(description="Learn per-cat colour signatures from videos")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover", help="cluster cat boxes from videos into two cats")
    d.add_argument("videos", nargs="+", help="video files or folders")
    d.add_argument("--out", required=True)
    d.add_argument("--every", type=int, default=15, help="look at every Nth frame")
    d.add_argument("--min-conf", type=float, default=0.5)
    d.add_argument("--samples", type=int, default=12, help="thumbnails per cat in the preview")
    d.add_argument("--yolo-ckpt", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "yolov8s.pt"))
    d.add_argument("--device", default="mps")
    a = p.parse_args()
    discover(_videos(a.videos), Path(a.out), a.every, a.min_conf, a.samples, a.yolo_ckpt, a.device)


if __name__ == "__main__":
    main()
