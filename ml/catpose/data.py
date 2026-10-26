"""Dataset wrappers for CatFLW and Animal Pose (cat category)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .crop import CropRect, denormalize_points, expand_bbox, normalize_points
from .model import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from .schema import ANIMALPOSE_FLIP_MAP


IMG_TF = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _pil_crop(img: Image.Image, rect: CropRect) -> Image.Image:
    return img.crop((rect.x0, rect.y0, rect.x1, rect.y1))


class CatFLWDataset(Dataset):
    """CatFLW: cat face + 48 landmarks + bbox.

    Args:
        root: path to `ml/datasets/CatFLW`.
        indices: optional subset for train/val split.
        augment: apply mild train-time augmentation.
    """

    NUM_LANDMARKS = 48

    def __init__(
        self,
        root: str | Path,
        indices: Iterable[int] | None = None,
        augment: bool = False,
    ) -> None:
        root = Path(root)
        labels_dir = root / "labels"
        images_dir = root / "images"
        self.samples: list[tuple[Path, Path]] = []
        for label_path in sorted(labels_dir.glob("*.json")):
            # CatFLW ships .png images; guard against alternate extensions.
            stem = label_path.stem
            image_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            self.samples.append((image_path, label_path))
        if indices is not None:
            indices = list(indices)
            self.samples = [self.samples[i] for i in indices]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label_path = self.samples[idx]
        payload = json.loads(label_path.read_text())
        bbox = payload["bounding_boxes"]  # already [x0, y0, x1, y1]
        pts = np.asarray(payload["labels"], dtype=np.float32)

        image = Image.open(image_path).convert("RGB")
        margin = 0.20
        if self.augment:
            margin = random.uniform(0.15, 0.30)
        rect = expand_bbox(tuple(bbox), image.size, margin=margin, square=True)
        crop = _pil_crop(image, rect)
        norm_pts = normalize_points(pts, rect)  # in crop space, [0, 1]

        if self.augment and random.random() < 0.5:
            # Horizontal flip: without a landmark-flip map for CatFLW the point
            # order becomes ambiguous, so we only flip when disabled.
            pass

        tensor = IMG_TF(crop)
        return {
            "image": tensor,
            "landmarks": torch.from_numpy(norm_pts),
            "visibility": torch.ones(self.NUM_LANDMARKS, dtype=torch.float32),
            "meta": {"path": str(image_path), "rect": rect},
        }


class AnimalPoseCatDataset(Dataset):
    """Animal Pose cat category with 20 keypoints and per-point visibility."""

    NUM_LANDMARKS = 20

    def __init__(
        self,
        root: str | Path,
        indices: Iterable[int] | None = None,
        augment: bool = False,
    ) -> None:
        root = Path(root)
        payload = json.loads((root / "keypoints.json").read_text())
        categories = {c["id"]: c for c in payload["categories"]}
        cat_id = next(cid for cid, c in categories.items() if c["name"] == "cat")
        image_lookup = payload["images"]  # dict[str, str]

        # Collect (image_path, bbox, keypoints, visibility) triples.
        samples: list[tuple[Path, list[float], np.ndarray, np.ndarray]] = []
        image_root = root / "animalpose_image_part2" / "cat"
        for annot in payload["annotations"]:
            if annot["category_id"] != cat_id:
                continue
            fname = image_lookup[str(annot["image_id"])]
            image_path = image_root / Path(fname).name
            if not image_path.exists():
                continue
            bbox_xywh = annot["bbox"]
            # AnimalPose annotations store bboxes as [x0, y0, x1, y1] despite the
            # COCO-style key. Detect XYWH by checking sanity against image later.
            kps = np.asarray(annot["keypoints"], dtype=np.float32)  # (20, 3)
            visibility = (kps[:, 2] > 0).astype(np.float32)
            samples.append((image_path, bbox_xywh, kps[:, :2], visibility))

        if indices is not None:
            indices = list(indices)
            samples = [samples[i] for i in indices]
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _bbox_to_xyxy(bbox: list[float], img_size: tuple[int, int]) -> tuple[float, float, float, float]:
        x0, y0, a, b = bbox
        W, H = img_size
        # If (a, b) already extends beyond (x0, y0) inside the image, treat as xyxy.
        if a > x0 and b > y0 and a <= W and b <= H:
            return float(x0), float(y0), float(a), float(b)
        return float(x0), float(y0), float(x0 + a), float(y0 + b)

    def __getitem__(self, idx: int) -> dict:
        image_path, bbox_raw, pts, vis = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        bbox = self._bbox_to_xyxy(bbox_raw, image.size)
        margin = 0.15
        if self.augment:
            margin = random.uniform(0.10, 0.25)
        rect = expand_bbox(bbox, image.size, margin=margin, square=True)
        crop = _pil_crop(image, rect)
        norm_pts = normalize_points(pts, rect)

        if self.augment and random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            norm_pts[:, 0] = 1.0 - norm_pts[:, 0]
            norm_pts = norm_pts[ANIMALPOSE_FLIP_MAP]
            vis = vis[ANIMALPOSE_FLIP_MAP]

        tensor = IMG_TF(crop)
        return {
            "image": tensor,
            "landmarks": torch.from_numpy(norm_pts.astype(np.float32)),
            "visibility": torch.from_numpy(vis.astype(np.float32)),
            "meta": {"path": str(image_path), "rect": rect},
        }


def split_indices(n: int, val_ratio: float = 0.1, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    return indices[n_val:], indices[:n_val]


__all__ = [
    "CatFLWDataset",
    "AnimalPoseCatDataset",
    "IMG_TF",
    "split_indices",
    "denormalize_points",
]
