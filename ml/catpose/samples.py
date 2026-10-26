"""Annotation loading and fixed train/val splits shared by every experiment.

A `Sample` is one annotated animal instance: image path, xyxy bbox, K points,
per-point visibility. Splits are written once to `ml/splits/splits.json` and
reused so that old and new checkpoints are always compared on the same
held-out instances.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "ml" / "datasets"
SPLITS_PATH = REPO_ROOT / "ml" / "splits" / "splits.json"


@dataclass
class Sample:
    key: str
    image: Path
    bbox: np.ndarray  # (4,) x0, y0, x1, y1
    pts: np.ndarray  # (K, 2)
    vis: np.ndarray  # (K,) 1.0 visible / 0.0 missing
    species: str


def load_catflw(root: Path = DATASETS / "CatFLW") -> list[Sample]:
    out = []
    for label in sorted((root / "labels").glob("*.json")):
        image = next(
            (root / "images" / f"{label.stem}{ext}" for ext in (".png", ".jpg", ".jpeg")
             if (root / "images" / f"{label.stem}{ext}").exists()),
            None,
        )
        if image is None:
            continue
        d = json.loads(label.read_text())
        pts = np.asarray(d["labels"], dtype=np.float32)
        out.append(Sample(
            key=label.stem,
            image=image,
            bbox=np.asarray(d["bounding_boxes"], dtype=np.float32),
            pts=pts,
            vis=np.ones(len(pts), dtype=np.float32),
            species="cat",
        ))
    return out


def _animalpose_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for folder in (root / "animalpose_image_part2", root / "voc_images"):
        if folder.exists():
            for p in folder.rglob("*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    index.setdefault(p.name, p)
    return index


def load_animalpose(root: Path = DATASETS / "AnimalPose") -> list[Sample]:
    """All species with an image on disk. Keys are `ap<annotation index>`."""
    payload = json.loads((root / "keypoints.json").read_text())
    species = {c["id"]: c["name"] for c in payload["categories"]}
    index = _animalpose_image_index(root)
    out = []
    for i, ann in enumerate(payload["annotations"]):
        name = payload["images"][str(ann["image_id"])]
        image = index.get(Path(name).name)
        if image is None:
            continue
        kps = np.asarray(ann["keypoints"], dtype=np.float32)
        out.append(Sample(
            key=f"ap{i}",
            image=image,
            bbox=np.asarray(ann["bbox"], dtype=np.float32),  # xyxy per dataset README
            pts=kps[:, :2].copy(),
            vis=(kps[:, 2] > 0).astype(np.float32),
            species=species[ann["category_id"]],
        ))
    return out


def all_animalpose_cat_keys(root: Path = DATASETS / "AnimalPose") -> tuple[list[str], dict[str, str]]:
    """Every cat annotation key in keypoints.json (whether downloaded or not) + key -> filename."""
    payload = json.loads((root / "keypoints.json").read_text())
    cat_id = next(c["id"] for c in payload["categories"] if c["name"] == "cat")
    keys, names = [], {}
    for i, ann in enumerate(payload["annotations"]):
        if ann["category_id"] == cat_id:
            keys.append(f"ap{i}")
            names[f"ap{i}"] = Path(payload["images"][str(ann["image_id"])]).name
    return keys, names


def _legacy_val_indices(n: int, val_ratio: float = 0.1, seed: int = 0) -> list[int]:
    # Identical to the split the first-generation checkpoints were trained with.
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[: max(1, int(n * val_ratio))]


def build_splits() -> dict:
    """Face val = legacy CatFLW val. Body val = legacy part2 val + 10% of VOC cat images."""
    face = load_catflw()
    face_val = [face[i].key for i in _legacy_val_indices(len(face))]

    cat_keys, names = all_animalpose_cat_keys()
    part2 = [k for k in cat_keys if names[k].startswith("ca")]
    voc = [k for k in cat_keys if not names[k].startswith("ca")]
    legacy_body_val = [part2[i] for i in _legacy_val_indices(len(part2))]

    voc_images = sorted({names[k] for k in voc})
    random.Random(1).shuffle(voc_images)
    held_voc_images = set(voc_images[: max(1, len(voc_images) // 10)])
    voc_val = [k for k in voc if names[k] in held_voc_images]

    body_val = legacy_body_val + voc_val
    return {
        "face_val": face_val,
        "body_val_part2": legacy_body_val,
        "body_val_voc": voc_val,
        "body_val_images": sorted({names[k] for k in body_val}),
    }


def load_splits() -> dict:
    if not SPLITS_PATH.exists():
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLITS_PATH.write_text(json.dumps(build_splits(), indent=1))
    return json.loads(SPLITS_PATH.read_text())


def face_split() -> tuple[list[Sample], list[Sample]]:
    val_keys = set(load_splits()["face_val"])
    samples = load_catflw()
    return [s for s in samples if s.key not in val_keys], [s for s in samples if s.key in val_keys]


def body_split(species: str = "cat") -> tuple[list[Sample], list[Sample]]:
    """Train = chosen species minus any instance on a held-out image. Val = cat only.

    `species` is a species name, "all", or "cat-part2" (only the 180 original
    Part II cat images, for ablations against the first-generation model).
    """
    if species == "cat-part2":
        train, val = body_split("cat")
        return [s for s in train if s.image.name.startswith("ca")], val
    splits = load_splits()
    val_keys = set(splits["body_val_part2"]) | set(splits["body_val_voc"])
    held_images = set(splits["body_val_images"])
    samples = load_animalpose()
    val = [s for s in samples if s.key in val_keys]
    train = [
        s for s in samples
        if s.image.name not in held_images and (species == "all" or s.species == species)
    ]
    return train, val
