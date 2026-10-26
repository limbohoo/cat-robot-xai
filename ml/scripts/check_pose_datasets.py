#!/usr/bin/env python3
"""Validate local CatFLW and Animal Pose layouts without ML dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_catflw() -> bool:
    dataset = ROOT / "datasets" / "CatFLW"
    images = dataset / "images"
    labels = dataset / "labels"
    image_files = sorted(images.glob("*")) if images.exists() else []
    label_files = sorted(labels.glob("*.json")) if labels.exists() else []

    if not image_files or not label_files:
        print("CatFLW: pending — expected images/ and labels/ after archive extraction.")
        return False

    sample = json.loads(label_files[0].read_text())
    points = sample.get("labels", [])
    box = sample.get("bounding_boxes", [])
    valid = len(points) == 48 and len(box) == 4
    print(
        f"CatFLW: {len(image_files)} images, {len(label_files)} labels; "
        f"sample has {len(points)} landmarks and bbox length {len(box)}."
    )
    if not valid:
        raise ValueError("CatFLW sample must contain 48 landmarks and one four-value bounding box.")
    return True


def find_animalpose_annotations(dataset: Path) -> list[Path]:
    return sorted(dataset.rglob("*.json"))


def check_animalpose() -> bool:
    dataset = ROOT / "datasets" / "AnimalPose"
    annotations = find_animalpose_annotations(dataset) if dataset.exists() else []
    if not annotations:
        print("Animal Pose: pending — expected extracted COCO-style JSON annotations.")
        return False

    cat_instances = 0
    first_keypoint_count = None
    for annotation_path in annotations:
        try:
            payload = json.loads(annotation_path.read_text())
        except json.JSONDecodeError:
            continue
        categories = {item.get("id"): item.get("name", "").lower() for item in payload.get("categories", [])}
        for item in payload.get("annotations", []):
            category = categories.get(item.get("category_id"), "")
            if category == "cat":
                cat_instances += 1
                keypoints = item.get("keypoints", [])
                if keypoints and first_keypoint_count is None:
                    first_keypoint_count = (
                        len(keypoints)
                        if isinstance(keypoints[0], list)
                        else len(keypoints) // 3
                    )

    print(
        f"Animal Pose: {len(annotations)} JSON files; {cat_instances} cat instances; "
        f"sample keypoints: {first_keypoint_count or 'not found'}."
    )
    if cat_instances == 0:
        raise ValueError("No cat instances found. Check that Animal Pose Part I was extracted.")
    if first_keypoint_count != 20:
        raise ValueError("Animal Pose cat annotations must contain 20 keypoints.")
    return True


if __name__ == "__main__":
    catflw_ok = check_catflw()
    animalpose_ok = check_animalpose()
    if catflw_ok and animalpose_ok:
        print("Ready for landmark-training setup.")
