#!/usr/bin/env python3
"""Render one locally stored Animal Pose cat image with its 20 keypoints."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "AnimalPose"
OUTPUT = ROOT / "outputs" / "animalpose-cat-keypoints.png"


def main() -> None:
    payload = json.loads((DATASET / "keypoints.json").read_text())
    categories = {item["id"]: item for item in payload["categories"]}
    images = payload["images"]
    cat = None
    image_path = None
    for candidate_annotation in payload["annotations"]:
        if categories[candidate_annotation["category_id"]]["name"] != "cat":
            continue
        candidate_name = images[str(candidate_annotation["image_id"])]
        candidates = list(DATASET.rglob(Path(candidate_name).name))
        if candidates:
            cat = candidate_annotation
            image_path = candidates[0]
            break
    if cat is None or image_path is None:
        raise FileNotFoundError("No labelled cat image from this annotation file was included in the download.")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    keypoints = cat["keypoints"]
    skeleton = categories[cat["category_id"]]["skeleton"]
    for start, end in skeleton:
        first, second = keypoints[start], keypoints[end]
        if first[2] and second[2]:
            draw.line((first[0], first[1], second[0], second[1]), fill="#24A8D8", width=3)
    for index, (x, y, visible) in enumerate(keypoints):
        if visible:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#FF3B30", outline="white", width=1)
            draw.text((x + 7, y - 7), str(index + 1), fill="white", stroke_width=2, stroke_fill="black")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"Rendered {image_path.relative_to(ROOT)} → {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
