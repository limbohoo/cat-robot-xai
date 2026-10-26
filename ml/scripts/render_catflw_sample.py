#!/usr/bin/env python3
"""Render one local CatFLW face with all 48 annotated landmarks."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "CatFLW"
OUTPUT = ROOT / "outputs" / "catflw-face-landmarks.png"


def main() -> None:
    label_path = sorted((DATASET / "labels").glob("*.json"))[0]
    payload = json.loads(label_path.read_text())
    image_path = DATASET / "images" / f"{label_path.stem}.png"
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    bbox = payload["bounding_boxes"]
    draw.rectangle(bbox, outline="#24A8D8", width=3)
    for index, (x, y) in enumerate(payload["labels"], start=1):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#FF3B30", outline="white")
        draw.text((x + 4, y - 6), str(index), fill="white", stroke_width=1, stroke_fill="black")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"Rendered {image_path.relative_to(ROOT)} → {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
