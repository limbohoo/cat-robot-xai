"""Build the data pack that the Colab tutorial (ml/colab/CatPose_XAI_Tutorial.ipynb) loads.

The pack holds the deployed ONNX models, a few held-out validation images with
their ground-truth keypoints, the CatFLW mean face shape, and the training
histories, so every cell in the notebook can run on Colab without the full
datasets. It also carries the deployed rule files themselves (code/catpose_core:
reactions, interpret, faithfulness; numpy only) so the notebook imports the real
rules instead of re-typing them, and the events of the analysed test video
(real_events/) for the faithfulness section.

    PYTHONPATH=ml ml/.venv/bin/python ml/scripts/build_colab_pack.py            # full rebuild
    PYTHONPATH=ml ml/.venv/bin/python ml/scripts/build_colab_pack.py --update   # rules + events only, then re-zip
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile

import numpy as np

from catpose.samples import REPO_ROOT, body_split, face_split

OUT_DIR = REPO_ROOT / "output" / "colab"
PACK = OUT_DIR / "catpose_colab_pack"
MODELS = REPO_ROOT / "hardware" / "unoq" / "catpose-edge" / "python" / "models"
N_BODY, N_FACE = 12, 12


def _sample_record(s, kind: str, name: str) -> dict:
    return {
        "kind": kind,
        "file": name,
        "bbox": [float(v) for v in s.bbox],
        "pts": s.pts.round(2).tolist(),
        "vis": s.vis.tolist(),
    }


RULE_FILES = ("reactions.py", "interpret.py", "faithfulness.py", "report_ui.py")
REAL_EVENTS = REPO_ROOT / "output" / "reports" / "ghost" / "events.json"


def add_rules_and_events() -> None:
    code = PACK / "code" / "catpose_core"
    if code.parent.exists():
        shutil.rmtree(code.parent)
    code.mkdir(parents=True)
    (code / "__init__.py").write_text('"""The deployed rule files of catpose, copied unchanged (numpy only)."""\n')
    for f in RULE_FILES:
        shutil.copy2(REPO_ROOT / "ml" / "catpose" / f, code / f)
    (PACK / "real_events").mkdir(exist_ok=True)
    if REAL_EVENTS.exists():
        shutil.copy2(REAL_EVENTS, PACK / "real_events" / "ghost_events.json")


def write_zip() -> Path:
    zip_path = OUT_DIR / "catpose_colab_pack.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACK.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(OUT_DIR))
    print(f"{zip_path}  {zip_path.stat().st_size / 1e6:.1f} MB")
    return zip_path


def main() -> None:
    if "--update" in sys.argv:
        add_rules_and_events()
        write_zip()
        return
    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / "models").mkdir(parents=True)
    (PACK / "samples").mkdir()
    (PACK / "history").mkdir()

    for f in ("yolo.onnx", "body.onnx", "face.onnx", "meta.json"):
        shutil.copy2(MODELS / f, PACK / "models" / f)

    records = []
    _, body_val = body_split("cat")
    step = max(1, len(body_val) // N_BODY)  # spread picks across Part II and VOC images
    for i, s in enumerate(body_val[::step][:N_BODY]):
        name = f"body_{i:02d}{s.image.suffix.lower()}"
        shutil.copy2(s.image, PACK / "samples" / name)
        records.append(_sample_record(s, "body", name))

    face_train, face_val = face_split()
    step = max(1, len(face_val) // N_FACE)
    for i, s in enumerate(face_val[::step][:N_FACE]):
        name = f"face_{i:02d}{s.image.suffix.lower()}"
        shutil.copy2(s.image, PACK / "samples" / name)
        records.append(_sample_record(s, "face", name))
    (PACK / "samples" / "samples.json").write_text(json.dumps(records))

    # Mean CatFLW face in bbox-normalized coordinates (train split only).
    shapes = []
    for s in face_train:
        x0, y0, x1, y1 = s.bbox
        shapes.append((s.pts - [x0, y0]) / [max(x1 - x0, 1), max(y1 - y0, 1)])
    mean_face = np.mean(shapes, axis=0)
    (PACK / "mean_face.json").write_text(json.dumps(mean_face.round(4).tolist()))

    for f in (REPO_ROOT / "ml" / "checkpoints" / "exp").glob("*.history.json"):
        shutil.copy2(f, PACK / "history" / f.name)
    final_eval = REPO_ROOT / "ml" / "logs" / "final_eval.json"
    if final_eval.exists():
        shutil.copy2(final_eval, PACK / "history" / final_eval.name)

    add_rules_and_events()
    write_zip()
    print("mean face (x, y) per index:")
    for i, (x, y) in enumerate(mean_face):
        print(f"{i:2d}  {x:.3f}  {y:.3f}")


if __name__ == "__main__":
    main()
