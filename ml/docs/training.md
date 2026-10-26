# Retraining

The bundled `checkpoints/body.pt` and `checkpoints/face.pt` can be reproduced
with the commands below. YOLO is used as-is and not retrained.

## 1. Datasets

Put them under `datasets/` in the `ml/` folder:

```
datasets/
├── CatFLW/
│   ├── images/          2079 png
│   └── labels/          2079 json (48 points + bbox)
└── AnimalPose/
    ├── keypoints.json
    ├── animalpose_image_part2/{cat,dog,horse,sheep,cow}/
    └── voc_images/      JPEGs from PASCAL VOC 2012 referenced by keypoints.json
```

- **CatFLW**: https://www.kaggle.com/datasets/georgemartvel/catflw
  (CC BY-NC 4.0).
- **Animal Pose**: annotations and Part II images via the Google Drive links in
  https://github.com/noahcao/animal-pose-dataset
- **PASCAL VOC 2012 trainval** (1.9 GB):
  https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar (mirror of
  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). Only the 3608 images that
  `keypoints.json` references are needed. Extract them:

```bash
.venv/bin/python - <<'EOF'
import json, tarfile
from pathlib import Path
ap = json.loads(Path('datasets/AnimalPose/keypoints.json').read_text())
need = {Path(v).name for v in ap['images'].values() if not v.startswith(('ca', 'do', 'ho', 'sh', 'co'))}
out = Path('datasets/AnimalPose/voc_images'); out.mkdir(exist_ok=True)
with tarfile.open('VOCtrainval_11-May-2012.tar') as tf:
    for m in tf:
        if m.isfile() and '/JPEGImages/' in m.name and Path(m.name).name in need:
            m.name = Path(m.name).name
            tf.extract(m, out)
print('extracted', len(list(out.iterdir())))
EOF
```

Check the layout:

```bash
./scripts/run_dataset_tests.sh
```

## 2. Validation split

`splits/splits.json` fixes the held-out sets used for every reported number:
207 CatFLW faces, and 150 cat instances (20 from Animal Pose Part II + 130 from
10% of the VOC cat images, split by image). Keep it so results stay
comparable; delete it only if you want a new random split.

## 3. Train

Apple GPU (MPS) is used automatically; otherwise add `--device cuda` or
`--device cpu`.

```bash
# Body stage 1: all five species (~5963 instances), ~25 min on M5 Pro
PYTHONPATH=. .venv/bin/python -m catpose.train_hm --kind body --species all --epochs 35 --out body_all.pt

# Body stage 2: fine-tune on cats only, ~3 min
PYTHONPATH=. .venv/bin/python -m catpose.train_hm --kind body --species cat --epochs 25 --lr 3e-4 \
    --init checkpoints/exp/body_all.pt --out body_ft.pt

# Face, ~10 min
PYTHONPATH=. .venv/bin/python -m catpose.train_hm --kind face --epochs 40 --out face_hm.pt
```

Outputs go to `checkpoints/exp/`, each with a `*.history.json` of per-epoch
metrics. The best epoch is kept (body by PCK@0.1, face by NME).

| Flag | Default | Notes |
|---|---|---|
| `--kind` | required | `face` or `body` |
| `--species` | `cat` | body only: `cat`, `all`, or `cat-part2` (the 180 original images, for ablations) |
| `--backbone` | `mnv3l` | `mnv3s` is ~2x faster, less accurate; useful for CPU-only deployment |
| `--img` | `256` | input size |
| `--epochs` / `--batch-size` / `--lr` | 30 / 32 / 1e-3 | |
| `--init` | none | start from a checkpoint (fine-tuning) |
| `--workers` | 8 | data-loading processes |

## 4. Evaluate

```bash
PYTHONPATH=. .venv/bin/python -m catpose.evaluate --kind body --ckpt checkpoints/exp/body_ft.pt
PYTHONPATH=. .venv/bin/python -m catpose.evaluate --kind face --ckpt checkpoints/exp/face_hm.pt
```

Reports PCK@0.05, PCK@0.1 and NME, using ground-truth boxes (the detector is
not involved). Body results are also split into Part II and VOC subsets.

## 5. Use new weights

Copy them over `checkpoints/body.pt` / `checkpoints/face.pt`, or point to them
with `--body-ckpt` / `--face-ckpt` (realtime) or `CATPOSE_BODY_CKPT` /
`CATPOSE_FACE_CKPT` (server). Restart the server after changing weights.

## 6. Where to push next

1. Label 100–200 failure frames from your own webcam and add them to training.
2. Legs and paws are the weakest parts (64–71% PCK@0.1): try `--img 320` or a
   larger backbone.
3. Train an `--backbone mnv3s` variant for CPU-only machines.
4. Carve out a separate test set; right now the validation set is used both
   to pick the best epoch and to report results.
