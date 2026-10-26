"""Train the v2 heatmap keypoint model (face or body).

Body is trained in two stages: all five Animal Pose species share the same
20-keypoint schema, so pretraining on ~6k quadruped instances then
fine-tuning on cats beats training on cats alone.

    python -m catpose.train_hm --kind body --species all --epochs 40 --out body_all.pt
    python -m catpose.train_hm --kind body --species cat --epochs 25 --lr 3e-4 \
        --init ml/checkpoints/exp/body_all.pt --out body_cat.pt
    python -m catpose.train_hm --kind face --epochs 40 --out face_hm.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import transforms as T
from .evaluate import Metrics
from .heatmap_model import HeatmapPoseNet, pose_loss
from .samples import body_split, face_split
from .schema import ANIMALPOSE_FLIP_MAP, CATFLW_FLIP_MAP

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "ml" / "checkpoints" / "exp"


class PoseDataset(Dataset):
    def __init__(self, samples, img, margin, flip_map, train):
        self.samples, self.img, self.margin, self.flip_map, self.train = samples, img, margin, flip_map, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        image = cv2.cvtColor(cv2.imread(str(s.image)), cv2.COLOR_BGR2RGB)
        patch, p, v, m = T.augmented_crop(image, s.bbox, s.pts, s.vis, self.img, self.margin,
                                          self.flip_map, self.train)
        return (torch.from_numpy(T.to_tensor_array(patch)),
                torch.from_numpy(p / self.img),
                torch.from_numpy(v),
                torch.from_numpy(m.astype(np.float32)),
                i)


@torch.no_grad()
def validate(model, loader, samples, img, device):
    model.eval()
    metrics = Metrics()
    for x, _, _, m, idx in loader:
        uv, _, _ = model(x.to(device))
        pts = uv.cpu().numpy() * img
        for j in range(len(idx)):
            inv = T.invert(m[j].numpy())
            metrics.add(T.apply(inv, pts[j]), samples[int(idx[j])])
    return metrics.summary()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["face", "body"], required=True)
    p.add_argument("--species", default="cat", help="body only: cat | all")
    p.add_argument("--backbone", default="mnv3l", choices=["mnv3s", "mnv3l"])
    p.add_argument("--img", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    p.add_argument("--out", required=True, help="filename inside ml/checkpoints/exp/")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    if args.kind == "face":
        train_s, val_s = face_split()
        K, flip_map, margin = 48, CATFLW_FLIP_MAP, 0.20
    else:
        train_s, val_s = body_split(args.species)
        K, flip_map, margin = 20, ANIMALPOSE_FLIP_MAP, 0.15
    print(f"[{args.kind}] train={len(train_s)} val={len(val_s)} device={args.device} backbone={args.backbone}")

    train_loader = DataLoader(PoseDataset(train_s, args.img, margin, flip_map, True),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(PoseDataset(val_s, args.img, margin, None, False),
                            batch_size=args.batch_size, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    device = torch.device(args.device)
    model = HeatmapPoseNet(K, args.backbone, pretrained=args.init is None)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True)["state_dict"])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    warm = min(len(train_loader), total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out = EXP_DIR / args.out
    # Face PCK@0.1 saturates near 100%, so rank face checkpoints by NME instead.
    def score(val):
        return -val["nme%"] if args.kind == "face" else val["pck@0.1"]

    best = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, uv_t, vis, _, _ in train_loader:
            x, uv_t, vis = x.to(device), uv_t.to(device), vis.to(device)
            uv, _, prob = model(x)
            loss = pose_loss(uv, prob, uv_t, vis)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            run += loss.item()
        val = validate(model, val_loader, val_s, args.img, device)
        history.append({"epoch": epoch, "loss": run / len(train_loader), **val})
        tag = ""
        if score(val) > best:
            best = score(val)
            tag = "  *best*"
            torch.save({
                "arch": "heatmap", "kind": args.kind, "backbone": args.backbone,
                "num_keypoints": K, "img": args.img, "margin": margin, "flip_map": flip_map,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "val": val,
            }, out)
        print(f"[{args.kind}] ep {epoch:02d}/{args.epochs} loss={run / len(train_loader):.3f} "
              f"pck@0.1={val['pck@0.1']:.1f} pck@0.05={val['pck@0.05']:.1f} nme={val['nme%']:.2f}% "
              f"({time.time() - t0:.0f}s){tag}", flush=True)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    print(f"[{args.kind}] best score={best:.2f} -> {out}")


if __name__ == "__main__":
    main()
