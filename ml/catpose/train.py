"""Train a MobileNetV3-small landmark regressor for face or body keypoints.

Designed for CPU / Mac (small model, small input). Not state of the art —
enough to prove the pipeline end to end and drive a live demo. Point loss is
masked by per-keypoint visibility (Animal Pose has occluded points).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import AnimalPoseCatDataset, CatFLWDataset, split_indices
from .model import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: (B, K, 2); mask: (B, K) with 1 for visible.
    diff = (pred - target) ** 2
    per_kp = diff.sum(dim=-1)  # (B, K)
    denom = mask.sum().clamp_min(1.0)
    return (per_kp * mask).sum() / denom


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    landmarks = torch.stack([b["landmarks"] for b in batch])
    visibility = torch.stack([b["visibility"] for b in batch])
    return {"image": images, "landmarks": landmarks, "visibility": visibility}


def _mean_pixel_err(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, image_size: int) -> float:
    d = torch.linalg.norm(pred - target, dim=-1) * image_size  # (B, K), pixels in the resized crop
    denom = mask.sum().clamp_min(1.0)
    return float(((d * mask).sum() / denom).item())


def train_one(kind: str, args: argparse.Namespace) -> Path:
    device = _pick_device(prefer_mps=not args.cpu)
    print(f"[{kind}] device: {device}")

    if kind == "face":
        full = CatFLWDataset(args.catflw_root)
    else:
        full = AnimalPoseCatDataset(args.animalpose_root)

    train_idx, val_idx = split_indices(len(full), val_ratio=0.1, seed=0)
    if args.limit:
        train_idx = train_idx[: args.limit]
    train_set = type(full)(
        args.catflw_root if kind == "face" else args.animalpose_root,
        indices=train_idx,
        augment=True,
    )
    val_set = type(full)(
        args.catflw_root if kind == "face" else args.animalpose_root,
        indices=val_idx,
        augment=False,
    )
    print(f"[{kind}] train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )

    model = build_model(kind, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.02)

    best_val = float("inf")
    ckpt_dir = REPO_ROOT / "ml" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{kind}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["landmarks"].to(device)
            mask = batch["visibility"].to(device)
            pred = model(image)
            per = smooth_l1(pred, target).sum(dim=-1)  # (B, K)
            loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
            # Add a small MSE component to steady training.
            loss = loss + 0.5 * _masked_mse(pred, target, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step()
        train_loss = epoch_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_pix = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target = batch["landmarks"].to(device)
                mask = batch["visibility"].to(device)
                pred = model(image)
                per = smooth_l1(pred, target).sum(dim=-1)
                val_loss += float(((per * mask).sum() / mask.sum().clamp_min(1.0)).item())
                val_pix += _mean_pixel_err(pred, target, mask, image_size=224)
        val_loss /= max(len(val_loader), 1)
        val_pix /= max(len(val_loader), 1)
        dt = time.time() - t0
        print(
            f"[{kind}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pix@224={val_pix:.2f}  ({dt:.1f}s)"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "kind": kind,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_pix_224": val_pix,
                },
                ckpt_path,
            )
    print(f"[{kind}] saved best checkpoint -> {ckpt_path.relative_to(REPO_ROOT)}  best_val={best_val:.4f}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cat landmark model on CPU/MPS")
    p.add_argument("--kind", choices=["face", "body", "both"], default="both")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="Force CPU (skip MPS)")
    p.add_argument("--limit", type=int, default=0, help="Cap train samples per epoch (0=all)")
    p.add_argument(
        "--catflw-root",
        default=str(REPO_ROOT / "ml" / "datasets" / "CatFLW"),
    )
    p.add_argument(
        "--animalpose-root",
        default=str(REPO_ROOT / "ml" / "datasets" / "AnimalPose"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kinds = ["face", "body"] if args.kind == "both" else [args.kind]
    for k in kinds:
        train_one(k, args)


if __name__ == "__main__":
    main()
