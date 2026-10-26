"""Export the catpose heatmap models and a YOLOv8 cat detector to ONNX for the UNO Q.

Usage:
    PYTHONPATH=ml ml/.venv/bin/python ml/scripts/export_onnx.py --out hardware/unoq/catpose-edge/python/models
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from catpose.heatmap_model import HeatmapPoseNet

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT = REPO_ROOT / "ml" / "checkpoints"


class UvConf(torch.nn.Module):
    """Drop the full probability map so the ONNX graph only returns what inference needs."""

    def __init__(self, net: HeatmapPoseNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        uv, conf, _ = self.net(x)
        return uv, conf


def export_pose(kind: str, out: Path) -> dict:
    payload = torch.load(str(CKPT / f"{kind}.pt"), map_location="cpu", weights_only=True)
    net = HeatmapPoseNet(int(payload["num_keypoints"]), payload["backbone"], pretrained=False)
    net.load_state_dict(payload["state_dict"])
    model = UvConf(net).eval()
    size = int(payload["img"])
    dummy = torch.randn(1, 3, size, size)
    path = out / f"{kind}.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["uv", "conf"],
                      opset_version=17, dynamo=False)

    with torch.inference_mode():
        ref_uv, ref_conf = model(dummy)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    uv, conf = sess.run(None, {"image": dummy.numpy()})
    err = float(np.abs(uv - ref_uv.numpy()).max())
    print(f"{kind}: {path.stat().st_size / 1e6:.1f} MB, max |uv - torch| = {err:.2e}")
    assert err < 1e-3, f"{kind} ONNX output drifted from PyTorch"
    return {"img": size, "margin": float(payload["margin"]), "flip_map": payload.get("flip_map"),
            "num_keypoints": int(payload["num_keypoints"])}


def export_yolo(weights: str, imgsz: int, out: Path) -> dict:
    from ultralytics import YOLO

    src = Path(YOLO(str(CKPT / weights)).export(format="onnx", imgsz=imgsz, simplify=True, opset=17))
    dst = out / "yolo.onnx"
    shutil.move(str(src), dst)
    print(f"yolo: {dst.stat().st_size / 1e6:.1f} MB, imgsz={imgsz}")
    return {"weights": weights, "imgsz": imgsz}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO_ROOT / "hardware" / "unoq" / "catpose-edge" / "python" / "models"))
    p.add_argument("--yolo", default="yolov8n.pt", help="yolov8n is ~3x cheaper than yolov8s on the board")
    p.add_argument("--yolo-imgsz", type=int, default=320)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"face": export_pose("face", out), "body": export_pose("body", out),
            "yolo": export_yolo(args.yolo, args.yolo_imgsz, out)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
