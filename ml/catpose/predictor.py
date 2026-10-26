"""Load a trained checkpoint and predict landmarks on new images.

`LandmarkPredictor(kind, ckpt)` inspects the checkpoint and returns either the
heatmap predictor (v2, has `arch == "heatmap"`) or the legacy global-pool
regressor (v1). Both expose `.predict(image, bbox) -> Prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import transforms as T
from .crop import denormalize_points, expand_bbox
from .data import IMG_TF
from .heatmap_model import HeatmapPoseNet
from .model import build_model


@dataclass
class Prediction:
    kind: str  # "face" | "body"
    landmarks: np.ndarray  # (K, 2) in pixel space of the original image
    bbox: tuple[float, float, float, float]  # xyxy region the model looked at
    confidences: np.ndarray | None = None  # (K,) in [0, 1]; None for legacy models


def _as_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return image


def _full_frame(arr: np.ndarray) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(arr.shape[1]), float(arr.shape[0]))


class HeatmapPredictor:
    def __init__(self, kind: str, payload: dict, device: str = "cpu") -> None:
        self.kind = kind
        self.device = torch.device(device)
        self.img = int(payload["img"])
        self.margin = float(payload["margin"])
        self.flip_map = payload.get("flip_map")
        # Flip TTA costs ~5 ms on Apple GPU but doubles CPU time; callers may turn it off.
        self.default_tta = True
        self.model = HeatmapPoseNet(int(payload["num_keypoints"]), payload["backbone"], pretrained=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image, bbox=None, flip_tta: bool | None = None) -> Prediction:
        arr = _as_rgb_array(image)
        bbox = bbox if bbox is not None else _full_frame(arr)
        center, side = T.box_to_center_side(bbox, self.margin)
        m = T.crop_matrix(center, side, 0.0, self.img)
        patch = T.warp(arr, m, self.img)
        views = [T.to_tensor_array(patch)]
        flip_tta = self.default_tta if flip_tta is None else flip_tta
        use_flip = flip_tta and self.flip_map is not None
        if use_flip:
            views.append(T.to_tensor_array(np.ascontiguousarray(patch[:, ::-1])))
        uv, conf, _ = self.model(torch.from_numpy(np.stack(views)).to(self.device))
        uv = uv.cpu().numpy() * self.img
        conf = conf.cpu().numpy()
        pts, c = uv[0], conf[0]
        if use_flip:
            pf = uv[1].copy()
            pf[:, 0] = self.img - 1 - pf[:, 0]
            pts = (pts + pf[self.flip_map]) / 2
            c = (c + conf[1][self.flip_map]) / 2
        half = side / 2
        return Prediction(
            kind=self.kind,
            landmarks=T.apply(T.invert(m), pts),
            bbox=(float(center[0] - half), float(center[1] - half),
                  float(center[0] + half), float(center[1] + half)),
            confidences=c,
        )


class RegressionPredictor:
    """First-generation model: MobileNetV3-small + global pool + MLP."""

    # Must match the val-time crop margin each model was trained with.
    MARGIN = {"face": 0.20, "body": 0.15}

    def __init__(self, kind: str, payload, device: str = "cpu") -> None:
        self.kind = kind
        self.device = torch.device(device)
        self.model = build_model(kind, pretrained=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image, bbox=None, **_) -> Prediction:
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        pil = pil.convert("RGB")
        bbox = bbox if bbox is not None else (0.0, 0.0, float(pil.size[0]), float(pil.size[1]))
        rect = expand_bbox(bbox, pil.size, margin=self.MARGIN[self.kind], square=True)
        crop = pil.crop((rect.x0, rect.y0, rect.x1, rect.y1))
        norm = self.model(IMG_TF(crop).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
        return Prediction(
            kind=self.kind,
            landmarks=denormalize_points(norm, rect),
            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
        )


def LandmarkPredictor(kind: str, checkpoint: str | Path, device: str = "cpu"):
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("arch") == "heatmap":
        return HeatmapPredictor(kind, payload, device)
    return RegressionPredictor(kind, payload, device)
