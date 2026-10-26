"""MobileNetV3-small backbone with a landmark regression head.

Outputs normalized (x, y) coordinates in [0, 1] relative to the input crop.
Training targets are normalized the same way, so the head learns crop-space
positions and the caller maps them back to pixel space with the crop rect.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LandmarkRegressor(nn.Module):
    """MobileNetV3-small feature extractor + MLP regression head."""

    def __init__(self, num_landmarks: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576  # last channel of MobileNetV3-small features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 2),
            nn.Sigmoid(),  # keep predictions inside [0, 1]
        )
        self.num_landmarks = num_landmarks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x.view(-1, self.num_landmarks, 2)


def build_model(kind: str, pretrained: bool = True) -> LandmarkRegressor:
    if kind == "face":
        return LandmarkRegressor(num_landmarks=48, pretrained=pretrained)
    if kind == "body":
        return LandmarkRegressor(num_landmarks=20, pretrained=pretrained)
    raise ValueError(f"unknown model kind: {kind}")
