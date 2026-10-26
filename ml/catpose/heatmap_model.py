"""Heatmap keypoint network: MobileNetV3 backbone + light FPN + soft-argmax.

Unlike the first-generation regressor (global average pool -> MLP -> x, y),
this keeps a stride-8 spatial grid. Each keypoint gets its own heatmap; the
coordinate is the probability-weighted mean position (integral regression /
DSNT), so it is differentiable and sub-cell precise. A per-keypoint
confidence falls out of how concentrated that heatmap is.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

# (builder, weights, last index of the stride-8 / stride-16 stages, channels at 8/16/32)
BACKBONES = {
    "mnv3s": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 3, 8, (24, 48, 576)),
    "mnv3l": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2, 6, 12, (40, 112, 960)),
}


def _sep_conv(ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
        nn.BatchNorm2d(ch),
        nn.Conv2d(ch, ch, 1, bias=False),
        nn.BatchNorm2d(ch),
        nn.Hardswish(inplace=True),
    )


def _lateral(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))


class HeatmapPoseNet(nn.Module):
    def __init__(self, num_keypoints: int, backbone: str = "mnv3l",
                 pretrained: bool = True, ch: int = 96) -> None:
        super().__init__()
        builder, weights, i8, i16, (c8, c16, c32) = BACKBONES[backbone]
        feats = builder(weights=weights if pretrained else None).features
        self.s8 = feats[: i8 + 1]
        self.s16 = feats[i8 + 1: i16 + 1]
        self.s32 = feats[i16 + 1:]
        self.lat8, self.lat16, self.lat32 = _lateral(c8, ch), _lateral(c16, ch), _lateral(c32, ch)
        self.refine = nn.Sequential(_sep_conv(ch), _sep_conv(ch))
        self.head = nn.Conv2d(ch, num_keypoints, 1)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor):
        c8 = self.s8(x)
        c16 = self.s16(c8)
        c32 = self.s32(c16)
        p = self.lat32(c32)
        p = self.lat16(c16) + F.interpolate(p, size=c16.shape[-2:], mode="nearest")
        p = self.lat8(c8) + F.interpolate(p, size=c8.shape[-2:], mode="nearest")
        logits = self.head(self.refine(p))  # (B, K, H, W)
        return decode(logits)


def decode(logits: torch.Tensor):
    """Soft-argmax. Returns (uv in [0,1] (B,K,2), confidence (B,K), prob (B,K,H,W))."""
    b, k, h, w = logits.shape
    prob = torch.softmax(logits.flatten(2), dim=-1).view(b, k, h, w)
    xs = (torch.arange(w, device=logits.device, dtype=prob.dtype) + 0.5) / w
    ys = (torch.arange(h, device=logits.device, dtype=prob.dtype) + 0.5) / h
    u = (prob.sum(dim=2) * xs).sum(dim=-1)
    v = (prob.sum(dim=3) * ys).sum(dim=-1)
    # Confidence: probability mass in the 3x3 cell window around the peak.
    local = F.avg_pool2d(prob, 3, stride=1, padding=1) * 9
    conf = local.flatten(2).max(dim=-1).values.clamp(0, 1)
    return torch.stack([u, v], dim=-1), conf, prob


def gaussian_targets(uv: torch.Tensor, h: int, w: int, sigma: float = 1.5) -> torch.Tensor:
    """Normalized Gaussian heatmaps centred on ground-truth uv (B,K,2) -> (B,K,H,W)."""
    xs = torch.arange(w, device=uv.device, dtype=uv.dtype) + 0.5
    ys = torch.arange(h, device=uv.device, dtype=uv.dtype) + 0.5
    gx = torch.exp(-((xs - uv[..., 0:1] * w) ** 2) / (2 * sigma ** 2))  # (B,K,W)
    gy = torch.exp(-((ys - uv[..., 1:2] * h) ** 2) / (2 * sigma ** 2))  # (B,K,H)
    g = gy.unsqueeze(-1) * gx.unsqueeze(-2)
    return g / g.flatten(2).sum(-1).clamp_min(1e-8)[..., None, None]


def pose_loss(uv, prob, target_uv, vis):
    """L1 in heatmap-cell units + KL(target || prob) regularizer, masked by visibility."""
    h, w = prob.shape[-2:]
    scale = torch.tensor([w, h], device=uv.device, dtype=uv.dtype)
    l1 = ((uv - target_uv).abs() * scale).sum(-1)  # (B,K)
    tgt = gaussian_targets(target_uv, h, w)
    kl = (tgt * (torch.log(tgt.clamp_min(1e-10)) - torch.log(prob.clamp_min(1e-10)))).flatten(2).sum(-1)
    denom = vis.sum().clamp_min(1.0)
    return ((l1 + kl) * vis).sum() / denom
