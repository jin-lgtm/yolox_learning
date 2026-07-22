#!/usr/bin/env python3
"""Fusion-style YOLOX head variants for reducing prediction grids."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from yolox.models.network_blocks import BaseConv, DWConv
from yolox.models.yolo_head import YOLOXHead


class YOLOXFusionHead(YOLOXHead):
    """Reduce candidate boxes by predicting on fused 1/16 and 1/32 grids only.

    Input feature maps are expected to be the standard YOLOX P3/P4/P5 outputs:
    - P3: stride 8
    - P4: stride 16
    - P5: stride 32

    The head spatially compresses P3 to the P4 resolution, optionally upsamples
    P5 to the same resolution, and predicts on:
    - fused P4-scale map (e.g. 40x40 for 640 input)
    - fused P5-scale map (e.g. 20x20 for 640 input)

    This reduces candidate boxes from 80x80 + 40x40 + 20x20 = 8400
    to 40x40 + 20x20 = 2000 for 640x640 input.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        act: str = "silu",
        depthwise: bool = False,
        use_p5_fusion: bool = True,
    ) -> None:
        hidden_channels = 256
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=[16, 32],
            in_channels=[hidden_channels, hidden_channels],
            act=act,
            depthwise=depthwise,
        )
        Conv = DWConv if depthwise else BaseConv

        self.use_p5_fusion = use_p5_fusion
        c3 = int(256 * width)
        c4 = int(512 * width)
        c5 = int(1024 * width)
        hidden = int(hidden_channels * width)

        self.p3_downsample = Conv(c3, hidden, ksize=3, stride=2, act=act)
        self.p4_lateral = BaseConv(c4, hidden, ksize=1, stride=1, act=act)
        self.p5_lateral = BaseConv(c5, hidden, ksize=1, stride=1, act=act)

        fuse40_inputs = 3 if use_p5_fusion else 2
        self.p4_fusion = torch.nn.Sequential(
            BaseConv(hidden * fuse40_inputs, hidden, ksize=1, stride=1, act=act),
            Conv(hidden, hidden, ksize=3, stride=1, act=act),
            Conv(hidden, hidden, ksize=3, stride=1, act=act),
        )

        self.p4_to_p5 = Conv(hidden, hidden, ksize=3, stride=2, act=act)
        self.p5_fusion = torch.nn.Sequential(
            BaseConv(hidden * 2, hidden, ksize=1, stride=1, act=act),
            Conv(hidden, hidden, ksize=3, stride=1, act=act),
            Conv(hidden, hidden, ksize=3, stride=1, act=act),
        )

    def _build_head_features(self, xin):
        if len(xin) != 3:
            raise ValueError(f"YOLOXFusionHead expects 3 backbone features, got {len(xin)}")

        p3, p4, p5 = xin
        p3_down = self.p3_downsample(p3)
        p4_lat = self.p4_lateral(p4)
        p5_lat = self.p5_lateral(p5)

        p4_parts = [p3_down, p4_lat]
        if self.use_p5_fusion:
            p5_up = F.interpolate(p5_lat, size=p4_lat.shape[-2:], mode="nearest")
            p4_parts.append(p5_up)
        p4_fused = self.p4_fusion(torch.cat(p4_parts, dim=1))

        p4_down = self.p4_to_p5(p4_fused)
        p5_fused = self.p5_fusion(torch.cat([p4_down, p5_lat], dim=1))
        return [p4_fused, p5_fused]

    def forward(self, xin, labels=None, imgs=None):
        return super().forward(self._build_head_features(xin), labels=labels, imgs=imgs)
