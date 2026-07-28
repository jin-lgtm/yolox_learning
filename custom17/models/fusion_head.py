#!/usr/bin/env python3
"""Fusion-style YOLOX head variants for reducing prediction grids."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from yolox.models.network_blocks import BaseConv, DWConv
from yolox.models.yolo_head import YOLOXHead


class YOLOXFusionHead(YOLOXHead):
    """Fusion-style YOLOX head with configurable prediction levels.

    Input feature maps are expected to be the standard YOLOX P3/P4/P5 outputs:
    - P3: stride 8
    - P4: stride 16
    - P5: stride 32

    The head spatially compresses P3 to the P4 resolution, optionally upsamples
    P5 to the same resolution, and can predict on:
    - fused P4 only (`prediction_mode="p4"`)
    - fused P5 only (`prediction_mode="p5"`)
    - fused P4 + P5 (`prediction_mode="p4p5"`)
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        act: str = "silu",
        depthwise: bool = False,
        use_p5_fusion: bool = True,
        prediction_mode: str = "p5",
        p4_residual: bool = False,
        p5_residual: bool = False,
    ) -> None:
        hidden_channels = 256
        normalized_mode = prediction_mode.strip().lower()
        stride_map = {
            "p4": [16],
            "p5": [32],
            "p4p5": [16, 32],
        }
        if normalized_mode not in stride_map:
            raise ValueError(
                f"Unsupported prediction_mode={prediction_mode!r}. Expected one of {sorted(stride_map)}"
            )
        out_strides = stride_map[normalized_mode]
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=out_strides,
            in_channels=[hidden_channels] * len(out_strides),
            act=act,
            depthwise=depthwise,
        )
        Conv = DWConv if depthwise else BaseConv

        self.use_p5_fusion = use_p5_fusion
        self.prediction_mode = normalized_mode
        self.p4_residual = p4_residual
        self.p5_residual = p5_residual
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
        if self.p4_residual:
            p4_fused = p4_fused + p4_lat

        p4_down = self.p4_to_p5(p4_fused)
        p5_fused = self.p5_fusion(torch.cat([p4_down, p5_lat], dim=1))
        if self.p5_residual:
            p5_fused = p5_fused + p5_lat

        if self.prediction_mode == "p4":
            return [p4_fused]
        if self.prediction_mode == "p5":
            return [p5_fused]
        return [p4_fused, p5_fused]

    def forward(self, xin, labels=None, imgs=None):
        return super().forward(self._build_head_features(xin), labels=labels, imgs=imgs)
