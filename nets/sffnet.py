from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .ffa import FrequencyDomainFeatureAwareness
from .sfa import Bottleneck, SpatialDomainFeatureAwareness
from .sfi import SpatialFrequencyInteraction


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        x = self.act(self.pointwise_bn(self.pointwise(x)))
        return x


class UShapedDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_stage4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.stage3_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fuse_stage3 = DepthwiseSeparableConv(1024, 512)

        self.up_stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stage2_reduce = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fuse_stage2 = DepthwiseSeparableConv(512, 256)

        self.up_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage1_reduce = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse_stage1 = DepthwiseSeparableConv(256, 128)

        self.stage0_reduce = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse_stage0 = DepthwiseSeparableConv(256, 128)
        self.output = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        fused_stage0: torch.Tensor,
        fused_stage1: torch.Tensor,
        fused_stage2: torch.Tensor,
        fused_stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up_stage4(stage4)
        x = self.fuse_stage3(torch.cat([x, self.stage3_reduce(fused_stage3)], dim=1))

        x = self.up_stage3(x)
        x = self.fuse_stage2(torch.cat([x, self.stage2_reduce(fused_stage2)], dim=1))

        x = self.up_stage2(x)
        x = self.fuse_stage1(torch.cat([x, self.stage1_reduce(fused_stage1)], dim=1))

        x = self.fuse_stage0(torch.cat([x, self.stage0_reduce(fused_stage0)], dim=1))
        return self.output(x)


class PredictionHead(nn.Module):
    def __init__(self, num_classes: int = 1, channels: int = 64, bn_momentum: float = 0.1) -> None:
        super().__init__()

        def branch(out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(64, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, out_channels, 1),
            )

        self.heatmap_head = branch(num_classes)
        self.size_head = branch(2)
        self.offset_head = branch(2)

    def forward(self, x: torch.Tensor):
        heatmap = self.heatmap_head(x).sigmoid()
        size = self.size_head(x)
        offset = self.offset_head(x)
        return heatmap, size, offset


class MultiScaleSpatialFrequencyBridge(nn.Module):
    def __init__(
        self,
        channels: Sequence[int] = (64, 256, 512, 1024),
        window_sizes: Sequence[int] = (16, 16, 8, 4),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(window_sizes) != 4:
            raise ValueError("SFFNet uses four FFA/SFI bridge stages")
        self.ffa_modules = nn.ModuleList(
            [FrequencyDomainFeatureAwareness(c, num_heads=num_heads) for c in channels]
        )
        self.sfi_modules = nn.ModuleList(
            [
                SpatialFrequencyInteraction(c, window_size=w, num_heads=num_heads)
                for c, w in zip(channels, window_sizes)
            ]
        )

    def forward(self, features: Iterable[torch.Tensor]):
        features = tuple(features)
        if len(features) != 4:
            raise ValueError(f"Expected four SFA features, got {len(features)}")
        fused = []
        for spatial_feature, ffa, sfi in zip(features, self.ffa_modules, self.sfi_modules):
            frequency_coeffs, original_size = ffa(spatial_feature)
            fused.append(sfi(spatial_feature, frequency_coeffs, original_size))
        return tuple(fused)


class SFFNet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.sfa = SpatialDomainFeatureAwareness(
            Bottleneck,
            (3, 4, 6, 3),
            pretrained=pretrained,
        )
        self.spatial_frequency_bridge = MultiScaleSpatialFrequencyBridge()
        self.decoder = UShapedDecoder()
        self.prediction_head = PredictionHead(num_classes=num_classes)
        self._init_detection_head()

    def freeze_backbone(self) -> None:
        for param in self.sfa.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.sfa.parameters():
            param.requires_grad = True

    def _init_detection_head(self) -> None:
        final_heatmap = self.prediction_head.heatmap_head[-1]
        nn.init.constant_(final_heatmap.weight, 0)
        nn.init.constant_(final_heatmap.bias, -2.19)

    def forward(self, x: torch.Tensor):
        stage0, stage1, stage2, stage3, stage4 = self.sfa(x)
        fused0, fused1, fused2, fused3 = self.spatial_frequency_bridge(
            (stage0, stage1, stage2, stage3)
        )
        decoded = self.decoder(fused0, fused1, fused2, fused3, stage4)
        return self.prediction_head(decoded)
