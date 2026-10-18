from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cross_attention import BidirectionalLocalCrossAttention
from .ffa import HaarIDWT


class SpatialFrequencyInteraction(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.idwt = HaarIDWT(self.channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bidirectional_cross_attention = BidirectionalLocalCrossAttention(
            dim=self.channels,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        frequency_coeffs: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if spatial_feature.ndim != 4:
            raise ValueError(f"spatial_feature must be 4D, got {tuple(spatial_feature.shape)}")
        if spatial_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} spatial channels, got {spatial_feature.shape[1]}"
            )
        if original_size is None:
            original_size = spatial_feature.shape[-2:]

        frequency_feature = self.idwt(frequency_coeffs, output_size=original_size)
        if frequency_feature.shape != spatial_feature.shape:
            raise RuntimeError(
                "SFI domain alignment failed: "
                f"spatial={tuple(spatial_feature.shape)}, frequency={tuple(frequency_feature.shape)}"
            )

        spatial_hat = self.avg_pool(spatial_feature) + self.max_pool(spatial_feature)
        frequency_hat = self.avg_pool(frequency_feature) + self.max_pool(frequency_feature)

        y_frequency_to_spatial, y_spatial_to_frequency = (
            self.bidirectional_cross_attention(
                frequency_feature=frequency_hat,
                spatial_feature=spatial_hat,
            )
        )

        # Eq. (9): F_fused = S + F + Y_F->S + Y_S->F
        return (
            spatial_feature
            + frequency_feature
            + y_frequency_to_spatial
            + y_spatial_to_frequency
        )


class AlignedSpatialFrequencyInteraction(nn.Module):
     def __init__(self, channels: int, window_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.attention = BidirectionalLocalCrossAttention(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(self, spatial_feature: torch.Tensor, frequency_feature: torch.Tensor) -> torch.Tensor:
        if spatial_feature.shape != frequency_feature.shape:
            raise ValueError("Legacy SFI inputs must have identical shapes")
        s_hat = self.avg_pool(spatial_feature) + self.max_pool(spatial_feature)
        f_hat = self.avg_pool(frequency_feature) + self.max_pool(frequency_feature)
        y_fs, y_sf = self.attention(f_hat, s_hat)
        return spatial_feature + frequency_feature + y_fs + y_sf
