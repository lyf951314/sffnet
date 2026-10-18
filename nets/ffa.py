from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_kernels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return orthonormal 2x2 Haar kernels ordered as LL, LH, HL, HH."""
    return torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, -1.0], [1.0, -1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ],
        dtype=dtype,
    ).unsqueeze(1) / 2.0


class HaarDWT(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        weight = _haar_kernels().repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"HaarDWT expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        y = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=self.channels)
        b, _, h2, w2 = y.shape
        y = y.view(b, self.channels, 4, h2, w2)
        return y, (h, w)


class HaarIDWT(nn.Module):
    """Inverse transform paired with :class:`HaarDWT`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        weight = _haar_kernels().repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=True)

    def forward(
        self,
        coeffs: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if coeffs.ndim != 5 or coeffs.shape[1] != self.channels or coeffs.shape[2] != 4:
            raise ValueError(
                f"HaarIDWT expected [B,{self.channels},4,H,W], got {tuple(coeffs.shape)}"
            )
        b, c, _, h2, w2 = coeffs.shape
        x = coeffs.reshape(b, c * 4, h2, w2)
        x = F.conv_transpose2d(
            x,
            self.weight.to(dtype=coeffs.dtype),
            stride=2,
            groups=self.channels,
        )
        if output_size is not None:
            h, w = output_size
            x = x[:, :, :h, :w]
        return x


class HighFrequencyAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.pos_enc = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, groups=num_heads, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, groups=num_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(x + self.pos_enc(x)).chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h, w)
        k = k.view(b, self.num_heads, self.head_dim, h, w)
        v = v.view(b, self.num_heads, self.head_dim, h, w)

        def attend_axis(q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
            logits = torch.matmul(q_, k_.transpose(-2, -1)) * self.scale
            weights = logits.softmax(dim=-1)
            return torch.matmul(weights, v_)

        out_h = attend_axis(q, k, v)
        q_t = q.transpose(-2, -1)
        k_t = k.transpose(-2, -1)
        v_t = v.transpose(-2, -1)
        out_w = attend_axis(q_t, k_t, v_t).transpose(-2, -1)
        out = 0.5 * (out_h + out_w)
        return self.proj(out.reshape(b, c, h, w))


class HighLowFrequencyChannelAttention(nn.Module):
    """High/low-frequency channel recalibration producing four C-channel weights."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.channels = channels
        self.low_proj = nn.Linear(channels, channels, bias=False)
        self.high_proj = nn.Linear(channels * 3, channels, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 4, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        b, c, _, _ = ll.shape
        low_desc = F.adaptive_avg_pool2d(ll, 1).view(b, c)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_desc = F.adaptive_avg_pool2d(high_cat, 1).view(b, 3 * c)

        low_desc = F.relu(self.low_proj(low_desc), inplace=True)
        high_desc = F.relu(self.high_proj(high_desc), inplace=True)
        weights = self.mlp(torch.cat([low_desc, high_desc], dim=1))
        weights = weights.view(b, 4, c, 1, 1)

        components = torch.stack([ll, lh, hl, hh], dim=1)  # [B,4,C,H,W]
        components = components * weights
        return components.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,4,H,W]


class FrequencyDomainFeatureAwareness(nn.Module):
    """SFFNet Frequency Domain Feature Awareness (FFA) module."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT(channels)

        # Component-specific local refinement described in the FFA section.
        self.low_avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.low_max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.lh_attention = HighFrequencyAttention(channels, num_heads=num_heads)
        self.hl_attention = HighFrequencyAttention(channels, num_heads=num_heads)
        self.hh_attention = HighFrequencyAttention(channels, num_heads=num_heads)

        self.recalibration = HighLowFrequencyChannelAttention(channels)

    def forward(self, spatial_feature: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        coeffs, original_size = self.dwt(spatial_feature)
        ll, lh, hl, hh = coeffs.unbind(dim=2)

        ll = ll + self.low_avg_pool(ll) + self.low_max_pool(ll)
        lh = self.lh_attention(lh)
        hl = self.hl_attention(hl)
        hh = self.hh_attention(hh)

        refined = self.recalibration(ll, lh, hl, hh)
        return refined, original_size
