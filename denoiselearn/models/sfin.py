"""Spatial-Frequency Interaction Network (SFIN) architecture.

Adapted from SFIN at commit 8aa3442e59cab26ac7328b7ad3aec5aaf9c67b93:
https://github.com/HeasonLee/SFIN

Copyright 2022 Heason Lee
Licensed under the Apache License, Version 2.0. See
``licenses/SFIN_LICENSE.txt``.

Only the model architecture is included. Checkpoint loading, image
normalization, and inference policy are intentionally outside this module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


_CHANNELS = 64
_NUM_BLOCKS = 8


class FourierUnit(nn.Module):
    """Apply a learned pointwise transform in the real-valued FFT domain."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels * 2 + 2,
            out_channels * 2,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, *ffted.shape[3:])

        height, width = ffted.shape[-2:]
        coords_vert = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        coords_vert = coords_vert.view(1, 1, height, 1).expand(
            batch, 1, height, width
        )
        coords_hor = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        coords_hor = coords_hor.view(1, 1, 1, width).expand(
            batch, 1, height, width
        )

        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view(batch, -1, 2, *ffted.shape[2:])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(
            ffted, s=x.shape[-2:], dim=(-2, -1), norm="ortho"
        )


class SpectralTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        half_channels = _CHANNELS // 2
        self.conv1 = nn.Conv2d(half_channels, half_channels, 3, padding=1)
        self.fu = FourierUnit(half_channels, half_channels)
        self.conv2 = nn.Conv2d(_CHANNELS, half_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        transformed = self.fu(self.conv1(x))
        return self.conv2(torch.cat((x, transformed), dim=1))


class FFC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        half_channels = _CHANNELS // 2
        self.convl2l = nn.Conv2d(half_channels, half_channels, 3, padding=1)
        self.convl2g = nn.Conv2d(half_channels, half_channels, 3, padding=1)
        self.convg2l = nn.Conv2d(half_channels, half_channels, 3, padding=1)
        self.convg2g = SpectralTransform()

    def forward(self, x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x_l, x_g = x
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class SFIB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        half_channels = _CHANNELS // 2
        self.ffc = FFC()
        self.bn_l = nn.BatchNorm2d(half_channels)
        self.bn_g = nn.BatchNorm2d(half_channels)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class ResnetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = SFIB()
        self.conv2 = SFIB()

    def forward(self, x: Tensor) -> Tensor:
        x_l, x_g = torch.chunk(x, 2, dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return torch.cat((id_l + x_l, id_g + x_g), dim=1)


class SFIN(nn.Module):
    """Original grayscale SFIN denoising architecture.

    Input and output tensors have shape ``(N, 1, H, W)``. The architecture
    preserves spatial dimensions and does not constrain the output range.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = [ResnetBlock() for _ in range(_NUM_BLOCKS)]
        self.body = nn.Sequential(*self.blocks)
        self.head_conv = nn.Conv2d(1, _CHANNELS, 3, padding=1)
        self.tail_conv = nn.Conv2d(_CHANNELS, 1, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.head_conv(x)
        x = self.body(x) + x
        return self.tail_conv(x)
