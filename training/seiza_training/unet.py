"""The restoration models' network: a residual U-Net built only from
operations that Core ML, TensorRT and ONNX Runtime's CPU provider all run
natively — 3×3 and 1×1 convolutions, stride-2 downsampling, nearest
upsampling, concatenation, addition and SiLU — with no normalisation
layers, so tiles never disagree through per-tile statistics.
"""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Sequential):
    """Two 3×3 convolutions, each followed by SiLU."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.SiLU(),
        )


class ResidualUNet(nn.Module):
    """`levels` resolutions, `width` channels at the first and twice as many
    at each one below. The output is a residual the caller adds back."""

    def __init__(self, inputs: int = 2, outputs: int = 1, width: int = 16, levels: int = 4) -> None:
        super().__init__()
        widths = [width * 2**level for level in range(levels)]
        self.levels = levels
        self.first = Block(inputs, widths[0])
        self.down = nn.ModuleList(
            nn.Sequential(nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1), nn.SiLU(), Block(widths[i + 1], widths[i + 1]))
            for i in range(levels - 1)
        )
        self.lift = nn.ModuleList(nn.Conv2d(widths[i + 1], widths[i], 1) for i in range(levels - 1))
        self.up = nn.ModuleList(Block(2 * widths[i], widths[i]) for i in range(levels - 1))
        self.last = nn.Conv2d(widths[0], outputs, 1)
        # Start as the identity: the residual is zero until training moves it.
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    @property
    def multiple(self) -> int:
        """A tile's side must be a multiple of this."""
        return 2 ** (self.levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.first(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        y = skips.pop()
        for level in reversed(range(self.levels - 1)):
            y = nn.functional.interpolate(y, scale_factor=2.0, mode="nearest")
            y = self.up[level](torch.cat([self.lift[level](y), skips[level]], dim=1))
        return self.last(y)


def receptive_field(levels: int) -> int:
    """The network's receptive field in pixels: each 3×3 convolution widens
    it by two pixels at its own scale."""
    field = 1
    scale = 1
    field += 4 * scale  # the first block
    for _ in range(levels - 1):
        field += 2 * scale  # the stride-2 convolution, at the finer scale
        scale *= 2
        field += 4 * scale  # the block below
    for _ in range(levels - 1):
        scale //= 2
        field += 4 * scale  # each block on the way up
    return field


def reach(levels: int) -> int:
    """How far the network's output looks to one side, at most: half its
    receptive field, and on the side it starts from, a pixel more at each
    finer scale than the coarsest, because nearest upsampling copies a
    coarse pixel onto the fine pixel after it. At five levels it looks 122
    pixels left and up, but 107 right and down."""
    return (receptive_field(levels) - 1) // 2 + 2 ** (levels - 1) - 1
