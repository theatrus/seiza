"""The restoration models' network: a residual U-Net built only from
operations that Core ML, TensorRT and ONNX Runtime's CPU provider all run
natively — 3×3 and 1×1 convolutions, stride-2 downsampling, nearest
upsampling, concatenation, addition, broadcast multiplication and SiLU —
with no normalisation layers, so tiles never disagree through per-tile
statistics.

The PSF model's network (`SharpeningUNet`) is the same U-Net told the
local PSF: a small encoder reduces the PSF stamp to a vector with strided
convolutions alone, and that vector scales and shifts the channels at the
bottleneck and after each decoder level (FiLM). Over a whole image the
vector may be given as a map, sampled from the PSF field's nodes, which
FiLM brings to each level's grid by bilinear resizing.
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


# The side of the PSF stamps the PSF model reads, as the sharpening bank
# keeps them.
STAMP_SIDE = 129


class StampEncoder(nn.Sequential):
    """A PSF stamp, `inputs` planes of `STAMP_SIDE` pixels a side, reduced to
    `features` numbers: a 3×3 convolution at full resolution, so a stamp a
    pixel or two wide is seen before any stride, then stride-2 3×3
    convolutions, each halving the side, until one pixel is left (129, 65,
    33, 17, 9, 5, 3, 2, 1). Strides alone reduce it, as no pooling is in
    the operations every backend runs."""

    def __init__(self, inputs: int = 2, width: int = 16, features: int = 64, side: int = STAMP_SIDE) -> None:
        layers: list[nn.Module] = [nn.Conv2d(inputs, width, 3, padding=1), nn.SiLU()]
        channels = width
        while side > 1:
            # Twice as many channels at each halving once the side is below
            # half the stamp's, up to `features`.
            following = min(features, channels * 2) if side < STAMP_SIDE // 2 else channels
            layers += [nn.Conv2d(channels, following, 3, stride=2, padding=1), nn.SiLU()]
            channels, side = following, (side - 1) // 2 + 1
        if channels != features:
            layers += [nn.Conv2d(channels, features, 1), nn.SiLU()]
        super().__init__(*layers)
        self.features = features


class FiLM(nn.Module):
    """Per-channel scale and shift of `channels` feature maps, from the
    stamp's vector by two 1×1 convolutions. The vector is one for the whole
    tile (1×1), or a map of it over the tile, whose scale and shift are
    brought to the feature maps' grid bilinearly, as the vector would be:
    the convolutions are linear. It starts as the identity: a scale of one
    and a shift of zero, whatever the stamp."""

    def __init__(self, features: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Conv2d(features, channels, 1)
        self.shift = nn.Conv2d(features, channels, 1)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, y: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        scale, shift = self.scale(vector), self.shift(vector)
        if scale.shape[-2:] != (1, 1) and scale.shape[-2:] != y.shape[-2:]:
            scale = nn.functional.interpolate(scale, size=y.shape[-2:], mode="bilinear", align_corners=False)
            shift = nn.functional.interpolate(shift, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return y * scale + shift


class Attention(nn.Module):
    """A transformer block over a feature map's pixels: self-attention and
    a two-layer feed-forward, each after a normalisation over a pixel's
    own channels (never over a tile, whose statistics tiles would disagree
    through) and each added back. Every pixel of the coarsest grid can
    then draw on every other in its tile, which is how a long rim is
    drawn as one thing. Both added parts start at zero, so the block
    starts as the identity. With it a tile's output depends on the whole
    tile, not only on the network's convolutional reach."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_forward = nn.LayerNorm(channels)
        self.forward_layers = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.forward_layers[-1].weight)
        nn.init.zeros_(self.forward_layers[-1].bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = y.shape
        tokens = y.flatten(2).transpose(1, 2)
        normed = self.norm_attention(tokens)
        tokens = tokens + self.attention(normed, normed, normed, need_weights=False)[0]
        tokens = tokens + self.forward_layers(self.norm_forward(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SharpeningUNet(ResidualUNet):
    """The residual U-Net with the local PSF stamp's vector (`StampEncoder`)
    scaling and shifting its channels (`FiLM`) at the bottleneck and after
    each decoder level, and with `attention` transformer blocks
    (`Attention`) at the bottleneck. It reads `inputs` image planes and
    `stamp_inputs` stamp planes; without attention its receptive field and
    tiling are the U-Net's.

    With `oversample` above 1 (a power of two) the U-Net works on a grid
    that many times finer than the image's: each pixel of the planes is
    repeated over a square of `oversample` a side (nearest upsampling, so
    nothing is interpolated for it), and the last convolution takes each
    such square of the finest features back to one pixel. What goes in and
    what comes out are still the image's size, so tiling and the exported
    graph are unchanged for whoever runs it; the first level can then
    hold what falls between pixels, such as a star a pixel or two wide,
    at `oversample` squared times the work. Its reach and its grid, in the
    image's pixels, are that many times smaller."""

    def __init__(self, inputs: int = 4, outputs: int = 1, width: int = 32, levels: int = 4, stamp_inputs: int = 2,
                 stamp_width: int = 16, features: int = 64, attention: int = 0, oversample: int = 1) -> None:
        super().__init__(inputs=inputs, outputs=outputs, width=width, levels=levels)
        if oversample < 1 or oversample & (oversample - 1) or oversample > 2 ** (levels - 1):
            raise ValueError(f"oversampling is by a power of two up to {2 ** (levels - 1)}, not {oversample}")
        widths = [width * 2**level for level in range(levels)]
        self.encoder = StampEncoder(stamp_inputs, stamp_width, features)
        self.film_bottom = FiLM(features, widths[-1])
        self.film_up = nn.ModuleList(FiLM(features, widths[i]) for i in range(levels - 1))
        self.attention = nn.ModuleList(Attention(widths[-1]) for _ in range(attention))
        self.oversample = oversample
        if oversample > 1:
            self.last = nn.Conv2d(widths[0], outputs, oversample, stride=oversample)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

    @property
    def multiple(self) -> int:
        """A tile's side must be a multiple of this, in the image's pixels."""
        return 2 ** (self.levels - 1) // self.oversample

    def forward(self, x: torch.Tensor, stamp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.mapped(x, self.encoder(stamp))

    def mapped(self, x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """The network given the stamp's vector rather than the stamp: one
        a tile, (batch, features, 1, 1), or a map of it over the tile at the
        bottleneck's grid, (batch, features, side / multiple, side /
        multiple), each value the vector where that bottleneck pixel's
        middle falls, so a PSF that changes over the image changes the
        output smoothly, wherever the tiles fall."""
        if self.oversample > 1:
            x = nn.functional.interpolate(x, scale_factor=float(self.oversample), mode="nearest")
        skips = [self.first(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        y = self.film_bottom(skips.pop(), vector)
        for block in self.attention:
            y = block(y)
        for level in reversed(range(self.levels - 1)):
            y = nn.functional.interpolate(y, scale_factor=2.0, mode="nearest")
            y = self.up[level](torch.cat([self.lift[level](y), skips[level]], dim=1))
            y = self.film_up[level](y, vector)
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


def reach(levels: int, oversample: int = 1) -> int:
    """How far the network's output looks to one side, at most: half its
    receptive field, and on the side it starts from, a pixel more at each
    finer scale than the coarsest, because nearest upsampling copies a
    coarse pixel onto the fine pixel after it. At five levels it looks 122
    pixels left and up, but 107 right and down. A network that works
    `oversample` times finer than the image (`SharpeningUNet`) looks that
    many times less far in the image's pixels, rounded up."""
    return -(-((receptive_field(levels) - 1) // 2 + 2 ** (levels - 1) - 1) // oversample)
