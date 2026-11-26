"""Contact sheets for the PSF model: what a sharpening run does to a few
chosen images, side by side at 1:1, with full-size images and measures over
the whole of each, as the noise model's sheets give (`sheets`).

The images are listed in `STORE/sheets/sharpen.json`:

    {"sheets": [{"name": "elephants-trunk-ha-deep",
                 "title": "Elephant's Trunk Nebula, Hα, 2026-10-15",
                 "input": "<product id>", "full": true,
                 "crops": [{"x": 1300, "y": 1550, "label": "The trunk's head and bright rim"}],
                 "baselines": [{"name": "other", "label": "Goal", "product": "<product id>"}]}]}

A sheet with `"denoise": "<noise run>"` has its input denoised by that run
first; a baseline of such a sheet is an array made from that denoised
input, given as `"file": "<path under STORE>.npy"` in place of a product.

The model reads an input as it reads any image: its PSF field measured from
it (the PSF pass's record, measured first where there is none), each tile
conditioned on the stamp at its middle, the point sources found on it, and
a noise map made from it. Each crop is a row: the input; the run asked to
bring every star to each of `STRENGTHS` times its width, never below the
run's floor and never wider than it is; and each baseline built, labelled
as listed. Every panel takes the input's stretch, moved to its own
background.

Sheets go to `STORE/models/sharpen/<run>/sheets/<name>.png`. A sheet marked
`"full": true` also gets, in `STORE/models/sharpen/<run>/full/`, the whole
image at `FULL_STRENGTH` as 32-bit FITS (when astropy is installed) and PNG,
and a seam map: the image run a second time with the tile grid moved half a
stride, the two outputs' difference in units of each pixel's noise, from
−SEAM_SPAN to +SEAM_SPAN. Each tile reads the stamp at its own middle, so
two grids differ by how the PSF changes over half a stride as well as by
their blending. Each baseline is written whole once, in the same stretch,
to `STORE/sheets/full-sharpen/<sheet>-<name>.png`. The run's record lists
the files and the measures of the whole image (`whole_measures`), the
run's and each baseline's, side by side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import psf_fields
from .manifest import Manifest
from .noise_model import device, image_sigma
from .sharpen_bank import fitted_stamp, saturation, stamp_width
from .sharpen_model import (LIGHT_RADIUS, RING_PEAK, clarity_step, cubic_taps, edge_band, edge_step, empirical_stamp,
                            light_centre, load_run, FWHM_PER_SIGMA, blurred_images, field_point_map, measured_stars,
                            measured_width, noise_hold_step, peak_centre,
                            ring_and_flux, rolloff_step, run_directory, settings_from, sharpen,
                            smooth_edges_step)
from .sheets import typeface, stretch
from .store import Store, write_png

SPEC = Path("sheets") / "sharpen.json"
SIDE = 800
SEAM_SPAN = 0.05
# The widths every star is asked to be brought to, as a share of its own:
# the same on every sheet, so sheets compare.
STRENGTHS = (("Each star to 0.8 of its width", 0.8), ("To 0.65", 0.65), ("To 0.5", 0.5))
FULL_STRENGTH = 0.65
# A run with separate strengths for stars and the rest is asked for
# these, from the least sharpening on the left to the most, stars and the
# rest moving together. Its full-size image, and so its whole-image
# measures, are at `FULL_NONSTELLAR`, which is the third panel: its stars
# at the width the strong goal leaves them on
# the sheets (0.46 of the input's on the Elephant's Trunk), so the two
# are compared at one star size, and the rest at 0.5, which by eye
# sharpens the nebula without the grain a harder ask brings.
FULL_NONSTELLAR = (0.46, 0.5)
NONSTELLAR_STRENGTHS = (("Stars and the rest to 0.8", (0.8, 0.8)), ("Both to 0.65", (0.65, 0.65)),
                        ("Stars 0.46, the rest 0.5", FULL_NONSTELLAR), ("Both to 0.35", (0.35, 0.35)))
# How far under a run's sharpest trained ask a panel's may fall before it
# is marked untrained.
UNTRAINED_SLACK = 0.05
# The whole image's widths are measured in cells of this many across and
# down, and at most this many stars a measure, chosen evenly.
CELLS = (4, 3)
MEASURED_STARS = 3000
# An image whose stars are far wider than most of training's is also
# sharpened binned, where they are near `Sharpener.scale_to` px across,
# and the two put together by scale (`scale_blended`): the binned run's
# change below a Gaussian of `SCALE_BLEND` times the stars' width, the
# full-size run's above it. On the Elephant's Trunk (stars 11.5 px,
# binned 3 × 3 to 3.5 px) a run narrowed the rims it could not at full
# size whatever it was asked (2026-11-18). Only from `SCALE_LEAST` up:
# binned by less, the stars stay as wide as training mostly saw.
SCALE_BLEND = 0.5
SCALE_LEAST = 2
# What a bundle's card asks of the scaled pass, the Trunk's best of
# 2026-11-18: binned toward stars 3.5 px across, the rest asked 0.35 there
# (which the floor holds at 1.5 / 3.9 = 0.38 on the Trunk), and the
# full-size image asked `SCALED_FULL`, the rest a little harder than
# `FULL_NONSTELLAR`. On an image whose stars are under 3 px the floor
# holds the rest at more than either asks.
SCALE_TO = 3.5
SCALE_REST = 0.35
SCALED_FULL = (0.46, 0.45)
# Where a listed star's light (rendered with its node's PSF, the network's
# own point plane) stands this many times over the pixel noise, the
# full-size change is kept whole, and the scaled blend fades in below it.
# Brought up from a 3 × 3 binning by a separable cubic, the binned run's
# stars came out square, with a dark moat where the cubic's negative lobes
# fall (2026-11-18): the scaled pass is for the cloud, and a star is the
# full-size run's. 0: the blend everywhere, stars too.
SCALE_STARS = 2.0


def scale_factor(width: float, scale_to: float) -> int:
    """How many pixels a side an image whose stars are `width` px across is
    binned by to bring them near `scale_to` px: 1, not at all, where that
    is under `SCALE_LEAST` or `scale_to` is 0."""
    if not (scale_to > 0 and width > 0):
        return 1
    factor = round(width / scale_to)
    return factor if factor >= SCALE_LEAST else 1


def block_mean(plane: np.ndarray, factor: int) -> np.ndarray:
    """`plane` binned `factor` × `factor` by the mean of each block's
    finite pixels, its last blocks taking what is left of it; NaN where a
    block has none."""
    height, width = plane.shape
    padded = np.full((-(-height // factor) * factor, -(-width // factor) * factor), np.nan, dtype=np.float32)
    padded[:height, :width] = plane
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / count, np.nan).astype(np.float32)


def brought_up(plane: np.ndarray, factor: int) -> np.ndarray:
    """A binned plane, finite, put back up `factor` times by Keys' cubic
    (`cubic_taps`, which a ramp comes through whole), each block's value at
    the middle of the pixels it was made from."""
    values = np.asarray(plane, dtype=np.float32)
    found = values
    for axis in (0, 1):
        positions = (np.arange(values.shape[axis] * factor) + 0.5) / factor - 0.5
        taps, weights = cubic_taps(positions, values.shape[axis])
        weights = weights.astype(np.float32)
        if axis == 0:
            found = sum(found[taps[:, k], :] * weights[:, k, None] for k in range(4))
        else:
            found = sum(found[:, taps[:, k]] * weights[None, :, k] for k in range(4))
    return np.asarray(found, dtype=np.float32)


def scale_blended(image: np.ndarray, output: np.ndarray, smaller: np.ndarray, sigma: float,
                  stars: np.ndarray | None = None) -> np.ndarray:
    """`output`, a sharpening of `image`, with the part of its change under
    a Gaussian of `sigma` px (σ) taken from `smaller`, the change of the
    binned run brought back up (`brought_up`), and the part above it kept.
    With `stars`, a share from 0 to 1 at each pixel, `output` is kept whole
    in that share (`star_share`). NaN where `output` is NaN."""
    change = np.nan_to_num(output - image)
    smaller = np.nan_to_num(smaller)
    covariance = sigma**2 * np.eye(2)
    coarse, smooth = blurred_images(change, [covariance])[0], blurred_images(smaller, [covariance])[0]
    blended = image + coarse + smaller - smooth
    if stars is not None:
        blended = stars * output + (1.0 - stars) * blended
    return np.where(np.isfinite(output), blended, output).astype(np.float32)


def star_share(point_plane: np.ndarray, noise_map: np.ndarray, least: float) -> np.ndarray:
    """Where the stars are, from 0 to 1: a pixel's rendered star light
    (`point_plane`, above the background) over `least` times its noise,
    held to 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.nan_to_num(point_plane / (least * noise_map), nan=0.0, posinf=1.0)
    return np.clip(share, 0.0, 1.0).astype(np.float32)


def render_saved(store: Store, manifest: Manifest, name: str, *, only: set[str] | None = None,
                 clarity: float = 0.0, rolloff: float = 0.0, scale_to: float = 0.0,
                 scale_blend: float = SCALE_BLEND, full_ask: Any = None, scale_rest: float = 0.0,
                 scale_stars: float = SCALE_STARS, edge: float = 0.0, smooth_edges: float = 0.0,
                 hold_noise: float = 0.0) -> dict[str, Any]:
    """Make a recorded run's sheets, those named in `only` or all, and list
    them in its record. Without a sheet list, or without Pillow, nothing is
    made and the answer says why; a sheet whose input is not built is
    skipped. With `clarity`, every output is given that much local
    contrast after the model (`clarity_step`), and the sheets and
    full-size images go beside the run's own, in `sheets-clarity-<amount>`
    and `full-clarity-<amount>`, and are not listed in its record. With
    `rolloff`, the finest part of every output's change is first held
    back under that much of its input's noise (`rolloff_step`), and the
    directories are named `-rolloff-<amount>` too. With `scale_to`, an
    image whose stars are far wider is also sharpened binned to about that
    width and the two put together by scale (`SCALE_BLEND`), and the
    directories are named `-scaled-<px>` (and `-blend-<share>` where that
    is not `SCALE_BLEND`). With `full_ask`, an ask (or a pair, for the stars
    and the rest), the full-size images and their measures are at it, in
    directories named `-full-<ask>`. With `edge`, every output is given that
    much contrast at its strong edges after the model (`edge_step`), in
    directories named `-edge-<amount>`. With `smooth_edges`, what every
    output's change varies by along an edge is held back under that much of
    its input's noise (`smooth_edges_step`), in directories named
    `-smooth-edges-<amount>`. With `hold_noise`, every output's change
    between 4 and 16 px is held back where it lies under that much of its
    band's share of the input's noise (`noise_hold_step`), in directories
    named `-hold-noise-<amount>`."""
    spec = store.root / SPEC
    if not spec.exists():
        return {"skipped": f"no sheet list at {spec}"}
    try:
        import PIL  # noqa: F401
    except ImportError:
        return {"skipped": "Pillow, which draws the labels, is not installed"}
    record, model = load_run(store, name)
    directory = run_directory(store, name)
    made, full, missing = {}, {}, []
    for sheet in json.loads(spec.read_text())["sheets"]:
        if only is not None and sheet["name"] not in only:
            continue
        suffix = ((f"-scaled-{scale_to:g}" if scale_to > 0 else "")
                  + (f"-blend-{scale_blend:g}" if scale_to > 0 and scale_blend != SCALE_BLEND else "")
                  + (f"-rest-{scale_rest:g}" if scale_to > 0 and scale_rest > 0 else "")
                  + (f"-stars-{scale_stars:g}" if scale_to > 0 and scale_stars != SCALE_STARS else "")
                  + (f"-rolloff-{rolloff:g}" if rolloff > 0 else "")
                  + (f"-hold-noise-{hold_noise:g}" if hold_noise > 0 else "")
                  + (f"-clarity-{clarity:g}" if clarity > 0 else "")
                  + (f"-edge-{edge:g}" if edge > 0 else "")
                  + (f"-smooth-edges-{smooth_edges:g}" if smooth_edges > 0 else "")
                  + ("" if full_ask is None else "-full-" + "-".join(f"{value:g}" for value in np.atleast_1d(full_ask))))
        path = directory / f"sheets{suffix}" / f"{sheet['name']}.png"
        whole = directory / f"full{suffix}" / sheet["name"] if sheet.get("full") else None
        drawn = render(store, manifest, record, model, sheet, path, whole, clarity=clarity, rolloff=rolloff,
                       scale_to=scale_to, scale_blend=scale_blend, full_ask=full_ask, scale_rest=scale_rest,
                       scale_stars=scale_stars, edge=edge, smooth_edges=smooth_edges, hold_noise=hold_noise)
        if drawn is None:
            missing.append(sheet["name"])
            continue
        made[sheet["name"]] = str(path.relative_to(store.root))
        if drawn:
            full[sheet["name"]] = {key: str(value.relative_to(store.root)) if isinstance(value, Path) else value
                                   for key, value in drawn.items()}
    if not (clarity > 0 or rolloff > 0 or scale_to > 0 or full_ask is not None or edge > 0 or smooth_edges > 0
            or hold_noise > 0):
        record = json.loads((directory / "run.json").read_text())
        record["sheets"] = {**record.get("sheets", {}), **made}
        record["full"] = {**record.get("full", {}), **full}
        (directory / "run.json").write_text(json.dumps(record, indent=2))
    elif full:
        # Beside the run's own: the files and measures in their directory,
        # as the record would list them.
        kept = directory / f"full{suffix}" / "full.json"
        before = json.loads(kept.read_text()) if kept.exists() else {}
        kept.write_text(json.dumps({**before, **full}, indent=2))
    return {"made": made, "full": {key: {k: v for k, v in value.items() if k != "measures"}
                                   for key, value in full.items()}, "missing": missing}


def built(manifest: Manifest, product_id: str | None) -> dict[str, Any] | None:
    if product_id is None:
        return None
    rows = manifest.rows("SELECT p.*, s.split, s.telescope, s.pixel_scale_arcsec FROM products p "
                         "JOIN sets s ON s.id = p.set_id WHERE p.id = ? AND p.status = 'built'", (product_id,))
    return dict(rows[0]) if rows else None


def baseline_image(store: Store, manifest: Manifest, entry: dict[str, Any]) -> np.ndarray | None:
    """A baseline's image: its built `product`, or the array kept at its
    `file` under the store, which is how a sheet whose input is no product
    (one denoised first) is given a baseline made from that input. None
    where it is not there."""
    if "file" in entry:
        kept = store.root / entry["file"]
        return np.load(kept).astype(np.float32) if kept.exists() else None
    product = built(manifest, entry["product"])
    return None if product is None else psf_fields.load_image(store, product)


def input_field(store: Store, product: dict[str, Any]) -> psf_fields.Field:
    """The input's PSF field, as the PSF pass measures it from the image
    alone; measured and kept first where the pass has no record of it."""
    if psf_fields.recorded(store, product["id"]) is None:
        psf_fields.measure(store, product)
    return psf_fields.Field(store, product["id"])


def denoised_input(store: Store, source: dict[str, Any], image: np.ndarray, noise_run: str) -> np.ndarray:
    """A sheet's input with all the noise `noise_run` finds taken out, as a
    sharpening meets an image denoised first; kept under
    `STORE/sheets/denoised/`, as every sharpening run draws the same."""
    from .noise_model import load_run as load_noise_run
    from .sharpen_clean import denoised

    kept = store.root / "sheets" / "denoised" / f"{source['id'].replace('/', '-')}--{noise_run}.npy"
    if kept.exists():
        return np.load(kept)
    record, model = load_noise_run(store, noise_run)
    out = denoised(record, model, image, device())
    out[~np.isfinite(image)] = np.nan
    kept.parent.mkdir(parents=True, exist_ok=True)
    np.save(kept, out)
    return out


def node_widths(field: psf_fields.Field) -> np.ndarray:
    """`stamp_width` of each of the field's node stamps; NaN where a node
    has none."""
    stamps = field.arrays["stamps"]
    return np.array([[stamp_width(stamps[row, column]) for column in range(stamps.shape[1])]
                     for row in range(stamps.shape[0])])


def spread_over(nodes: np.ndarray, field: psf_fields.Field, shape: tuple[int, int]) -> np.ndarray:
    """Values at the field's nodes, bilinear over an image of `shape`."""
    weights = []
    for positions, length in ((field.arrays["ys"], shape[0]), (field.arrays["xs"], shape[1])):
        weights.append(np.stack([np.interp(np.arange(length), positions, column)
                                 for column in np.eye(len(positions))], axis=1))
    return (weights[0] @ nodes @ weights[1].T).astype(np.float32)


def ratio_map(widths: np.ndarray, field: psf_fields.Field, shape: tuple[int, int], strength: float,
              floor: float) -> np.ndarray:
    """The ratio the network reads, FWHM(T) / FWHM(P): each star brought to
    `strength` of its width, never below `floor`, never above its own."""
    known = np.where(np.isfinite(widths), widths, np.nanmedian(widths))
    return spread_over(np.minimum(1.0, np.maximum(strength, floor / known)), field, shape)


class Sharpener:
    """A run's model on one image: its field, points, noise map and scales,
    able to sharpen any region of it as the whole image's run would."""

    def __init__(self, record: dict[str, Any], model: Any, image: np.ndarray, field: psf_fields.Field) -> None:
        self.record, self.model, self.image, self.field = record, model, image, field
        self.settings = settings_from(record["config"]["settings"])
        self.noise_map = image_sigma(image)
        finite = np.isfinite(image) & np.isfinite(self.noise_map)
        self.background = float(np.median(image[finite][::17]))
        self.noise = float(np.median(self.noise_map[finite][::17]))
        # The noise the measures are in: the image's own, or, where the
        # sheet denoised it first, the image's before that, so a sheet and
        # its denoised twin read alike.
        self.measure_noise = self.noise
        self.points = field.points()
        self.widths = node_widths(field)
        self.where = device()
        stamps = field.arrays["stamps"]
        self.node_stamps = np.stack([np.stack([fitted_stamp(stamps[row, column])[0]
                                               for column in range(stamps.shape[1])])
                                     for row in range(stamps.shape[0])])
        self.ys, self.xs = field.arrays["ys"].astype(np.float64), field.arrays["xs"].astype(np.float64)
        # Local contrast added after the model (`clarity_step`). 0: none.
        self.clarity = 0.0
        # Contrast at strong edges added after the model, before any
        # clarity (`edge_step`). 0: none.
        self.edge = 0.0
        # The fine part of the change held back after the model, before
        # any clarity (`rolloff_step`). 0: none.
        self.rolloff = 0.0
        # What the change varies by along an edge held back after that,
        # before any contrast (`smooth_edges_step`). 0: none.
        self.smooth_edges = 0.0
        # The change between 4 and 16 px held back under its band's noise
        # after the model, after the roll-off and before the edges
        # (`noise_hold_step`). 0: none.
        self.hold_noise = 0.0
        # The stars' width the image is also sharpened at, binned, where
        # its own are far wider (`SCALE_BLEND`). 0: never.
        self.scale_to = 0.0
        self.scale_blend = SCALE_BLEND
        # The rest's ask on the binned image in place of the full size's,
        # for a run with the two. 0: the same.
        self.scale_rest = 0.0
        # Where the stars keep the full-size change whole (`SCALE_STARS`).
        self.scale_stars = SCALE_STARS
        self._smaller: Sharpener | None = None
        # The full-size image's ask in place of `FULL_NONSTELLAR` (or
        # `FULL_STRENGTH`). None: that one.
        self.full_ask: Any = None

    @property
    def factor(self) -> int:
        """How many pixels a side the image is binned by for the smaller
        scale's run; 1 for none (`scale_factor`)."""
        return scale_factor(float(np.nanmedian(self.widths)), self.scale_to)

    def smaller(self) -> Sharpener:
        """The same run on the image binned by `factor`, its PSF field,
        points and noise measured from the binned image, as any image's
        are."""
        if self._smaller is None:
            binned = block_mean(self.image, self.factor)
            field = psf_fields.Field.measured_from(binned)
            if not field.usable:
                raise ValueError(f"no PSF field on the image binned {self.factor} × {self.factor}: "
                                 f"{field.record.get('error')}")
            self._smaller = Sharpener(self.record, self.model, binned, field)
        return self._smaller

    def smaller_ratio(self, ratio: Any, smaller: Sharpener) -> Any:
        """A ratio map, or a pair of them, binned to `smaller`'s image:
        the same share of each star's width, no sharper than the run's
        floor there."""
        if isinstance(ratio, tuple):
            found = tuple(self.smaller_ratio(value, smaller) for value in ratio)
            if self.scale_rest > 0:
                # The rest asked for its own share there: the rims are the
                # binned run's, the cloud's breadth the full size's.
                found = (*found[:-1], smaller.ratio(self.scale_rest))
            return found
        known = np.where(np.isfinite(smaller.widths), smaller.widths, np.nanmedian(smaller.widths))
        floor = spread_over(np.minimum(1.0, smaller.settings.floor_px / known), smaller.field, smaller.image.shape)
        binned = np.nan_to_num(block_mean(ratio, self.factor), nan=1.0)
        return np.minimum(1.0, np.maximum(binned, floor)).astype(np.float32)

    def scaled(self, output: np.ndarray, ratio: Any, top: int, left: int, bottom: int, right: int) -> np.ndarray:
        """A region's full-size output put together with the binned run's
        over the same region (`scale_blended`)."""
        factor, smaller = self.factor, self.smaller()
        inference = self.record["inference"]
        pad, grid = inference["margin"] + 16, inference.get("multiple", 1)
        height, width = smaller.image.shape
        small_top, small_left = max(0, top // factor - pad) // grid * grid, max(0, left // factor - pad) // grid * grid
        small_bottom, small_right = min(height, -(-bottom // factor) + pad), min(width, -(-right // factor) + pad)
        made = smaller.run(self.smaller_ratio(ratio, smaller), small_top, small_left, small_bottom, small_right)
        up = brought_up(np.nan_to_num(made - smaller.image[small_top:small_bottom, small_left:small_right]), factor)
        rows = slice(top - small_top * factor, bottom - small_top * factor)
        columns = slice(left - small_left * factor, right - small_left * factor)
        sigma = self.scale_blend * float(np.nanmedian(self.widths))
        stars = None
        if self.scale_stars > 0:
            points = self.points.copy()
            points[:, 0] -= left
            points[:, 1] -= top
            plane = field_point_map(points, self.ys - top, self.xs - left, self.node_stamps, output.shape,
                                    self.background)
            stars = star_share(plane, self.noise_map[top:bottom, left:right], self.scale_stars)
        return scale_blended(self.image[top:bottom, left:right], output, up[rows, columns], sigma, stars)

    def ratio(self, strength: Any) -> Any:
        """The ratio map, or the stars' and the rest's, for a strength or a
        pair of them."""
        if isinstance(strength, tuple):
            return tuple(self.ratio(value) for value in strength)
        return ratio_map(self.widths, self.field, self.image.shape, strength, self.settings.floor_px)

    @property
    def strengths(self) -> tuple[tuple[str, Any], ...]:
        """The panels' labels and asks. An ask under the sharpest the run's
        known-kernel windows were drawn at (one over its widest kernel
        ratio) is one it was never trained on, and its panel says so: a
        run asked for 0.3 when trained down to 0.5 blurred the nebula and
        lost faint stars, and was read as a bad run."""
        floor = 1.0 / max(self.settings.kernel_ratios) - UNTRAINED_SLACK
        found = []
        for label, ask in NONSTELLAR_STRENGTHS if self.settings.nonstellar else STRENGTHS:
            lowest = min(ask) if isinstance(ask, tuple) else ask
            found.append((f"{label} (below the trained range)" if lowest < floor else label, ask))
        return tuple(found)

    @property
    def full_strength(self) -> Any:
        if self.full_ask is not None:
            return self.full_ask
        return FULL_NONSTELLAR if self.settings.nonstellar else FULL_STRENGTH

    def run(self, ratio: Any, top: int = 0, left: int = 0, bottom: int | None = None,
            right: int | None = None) -> np.ndarray:
        """The region's output, with the whole image's scales, PSF field and
        points, the field's vectors interpolated to every pixel (`sharpen`'s
        `grid`)."""
        bottom = self.image.shape[0] if bottom is None else bottom
        right = self.image.shape[1] if right is None else right
        points = self.points.copy()
        points[:, 0] -= left
        points[:, 1] -= top
        noise_map = np.ascontiguousarray(self.noise_map[top:bottom, left:right])
        image = np.ascontiguousarray(self.image[top:bottom, left:right])
        output = sharpen(self.model, image,
                         lambda x, y: fitted_stamp(self.field.stamp_at(x + left, y + top))[0],
                         ratio=cut(ratio, (slice(top, bottom), slice(left, right))), points=points,
                         noise_map=noise_map, background=self.background, noise=self.noise,
                         inference=self.record["inference"], residual=self.settings.residual, where=self.where,
                         grid=(self.ys - top, self.xs - left, self.node_stamps))
        if self.factor > 1:
            output = self.scaled(output, ratio, top, left, bottom, right)
        output = rolloff_step(output, image, self.rolloff, noise_map)
        output = noise_hold_step(output, image, self.hold_noise, noise_map)
        width = float(np.nanmedian(self.widths))
        output = smooth_edges_step(output, image, self.smooth_edges, width, points, noise_map)
        output = edge_step(output, self.edge, width, points, noise_map)
        return clarity_step(output, self.clarity, width, points, noise_map)


def render(store: Store, manifest: Manifest, record: dict[str, Any], model: Any, sheet: dict[str, Any], path: Path,
           whole: Path | None = None, clarity: float = 0.0, rolloff: float = 0.0, scale_to: float = 0.0,
           scale_blend: float = SCALE_BLEND, full_ask: Any = None, scale_rest: float = 0.0,
           scale_stars: float = SCALE_STARS, edge: float = 0.0, smooth_edges: float = 0.0,
           hold_noise: float = 0.0) -> dict[str, Any] | None:
    """Draw one sheet to `path`, and with `whole`, the stem of its full-size
    files, write them and measure them. None when its input is not built."""
    from PIL import Image, ImageDraw, ImageFont

    source = built(manifest, sheet["input"])
    if source is None:
        return None
    image = psf_fields.load_image(store, source)
    field = input_field(store, source)
    if not field.usable:
        return None
    if sheet.get("denoise"):
        finite = np.isfinite(image)
        before = float(np.median(image_sigma(image)[finite][::17]))
        image = denoised_input(store, source, image, sheet["denoise"])
    runner = Sharpener(record, model, image, field)
    runner.clarity = clarity
    runner.edge = edge
    runner.rolloff = rolloff
    runner.smooth_edges = smooth_edges
    runner.hold_noise = hold_noise
    runner.scale_to, runner.scale_blend, runner.full_ask = scale_to, scale_blend, full_ask
    runner.scale_rest, runner.scale_stars = scale_rest, scale_stars
    if sheet.get("denoise"):
        runner.measure_noise = before
    look = stretch(image)
    image_background = float(np.nanmedian(image[::31, ::31]))
    baselines = []
    for entry in sheet.get("baselines", ()):
        other = baseline_image(store, manifest, entry)
        if other is None:
            continue
        if other.shape != image.shape:
            raise ValueError(f"baseline {entry['name']} is {other.shape}, its sheet's input {image.shape}")
        other[np.isnan(image)] = np.nan
        baselines.append((entry, other, float(np.nanmedian(other[::31, ::31]))))

    side = sheet.get("side", SIDE)
    height, width = image.shape
    ratios = {strength: runner.ratio(strength) for _, strength in runner.strengths}
    inference = record["inference"]
    pad = inference["margin"] + 16
    grid = inference.get("multiple", 1)
    labels = ["Input", *(label for label, _ in runner.strengths), *(entry["label"] for entry, _, _ in baselines)]
    rows = []
    for crop in sheet["crops"]:
        x, y = min(crop["x"], width - side), min(crop["y"], height - side)
        # From a corner on the model's grid, with its reach about the crop,
        # which gives the whole image's output inside it.
        top, left = max(0, y - pad) // grid * grid, max(0, x - pad) // grid * grid
        bottom, right = min(height, y + side + pad), min(width, x + side + pad)
        inner = (slice(y - top, y - top + side), slice(x - left, x - left + side))
        panels = [image[y:y + side, x:x + side]]
        panels += [runner.run(ratios[strength], top, left, bottom, right)[inner] for _, strength in runner.strengths]
        tiles = [look(panel, image_background) for panel in panels]
        tiles += [look(other[y:y + side, x:x + side], background_of) for _, other, background_of in baselines]
        rows.append((f"{crop['label']}: 1:1 crop at x {x}, y {y}", tiles))

    gap, label_height, title_height, row_label = 12, 44, 64, 34
    columns = len(labels)
    canvas = Image.new("L", (columns * side + (columns + 1) * gap,
                             title_height + len(rows) * (row_label + label_height + side + gap) + gap), 18)
    draw = ImageDraw.Draw(canvas)
    font, small, big = (typeface(ImageFont, size) for size in (26, 22, 32))
    widths = runner.widths[np.isfinite(runner.widths)]
    title = (f"{sheet['title']}: {record['name']} ({record['parameters'] / 1e6:.3g}M parameters); stars "
             f"{np.min(widths):.1f} to {np.max(widths):.1f} px, floor {runner.settings.floor_px:g} px"
             + (f"; also binned {runner.factor} × {runner.factor}" if runner.factor > 1 else ""))
    draw.text((gap, 16), title, fill=235, font=big)
    top = title_height
    for row_title, tiles in rows:
        draw.text((gap, top + 4), row_title, fill=170, font=small)
        top += row_label
        for column, (label, tile) in enumerate(zip(labels, tiles)):
            left = gap + column * (side + gap)
            draw.text((left, top + 8), label, fill=235, font=font)
            canvas.paste(Image.fromarray((tile * 255 + 0.5).astype(np.uint8), mode="L"), (left, top + label_height))
        top += label_height + side + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, np.asarray(canvas))
    if whole is None:
        return {}
    written = full_size(runner, ratios.get(runner.full_strength, runner.ratio(runner.full_strength)),
                        lambda x: look(x, image_background), whole)
    valid = np.isfinite(image) & ~saturation(image, field.record.get("clip_level"))
    measures = {"run": whole_measures(image, written.pop("output"), runner, valid)}
    for entry, other, background_of in baselines:
        shared = store.root / "sheets" / "full-sharpen" / f"{sheet['name']}-{entry['name']}.png"
        shared.parent.mkdir(parents=True, exist_ok=True)
        write_png(shared, (look(other, background_of) * 255 + 0.5).astype(np.uint8))
        written[f"baseline-{entry['name']}"] = shared
        measures[entry["name"]] = whole_measures(image, other, runner, valid)
    written["strength"] = runner.full_strength
    written["measures"] = measures
    return written


def cut(ratio: Any, region: tuple[slice, slice]) -> Any:
    """A ratio map, or a pair of them, cut to `region`."""
    if isinstance(ratio, tuple):
        return tuple(np.ascontiguousarray(value[region]) for value in ratio)
    return np.ascontiguousarray(ratio[region])


def full_size(runner: Sharpener, ratio: Any, look: Any, stem: Path) -> dict[str, Any]:
    """Write the whole image sharpened at `ratio` as FITS and, through
    `look`, as PNG, and its seam map, at `stem` with their suffixes; answer
    their paths, the seams' size in units of each pixel's noise, and the
    output itself (under "output")."""
    output = runner.run(ratio)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {"image": stem.with_suffix(".png")}
    write_png(written["image"], (look(output) * 255 + 0.5).astype(np.uint8))
    try:
        from astropy.io import fits

        written["fits"] = stem.with_suffix(".fits")
        fits.PrimaryHDU(output.astype(np.float32)).writeto(written["fits"], overwrite=True)
    except ImportError:
        written.pop("fits")
    inference = runner.record["inference"]
    grid = inference.get("multiple", 1)
    shift = (inference["tile"] - inference["overlap"]) // grid * grid // 2 // grid * grid
    moved = runner.run(ratio, shift, shift)
    difference = np.full(output.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference[shift:, shift:] = (output[shift:, shift:] - moved) / runner.noise_map[shift:, shift:]
    edge = shift + inference["margin"]
    difference[:edge] = np.nan
    difference[:, :edge] = np.nan
    difference[np.isinf(difference)] = np.nan
    compared = difference[np.isfinite(difference)]
    written["seams"] = Path(f"{stem}-seams.png")
    seams = np.clip(np.nan_to_num(0.5 + difference / (2 * SEAM_SPAN), nan=0.5), 0.0, 1.0)
    write_png(written["seams"], (seams * 255 + 0.5).astype(np.uint8))
    written["seam_rms_sigma"] = float(np.sqrt(np.mean(compared ** 2))) if compared.size else None
    written["seam_max_sigma"] = float(np.max(np.abs(compared))) if compared.size else None
    written["output"] = output
    return written


def evenly(stars: np.ndarray, count: int) -> np.ndarray:
    if len(stars) <= count:
        return stars
    return stars[np.linspace(0, len(stars) - 1, count).round().astype(int)]


def spread(values: list[float]) -> list[float] | None:
    values = [value for value in values if np.isfinite(value)]
    return [round(float(np.percentile(values, q)), 4) for q in (5, 50, 95)] if values else None


def whole_measures(image: np.ndarray, output: np.ndarray, runner: Sharpener, valid: np.ndarray) -> dict[str, Any]:
    """What `output` did to `image`'s stars over the whole field: in each
    of `CELLS`, the width of its stars' mean stamp, output over input; each
    star's flux within three widths, output over input; how far each moved,
    its light (`light_centre`) and its peak; and the deepest dip in a ring
    about each bright star, in units of the image's median noise."""
    height, width = image.shape
    inner = edge_band(image.shape, 8)
    found: dict[str, Any] = {}
    cells = []
    for row in range(CELLS[1]):
        for column in range(CELLS[0]):
            top, left = row * height // CELLS[1], column * width // CELLS[0]
            bottom, right = (row + 1) * height // CELLS[1], (column + 1) * width // CELLS[0]
            local = stamp_width(runner.field.stamp_at((left + right - 1) / 2, (top + bottom - 1) / 2))
            if not local > 0:
                continue
            side = int(min(65, max(9, 2 * math.ceil(2.5 * local) + 1)))
            mask = np.zeros(image.shape, dtype=bool)
            mask[top:bottom, left:right] = True
            stars = evenly(measured_stars(runner.points, inner & mask, valid, side // 2 + 5), 400)
            if not len(stars):
                continue
            before = measured_width(empirical_stamp(image, stars, side), local)
            after = measured_width(empirical_stamp(output, stars, side), local)
            cells.append({"cell": [row, column], "stars": len(stars), "width_px": round(before, 3),
                          "ratio": round(after / before, 4) if before > 0 else None})
    found["cells"] = cells
    found["noise_gain"] = noise_gain(image, output, runner.points, float(np.nanmedian(runner.widths)))
    reblurred = [value for value in (reblur(image, output, runner, valid, cell) for cell in cells)
                 if value is not None]
    found["reblur_rms_sigma"] = spread(reblurred)
    ratios = [cell["ratio"] for cell in cells if cell["ratio"] is not None]
    found["width_ratio"] = spread(ratios)
    typical = float(np.nanmedian(runner.widths))
    stars = evenly(measured_stars(runner.points, inner, valid, int(math.ceil(3 * typical)) + 4), MEASURED_STARS)
    columns = psf_fields.DETECTION_COLUMNS
    kept, rings, moved_light, moved_peak = [], [], [], []
    radius = LIGHT_RADIUS * typical
    for star in stars:
        x, y = float(star[0]), float(star[1])
        measured = ring_and_flux(output, image, x, y, typical, runner.measure_noise)
        if measured["reference_flux"] > 0:
            kept.append(measured["flux"] / measured["reference_flux"])
        if (star[columns.index("peak")] - runner.background) / runner.measure_noise >= RING_PEAK:
            rings.append(measured["ring_profile"])
        before, after = light_centre(image, x, y, radius), light_centre(output, x, y, radius)
        if before is not None and after is not None:
            moved_light.append((after[0] - before[0], after[1] - before[1]))
        before, after = peak_centre(image, x, y), peak_centre(output, x, y)
        if before is not None and after is not None:
            moved_peak.append((after[0] - before[0], after[1] - before[1]))
    found["stars"] = len(stars)
    found["flux_kept"] = spread(kept)
    for name, shifts in (("light", moved_light), ("peak", moved_peak)):
        if shifts:
            shifts = np.array(shifts)
            found[f"moved_{name}_px"] = spread(list(np.hypot(shifts[:, 0], shifts[:, 1])))
            found[f"moved_{name}_median"] = [round(float(np.median(shifts[:, axis])), 4) for axis in (0, 1)]
    if rings:
        found["ring"] = spread(rings)
        found["rings_deeper_than_1"] = float(np.mean([ring < -1.0 for ring in rings]))
        found["bright_stars"] = len(rings)
    found["brightest"] = brightest_rings(image, output, runner)
    found["spikes"] = spikes(image, output, runner)
    found["roundness"] = roundness(image, output, runner, valid)
    found["edges"] = edges(image, output, runner)
    return found


# Edges are measured on means of `EDGE_MEAN` pixels a side: an edge is where
# those means, smoothed over `EDGE_SMOOTH` of them, change most, the
# steepest `EDGE_SHARE` of the cells no point source is within three widths
# of.
EDGE_MEAN = 2
EDGE_SMOOTH = 9
EDGE_SHARE = 0.02


def box_mean(plane: np.ndarray, side: int) -> np.ndarray:
    """The mean over `side` × `side` cells about each cell, by a summed
    table; the cells within half a side of the edges are NaN."""
    table = np.zeros((plane.shape[0] + 1, plane.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = np.cumsum(np.cumsum(plane, axis=0, dtype=np.float64), axis=1)
    out = np.full(plane.shape, np.nan, dtype=np.float32)
    half = side // 2
    total = table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]
    out[half:half + total.shape[0], half:half + total.shape[1]] = total / side**2
    return out


def edges(image: np.ndarray, output: np.ndarray, runner: Sharpener) -> dict[str, Any]:
    """What a sharpening did at the nebula's edges, away from the stars.
    The edges are the input's: where its `EDGE_MEAN`-pixel means, smoothed,
    are steepest. "gain" is the output's gradient across them over the
    input's, each the mean of the signed gradient along the edge's own
    normal, in which noise cancels: above 1, edges were steepened, their
    contrast raised. "noise" is the spread of the gradient along the edges,
    where sharpening an edge changes nothing, output over input: above the
    sky's noise gain, noise was sharpened at the edges, which shows as
    speckle."""
    height, width = image.shape
    rows, columns = height // EDGE_MEAN, width // EDGE_MEAN

    def means(plane: np.ndarray) -> np.ndarray:
        return plane[:rows * EDGE_MEAN, :columns * EDGE_MEAN].reshape(rows, EDGE_MEAN, columns, EDGE_MEAN).mean(
            axis=(1, 3))

    def gradient(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        across, down = np.full(cells.shape, np.nan, np.float32), np.full(cells.shape, np.nan, np.float32)
        across[:, 1:-1] = (cells[:, 2:] - cells[:, :-2]) / 2
        down[1:-1] = (cells[2:] - cells[:-2]) / 2
        return across, down

    before = means(image)
    filled = np.where(np.isfinite(before), before, np.nanmedian(before[::7, ::7]))
    smooth_x, smooth_y = gradient(box_mean(filled, EDGE_SMOOTH))
    steepness = np.hypot(smooth_x, smooth_y)
    free = np.isfinite(steepness) & np.isfinite(before)
    star_width = float(np.nanmedian(runner.widths))
    # A star is kept clear of by three widths, and by three more for every
    # tenfold its peak stands over `RING_PEAK` of the noise, up to fifteen:
    # a bright star's halo is the steepest thing near it, and no edge.
    peaks = (runner.points[:, psf_fields.DETECTION_COLUMNS.index("peak")] - getattr(runner, "background", 0.0)) \
        / runner.measure_noise
    with np.errstate(invalid="ignore", divide="ignore"):
        widths_out = np.clip(3.0 + 3.0 * np.log10(np.maximum(np.nan_to_num(peaks, nan=1e6) / RING_PEAK, 1.0)), 3.0, 15.0)
    for (x, y), clear in zip(runner.points[:, :2], widths_out):
        reach = int(np.ceil(clear * star_width / EDGE_MEAN)) + EDGE_SMOOTH // 2
        row, column = int(y // EDGE_MEAN), int(x // EDGE_MEAN)
        free[max(0, row - reach):row + reach + 1, max(0, column - reach):column + reach + 1] = False
    if free.sum() < 1000:
        return {"cells": int(free.sum())}
    chosen = free & (steepness >= np.quantile(steepness[free], 1.0 - EDGE_SHARE))
    normal_x, normal_y = smooth_x[chosen] / steepness[chosen], smooth_y[chosen] / steepness[chosen]
    found: dict[str, Any] = {"cells": int(chosen.sum())}
    measured = {}
    for label, cells in (("input", before), ("output", means(output))):
        across, down = gradient(cells)
        across, down = across[chosen], down[chosen]
        normal = across * normal_x + down * normal_y
        along = -across * normal_y + down * normal_x
        keep = np.isfinite(normal) & np.isfinite(along)
        centred = along[keep] - np.median(along[keep])
        measured[label] = (float(np.mean(normal[keep])), float(1.4826 * np.median(np.abs(centred))))
    if measured["input"][0] > 0 and measured["input"][1] > 0:
        found["gain"] = round(measured["output"][0] / measured["input"][0], 4)
        found["noise"] = round(measured["output"][1] / measured["input"][1], 4)
        found["steepness_sigma"] = round(measured["input"][0] / runner.measure_noise, 4)
    return found


# Stars' shapes are measured on unclipped stars at this signal to noise or
# more that stand apart, within this many typical widths of each.
ROUND_SNR = 50.0
ROUND_RADIUS = 2.0
ROUND_STARS = 1500


def ellipticity(plane: np.ndarray, x: float, y: float, radius: float) -> float | None:
    """A star's ellipticity, 1 − minor / major of its light's second
    moments about its own centre within `radius`, less the median of the
    annulus out to 1.5 `radius`: 0 for a round star."""
    reach = int(math.ceil(1.5 * radius)) + 1
    column, row = int(round(x)), int(round(y))
    if row - reach < 0 or column - reach < 0 or row + reach >= plane.shape[0] or column + reach >= plane.shape[1]:
        return None
    values = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
    if not np.isfinite(values).all():
        return None
    down, across = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    distance = np.hypot(across + column - x, down + row - y)
    light = np.where(distance <= radius, values - np.median(values[(distance > radius) & (distance <= 1.5 * radius)]),
                     0.0)
    light = np.clip(light, 0.0, None)
    total = light.sum()
    if not total > 0:
        return None
    cx, cy = (light * across).sum() / total, (light * down).sum() / total
    dx, dy = across - cx, down - cy
    xx, yy, xy = (light * dx * dx).sum() / total, (light * dy * dy).sum() / total, (light * dx * dy).sum() / total
    spread = math.hypot(xx - yy, 2 * xy)
    major, minor = (xx + yy + spread) / 2, (xx + yy - spread) / 2
    return 1.0 - math.sqrt(max(minor, 0.0) / major) if major > 0 else None


def roundness(image: np.ndarray, output: np.ndarray, runner: Sharpener, valid: np.ndarray) -> dict[str, Any]:
    """How round the stars are, before and after: the ellipticity of each
    bright star that stands apart (`ellipticity`), as its 5th, 50th and
    95th percentiles over the image for the input and the output, and the
    share of stars the output left less round than the input by 0.05 or
    more. A sharpening should leave no star pinched."""
    width = float(np.nanmedian(runner.widths))
    radius = max(ROUND_RADIUS * width, 3.0)
    columns = psf_fields.DETECTION_COLUMNS
    stars = measured_stars(runner.points, edge_band(image.shape, 8), valid, int(math.ceil(1.5 * radius)) + 2)
    stars = evenly(stars[stars[:, columns.index("snr")] >= ROUND_SNR], ROUND_STARS)
    before, after = [], []
    for x, y in stars[:, :2]:
        first, second = ellipticity(image, float(x), float(y), radius), ellipticity(output, float(x), float(y), radius)
        if first is not None and second is not None:
            before.append(first)
            after.append(second)
    if not before:
        return {"stars": 0}
    worse = np.array(after) - np.array(before)
    return {"stars": len(before), "input": spread(before), "output": spread(after),
            "worse_by_0_05": round(float(np.mean(worse >= 0.05)), 4)}


# Spikes are looked for on this many of the brightest stars that stand
# apart, stacked, from 2 to 6 widths out along arms 3 px wide.
SPIKE_STARS = 40


def spikes(image: np.ndarray, output: np.ndarray, runner: Sharpener) -> dict[str, Any]:
    """A cross of four arms about the stars, as a PSF may hold and a
    sharpening may leave standing once the round light about it is gone:
    over the median of the `SPIKE_STARS` brightest stars that stand apart,
    each less its sky, the mean along the four axis arms and along the four
    diagonals, 2 to 6 typical widths out, in the image's median noise, for
    the input and the output. Arms above diagonals are a cross."""
    width = float(np.nanmedian(runner.widths))
    reach = int(7 * width) + 4
    height, wide = image.shape
    columns = psf_fields.DETECTION_COLUMNS
    points = runner.points
    chosen: list[tuple[float, float]] = []
    for index in np.argsort(-np.nan_to_num(points[:, columns.index("peak")], nan=-np.inf)):
        x, y = float(points[index, 0]), float(points[index, 1])
        if not (reach < x < wide - reach and reach < y < height - reach):
            continue
        if any(abs(x - cx) < 2 * reach and abs(y - cy) < 2 * reach for cx, cy in chosen):
            continue
        chosen.append((x, y))
        if len(chosen) == SPIKE_STARS:
            break
    if len(chosen) < 5:
        return {"stars": len(chosen)}
    down, across = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    radius = np.hypot(across, down)
    band = (radius >= 2 * width) & (radius <= 6 * width)
    arms = band & ((np.abs(across) <= 1) | (np.abs(down) <= 1))
    diagonals = band & ((np.abs(across - down) <= 1) | (np.abs(across + down) <= 1))
    found: dict[str, Any] = {"stars": len(chosen)}
    for label, plane in (("input", image), ("output", output)):
        cuts = []
        for x, y in chosen:
            column, row = int(round(x)), int(round(y))
            cut = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
            cuts.append((cut - np.nanmedian(cut[radius > 6.5 * width])) / runner.measure_noise)
        with np.errstate(invalid="ignore"):
            stacked = np.nanmedian(np.stack(cuts), axis=0)
        found[label] = {"arms": round(float(np.nanmean(stacked[arms])), 3),
                        "diagonals": round(float(np.nanmean(stacked[diagonals])), 3)}
    return found


# Noise is compared on the faintest third of the image's blocks of this
# many pixels, in the differences of neighbouring means of `NOISE_MEAN`.
NOISE_BLOCK = 64
NOISE_MEAN = 4
# Reblur consistency is measured on this many pixels a side at most, at
# each cell's middle.
REBLUR_SIDE = 1024


def noise_gain(image: np.ndarray, output: np.ndarray, points: np.ndarray | None = None,
               star_width: float = 0.0) -> dict[str, float]:
    """The output's sky noise over the input's: at the pixel scale, the
    medians of the maps made from each image alone, and at `NOISE_MEAN`
    pixels, the robust spread of neighbouring block means' differences on
    the faintest third of the input's `NOISE_BLOCK` blocks. 1 is noise left
    as it was; below it, noise taken out; above, noise sharpened. Given the
    image's `points` and its stars' `star_width`, also "detail": the same spread
    on the brightest third of the blocks, away from every point source by
    three widths, where what differs between neighbours is the nebula's
    structure as well as noise; above the faint blocks' gain, structure
    was sharpened."""
    height, width = image.shape
    rows, columns = height // NOISE_BLOCK, width // NOISE_BLOCK
    with np.errstate(invalid="ignore"):
        coarse = np.nanmedian(image[:rows * NOISE_BLOCK, :columns * NOISE_BLOCK]
                              .reshape(rows, NOISE_BLOCK, columns, NOISE_BLOCK), axis=(1, 3))
    step = NOISE_BLOCK // NOISE_MEAN
    faint = np.repeat(np.repeat(coarse <= np.nanpercentile(coarse, 33), step, axis=0), step, axis=1)
    bright = None
    if points is not None and star_width > 0:
        bright = np.repeat(np.repeat(coarse >= np.nanpercentile(coarse, 67), step, axis=0), step, axis=1)
        reach = int(np.ceil(3 * star_width / NOISE_MEAN))
        for x, y in points[:, :2]:
            row, column = int(y // NOISE_MEAN), int(x // NOISE_MEAN)
            bright[max(0, row - reach):row + reach + 1, max(0, column - reach):column + reach + 1] = False

    def steps_spread(means: np.ndarray, where: np.ndarray) -> float:
        steps = (means[:, 1:] - means[:, :-1])[where[:, 1:] & where[:, :-1]]
        steps = steps[np.isfinite(steps)]
        return float(1.4826 * np.median(np.abs(steps - np.median(steps)))) if steps.size else float("nan")

    def measured(plane: np.ndarray) -> tuple[float, float, float]:
        fine = float(np.nanmedian(image_sigma(plane)[::9, ::9]))
        means = plane[:rows * NOISE_BLOCK, :columns * NOISE_BLOCK].reshape(
            rows * step, NOISE_MEAN, columns * step, NOISE_MEAN).mean(axis=(1, 3))
        return fine, steps_spread(means, faint), steps_spread(means, bright) if bright is not None else float("nan")

    before, after = measured(image), measured(output)
    found = {"pixel": round(after[0] / before[0], 4), "blocks": round(after[1] / before[1], 4)}
    if bright is not None and before[2] > 0:
        found["detail"] = round(after[2] / before[2], 4)
    return found


def reblur(image: np.ndarray, output: np.ndarray, runner: Sharpener, valid: np.ndarray,
           cell: dict[str, Any]) -> float | None:
    """Reblur consistency in one cell: the output blurred back by the round
    Gaussian that takes its stars' width to the input's, against the input,
    as the root mean square of their difference in the input's noise, on
    the cell's middle. A sharpening that only undid blur gives back the
    input; detail that no blur of the input hid does not. None where the
    cell's stars were not narrowed."""
    ratio, width = cell.get("ratio"), cell.get("width_px")
    if ratio is None or not 0 < ratio < 1 or not width > 0:
        return None
    height, wide = image.shape
    row, column = cell["cell"]
    top, left = row * height // CELLS[1], column * wide // CELLS[0]
    bottom, right = (row + 1) * height // CELLS[1], (column + 1) * wide // CELLS[0]
    side = min(REBLUR_SIDE, bottom - top, right - left)
    top, left = (top + bottom - side) // 2, (left + right - side) // 2
    region = (slice(top, top + side), slice(left, left + side))
    look = valid[region] & np.isfinite(output[region]) & np.isfinite(runner.noise_map[region])
    if look.mean() < 0.5:
        return None
    variance = (width / FWHM_PER_SIGMA) ** 2 * (1.0 - ratio**2)
    filled = np.where(np.isfinite(output[region]), output[region], runner.background)
    back = blurred_images(filled, [variance * np.eye(2)])[0]
    inner = edge_band(look.shape, int(3 * variance**0.5) + 1) & look
    if not inner.any():
        return None
    difference = (back - image[region])[inner] / (runner.noise_map[region][inner] * (runner.measure_noise
                                                                                      / runner.noise))
    return float(np.sqrt(np.mean(difference**2)))


# The brightest stars looked at for rings, clipped cores and neighbours
# and all, that stand apart: those the measures above leave out, and the
# ones that ring most.
BRIGHTEST = 30


def brightest_rings(image: np.ndarray, output: np.ndarray, runner: Sharpener) -> dict[str, Any]:
    """About each of the `BRIGHTEST` stars by peak that stand apart, clipped
    or not: the azimuthal mean profile less the median of an annulus 8 to
    10 typical widths out, and its deepest point 1 to 6 widths out, in the
    image's median noise, for input and output; and the output's highest
    point between that dip and 8 widths, the halo past a ring."""
    width = float(np.nanmedian(runner.widths))
    reach = int(math.ceil(10 * width)) + 2
    height, wide = image.shape
    columns = psf_fields.DETECTION_COLUMNS
    points = runner.points
    order = np.argsort(-np.nan_to_num(points[:, columns.index("peak")], nan=-np.inf))
    chosen: list[tuple[float, float]] = []
    for index in order:
        x, y = float(points[index, 0]), float(points[index, 1])
        if not (reach <= x < wide - reach and reach <= y < height - reach):
            continue
        if any(math.hypot(x - cx, y - cy) < 2 * reach for cx, cy in chosen):
            continue
        chosen.append((x, y))
        if len(chosen) == BRIGHTEST:
            break
    down, across = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    dips, halos, input_dips = [], [], []
    for x, y in chosen:
        column, row = int(round(x)), int(round(y))
        distance = np.hypot(across + column - x, down + row - y)
        bins = np.rint(distance).astype(int)
        counts = np.bincount(bins.ravel())
        sky_ring = (distance >= 8 * width) & (distance <= 10 * width)
        profiles = []
        for plane in (image, output):
            values = plane[row - reach:row + reach + 1, column - reach:column + reach + 1].astype(np.float64)
            finite = np.isfinite(values)
            if not finite[sky_ring].any():
                break
            sky = float(np.median(values[sky_ring & finite]))
            sums = np.bincount(bins[finite], values[finite] - sky, minlength=len(counts))
            seen = np.bincount(bins[finite], minlength=len(counts))
            with np.errstate(invalid="ignore", divide="ignore"):
                profiles.append(sums / seen / runner.measure_noise)
        if len(profiles) < 2:
            continue
        radii = np.arange(len(counts))
        inside = (radii >= width) & (radii <= 6 * width) & np.isfinite(profiles[1])
        if not inside.any():
            continue
        at = radii[inside][np.argmin(profiles[1][inside])]
        dips.append(float(profiles[1][at]))
        input_dips.append(float(np.nanmin(np.where(inside, profiles[0], np.nan))))
        past = (radii > at) & (radii <= 8 * width) & np.isfinite(profiles[1])
        halos.append(float(profiles[1][past].max()) if past.any() else float("nan"))
    if not dips:
        return {"stars": 0}
    return {"stars": len(dips), "dip": spread(dips), "dip_worst": round(float(min(dips)), 3),
            "deeper_than_1": float(np.mean([dip < -1.0 for dip in dips])), "input_dip": spread(input_dips),
            "halo": spread(halos)}
