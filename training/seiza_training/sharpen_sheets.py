"""Contact sheets for the PSF model: what a sharpening run does to a few
chosen images, side by side at 1:1, with full-size images and measures over
the whole of each, as the noise model's sheets give (`sheets`).

The images are listed in `STORE/sheets/sharpen.json`:

    {"sheets": [{"name": "elephants-trunk-ha-deep",
                 "title": "Elephant's Trunk Nebula, Hα, 2026-10-15",
                 "input": "<product id>", "full": true,
                 "crops": [{"x": 1300, "y": 1550, "label": "The trunk's head and bright rim"}],
                 "baselines": [{"name": "other", "label": "Another sharpener", "product": "<product id>"}]}]}

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
from .sharpen_model import (LIGHT_RADIUS, RING_PEAK, edge_band, empirical_stamp, light_centre, load_run,
                            measured_stars, measured_width, peak_centre, ring_and_flux, run_directory, settings_from,
                            sharpen)
from .sheets import typeface, stretch
from .store import Store, write_png

SPEC = Path("sheets") / "sharpen.json"
SIDE = 800
SEAM_SPAN = 0.05
# The widths every star is asked to be brought to, as a share of its own:
# the same on every sheet, so sheets compare.
STRENGTHS = (("Each star to 0.8 of its width", 0.8), ("To 0.65", 0.65), ("To 0.5", 0.5))
FULL_STRENGTH = 0.65
# The whole image's widths are measured in cells of this many across and
# down, and at most this many stars a measure, chosen evenly.
CELLS = (4, 3)
MEASURED_STARS = 3000


def render_saved(store: Store, manifest: Manifest, name: str, *, only: set[str] | None = None) -> dict[str, Any]:
    """Make a recorded run's sheets, those named in `only` or all, and list
    them in its record. Without a sheet list, or without Pillow, nothing is
    made and the answer says why; a sheet whose input is not built is
    skipped."""
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
        path = directory / "sheets" / f"{sheet['name']}.png"
        whole = directory / "full" / sheet["name"] if sheet.get("full") else None
        drawn = render(store, manifest, record, model, sheet, path, whole)
        if drawn is None:
            missing.append(sheet["name"])
            continue
        made[sheet["name"]] = str(path.relative_to(store.root))
        if drawn:
            full[sheet["name"]] = {key: str(value.relative_to(store.root)) if isinstance(value, Path) else value
                                   for key, value in drawn.items()}
    record = json.loads((directory / "run.json").read_text())
    record["sheets"] = {**record.get("sheets", {}), **made}
    record["full"] = {**record.get("full", {}), **full}
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {"made": made, "full": {key: {k: v for k, v in value.items() if k != "measures"}
                                   for key, value in full.items()}, "missing": missing}


def built(manifest: Manifest, product_id: str | None) -> dict[str, Any] | None:
    if product_id is None:
        return None
    rows = manifest.rows("SELECT p.*, s.split, s.telescope, s.pixel_scale_arcsec FROM products p "
                         "JOIN sets s ON s.id = p.set_id WHERE p.id = ? AND p.status = 'built'", (product_id,))
    return dict(rows[0]) if rows else None


def input_field(store: Store, product: dict[str, Any]) -> psf_fields.Field:
    """The input's PSF field, as the PSF pass measures it from the image
    alone; measured and kept first where the pass has no record of it."""
    if psf_fields.recorded(store, product["id"]) is None:
        psf_fields.measure(store, product)
    return psf_fields.Field(store, product["id"])


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
        self.points = field.points()
        self.widths = node_widths(field)
        self.where = device()

    def ratio(self, strength: float) -> np.ndarray:
        return ratio_map(self.widths, self.field, self.image.shape, strength, self.settings.floor_px)

    def run(self, ratio: np.ndarray, top: int = 0, left: int = 0, bottom: int | None = None,
            right: int | None = None) -> np.ndarray:
        """The region's output, with the whole image's scales, stamps and
        points."""
        bottom = self.image.shape[0] if bottom is None else bottom
        right = self.image.shape[1] if right is None else right
        points = self.points.copy()
        points[:, 0] -= left
        points[:, 1] -= top
        return sharpen(self.model, np.ascontiguousarray(self.image[top:bottom, left:right]),
                       lambda x, y: fitted_stamp(self.field.stamp_at(x + left, y + top))[0],
                       ratio=np.ascontiguousarray(ratio[top:bottom, left:right]), points=points,
                       noise_map=np.ascontiguousarray(self.noise_map[top:bottom, left:right]),
                       background=self.background, noise=self.noise, inference=self.record["inference"],
                       residual=self.settings.residual, where=self.where)


def render(store: Store, manifest: Manifest, record: dict[str, Any], model: Any, sheet: dict[str, Any], path: Path,
           whole: Path | None = None) -> dict[str, Any] | None:
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
    runner = Sharpener(record, model, image, field)
    look = stretch(image)
    image_background = float(np.nanmedian(image[::31, ::31]))
    baselines = []
    for entry in sheet.get("baselines", ()):
        product = built(manifest, entry["product"])
        if product is None:
            continue
        other = psf_fields.load_image(store, product)
        if other.shape != image.shape:
            raise ValueError(f"baseline {entry['product']} is {other.shape}, its sheet's input {image.shape}")
        other[np.isnan(image)] = np.nan
        baselines.append((entry, other, float(np.nanmedian(other[::31, ::31]))))

    side = sheet.get("side", SIDE)
    height, width = image.shape
    ratios = {strength: runner.ratio(strength) for _, strength in STRENGTHS}
    inference = record["inference"]
    pad = inference["margin"] + inference["tile"] // 2
    labels = ["Input", *(label for label, _ in STRENGTHS), *(entry["label"] for entry, _, _ in baselines)]
    rows = []
    for crop in sheet["crops"]:
        x, y = min(crop["x"], width - side), min(crop["y"], height - side)
        top, left = max(0, y - pad), max(0, x - pad)
        bottom, right = min(height, y + side + pad), min(width, x + side + pad)
        inner = (slice(y - top, y - top + side), slice(x - left, x - left + side))
        panels = [image[y:y + side, x:x + side]]
        panels += [runner.run(ratios[strength], top, left, bottom, right)[inner] for _, strength in STRENGTHS]
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
             f"{np.min(widths):.1f} to {np.max(widths):.1f} px, floor {runner.settings.floor_px:g} px")
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
    written = full_size(runner, ratios.get(FULL_STRENGTH, runner.ratio(FULL_STRENGTH)),
                        lambda x: look(x, image_background), whole)
    valid = np.isfinite(image) & ~saturation(image, field.record.get("clip_level"))
    measures = {"run": whole_measures(image, written.pop("output"), runner, valid)}
    for entry, other, background_of in baselines:
        shared = store.root / "sheets" / "full-sharpen" / f"{sheet['name']}-{entry['name']}.png"
        shared.parent.mkdir(parents=True, exist_ok=True)
        write_png(shared, (look(other, background_of) * 255 + 0.5).astype(np.uint8))
        written[f"baseline-{entry['name']}"] = shared
        measures[entry["name"]] = whole_measures(image, other, runner, valid)
    written["strength"] = FULL_STRENGTH
    written["measures"] = measures
    return written


def full_size(runner: Sharpener, ratio: np.ndarray, look: Any, stem: Path) -> dict[str, Any]:
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
    ratios = [cell["ratio"] for cell in cells if cell["ratio"] is not None]
    found["width_ratio"] = spread(ratios)
    typical = float(np.nanmedian(runner.widths))
    stars = evenly(measured_stars(runner.points, inner, valid, int(math.ceil(3 * typical)) + 4), MEASURED_STARS)
    columns = psf_fields.DETECTION_COLUMNS
    kept, rings, moved_light, moved_peak = [], [], [], []
    radius = LIGHT_RADIUS * typical
    for star in stars:
        x, y = float(star[0]), float(star[1])
        measured = ring_and_flux(output, image, x, y, typical, runner.noise)
        if measured["reference_flux"] > 0:
            kept.append(measured["flux"] / measured["reference_flux"])
        if (star[columns.index("peak")] - runner.background) / runner.noise >= RING_PEAK:
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
    return found
