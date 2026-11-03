"""Contact sheets: what a noise run does to a few chosen images, side by
side at 1:1, so runs are judged by eye as well as by their measures.

The images are listed in `STORE/sheets/noise.json`:

    {"sheets": [{"name": "elephants-trunk-ha-n04",
                 "title": "Elephant's Trunk Nebula, Hα, 2026-10-15",
                 "input": "<product id>", "reference": "<product id>",
                 "crops": [{"x": 1300, "y": 1550, "label": "The trunk's head"}]}]}

The input is a stack with its variance and coverage: a noise half, a deep
stack or a deep stack's half. Each crop is a row: the input; the run with
all the noise it finds taken out; with each of `SETTINGS`' shares kept;
and the reference, a deeper stack on the same grid, or, when nothing
deeper is given, the noise taken out divided by each pixel's own noise, in
which any shape is something other than noise the run changed. Every other
panel takes the reference's stretch, or the input's when there is none.

Sheets go to `STORE/models/noise/<run>/sheets/<name>.png` and are listed in
the run's record. `train-noise` makes them after every run; `sheet-noise`
makes them for a recorded one. Pillow draws the labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Manifest, loads
from .noise_model import NoiseBands, denoiser, device, load_half, load_run, modelled_sigma, network_input, scales
from .store import Store, write_png
from .tiles import infer
from .unet import ResidualUNet

SPEC = Path("sheets") / "noise.json"
# The side of a crop, in pixels, unless a sheet gives its own.
SIDE = 800
# Without a reference, the noise taken out is shown in units of each
# pixel's own noise, from −SPAN to +SPAN: where a run takes out only noise
# it is even grain at every brightness, and anything else shows as shape.
SPAN = 3.0
# The shares of the noise kept, besides none, on every sheet: the same for
# every run, so sheets compare.
SETTINGS = (
    ("15% of the noise kept", NoiseBands(high=0.85, low=0.85)),
    ("25% kept", NoiseBands(high=0.75, low=0.75)),
    ("35% kept", NoiseBands(high=0.65, low=0.65)),
    ("30% of the grain kept, 10% of the mottle", NoiseBands(high=0.70, low=0.90)),
)


def render_saved(store: Store, manifest: Manifest, name: str, *, only: set[str] | None = None) -> dict[str, Any]:
    """Make a recorded run's sheets, those named in `only` or all, and list
    them in its record. Without a sheet list, or without Pillow, nothing is
    made and the answer says why; a sheet whose products are not built is
    skipped."""
    spec = store.root / SPEC
    if not spec.exists():
        return {"skipped": f"no sheet list at {spec}"}
    try:
        import PIL  # noqa: F401
    except ImportError:
        return {"skipped": "Pillow, which draws the labels, is not installed"}
    record, model = load_run(store, name)
    directory = store.root / "models" / "noise" / name
    made, missing = {}, []
    for sheet in json.loads(spec.read_text())["sheets"]:
        if only is not None and sheet["name"] not in only:
            continue
        path = directory / "sheets" / f"{sheet['name']}.png"
        if render(store, manifest, record, model, sheet, path):
            made[sheet["name"]] = str(path.relative_to(store.root))
        else:
            missing.append(sheet["name"])
    record = json.loads((directory / "run.json").read_text())
    record["sheets"] = {**record.get("sheets", {}), **made}
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return {"made": made, "missing": missing}


def render(
    store: Store, manifest: Manifest, record: dict[str, Any], model: ResidualUNet, sheet: dict[str, Any], path: Path
) -> bool:
    """Draw one sheet to `path`; false when its products are not built."""
    from PIL import Image, ImageDraw, ImageFont

    def built(product_id: str | None) -> dict[str, Any] | None:
        if product_id is None:
            return None
        rows = manifest.rows("SELECT * FROM products WHERE id = ? AND status = 'built'", (product_id,))
        return dict(rows[0]) if rows else None

    source, reference_product = built(sheet["input"]), built(sheet.get("reference"))
    if source is None or (sheet.get("reference") and reference_product is None):
        return False
    image, sigma = load_half(store, source)
    if record["config"].get("noise_map", "raw") == "model":
        sigma = modelled_sigma(image, sigma, np.load(store.root / loads(source["files"])["coverage"]))
    background, noise = scales(image, sigma)
    planes = network_input(image, sigma, background, noise)
    run = denoiser(model, background, noise, device())
    reference = load_half(store, reference_product, noise_map=False)[0] if reference_product else None
    look = stretch(reference if reference is not None else image)
    reference_background = float(np.nanmedian(reference[::31, ::31])) if reference is not None else 0.0
    image_background = float(np.nanmedian(image[::31, ::31]))

    side = sheet.get("side", SIDE)
    height, width = image.shape
    # Crops run through the network with its receptive field around them,
    # which gives the whole image's result inside them.
    pad = record["inference"]["margin"] + 16
    frames = len(loads(source["frames"]))
    labels = [f"Input: {frames} frames", "All noise removed", *(label for label, _ in SETTINGS)]
    labels.append(f"Reference: {len(loads(reference_product['frames']))} frames" if reference is not None
                  else f"Noise taken out ÷ each pixel's noise, ±{SPAN:g}σ")
    rows = []
    for crop in sheet["crops"]:
        x, y = min(crop["x"], width - side), min(crop["y"], height - side)
        top, left = max(0, y - pad), max(0, x - pad)
        bottom, right = min(height, y + side + pad), min(width, x + side + pad)
        window = image[top:bottom, left:right]
        residual = infer(run, np.ascontiguousarray(planes[:, top:bottom, left:right]), **record["inference"])[0]
        inner = (slice(y - top, y - top + side), slice(x - left, x - left + side))
        panels = [window[inner], NoiseBands().apply(window, residual)[inner],
                  *(bands.apply(window, residual)[inner] for _, bands in SETTINGS)]
        if reference is not None:
            last = look(reference[y:y + side, x:x + side], reference_background)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                taken = -residual[inner] / sigma[y:y + side, x:x + side]
            last = np.clip(np.nan_to_num(0.5 + taken / (2 * SPAN), nan=0.5), 0.0, 1.0)
        rows.append((f"{crop['label']}: 1:1 crop at x {x}, y {y}",
                     [look(panel, image_background) for panel in panels] + [last]))

    gap, label_height, title_height, row_label = 12, 44, 64, 34
    columns = len(labels)
    canvas = Image.new("L", (columns * side + (columns + 1) * gap,
                             title_height + len(rows) * (row_label + label_height + side + gap) + gap), 18)
    draw = ImageDraw.Draw(canvas)
    font, small, big = (typeface(ImageFont, size) for size in (26, 22, 32))
    title = f"{sheet['title']}: {record['name']} ({record['parameters'] / 1e6:.3g}M parameters)"
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
    return True


# Faces with the labels' Greek letters and signs, macOS's then Linux's;
# Pillow's own face lacks them.
TYPEFACES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def typeface(fonts: Any, size: int) -> Any:
    """The first of `TYPEFACES` found, at `size`, or Pillow's own."""
    for path in TYPEFACES:
        if Path(path).exists():
            return fonts.truetype(path, size)
    return fonts.load_default(size)


def stretch(source: np.ndarray):
    """The display stretch `source` sets: black a little below its
    background, white at its 99.95th percentile, and the midtones set so
    the background shows at a fifth of white. The function it returns takes
    an image and that image's own background, which it moves onto the
    source's."""
    finite = source[::7, ::7]
    finite = finite[np.isfinite(finite)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    white = float(np.percentile(finite, 99.95))
    shadows = median - 2.8 * mad
    m0 = (median - shadows) / (white - shadows)
    target = 0.2
    # The midtones balance that sends m0 to the target.
    m = m0 * (target - 1) / (2 * m0 * target - m0 - target)

    def apply(x: np.ndarray, background: float) -> np.ndarray:
        v = (x - (background - median) - shadows) / (white - shadows)
        v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)
        return ((m - 1) * v) / ((2 * m - 1) * v - m)

    return apply
