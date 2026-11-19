"""Write the harness's answers for a sharpening run on a small synthetic star
field, so the crate's tests hold its Rust to the Python a model was trained
with.

It needs the sharpening trainer (`seiza_training.sharpen_model`,
`psf_fields`, `sharpen_sheets`, `unet.SharpeningUNet` and the bundle writer's
`sharpen_card` and `write_sharpen_bundle`), which is not in this branch yet:
the fixtures beside it were recorded with it, and are committed so the tests
run without it. Run it from a training environment that has the trainer and
the `seiza` binding, built from the same source as the crate under test:

    PYTHONPATH=training python seiza-model/tests/fixtures/make_sharpen_fixtures.py

`sharpen/image.f32` is the image, row-major little-endian floats.
`sharpen/tiny` is a bundle of random weights, small enough to commit.
`sharpen/expected.json` holds what the harness makes of the image at sampled
pixels: its PSF field's nodes and their widths, its point sources, the maps
the network reads, and the tiny model's output, as it is and rolled off. Its
`scaled` holds the same for `sharpen/scaled.f32`, an image of wider stars,
with a scaled pass (`SCALED`) that the committed card leaves out and a test
adds: the factor it is binned by, the binned image's reading and ratio
maps, a known change put together by scale, and the output, as it is and
rolled off.

With `--image FILE.npy --run RUN --store STORE --out DIRECTORY`, it writes
the same for a real image and a trained run instead, with the run's bundle,
for the crate's ignored comparison (`SEIZA_SHARPEN_PARITY`). With
`--scaled`, the bundle's card has the scaled pass and its own full-size
asks, and `scaled-output.f32` holds the output with the pass.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from seiza_training import psf_fields, sharpen_model
from seiza_training.bundle import sha256_of, sharpen_card, write_sharpen_bundle
from seiza_training.sharpen_sheets import SCALED_FULL, Sharpener, block_mean, brought_up, scale_blended
from seiza_training.unet import SharpeningUNet, reach

HERE = Path(__file__).parent
WIDTH, HEIGHT = 330, 250
STRIDE = 11
# Nodes closer than the PSF pass keeps them, so a small image has several,
# and a field that varies from fewer stars than the pass asks for, as a small
# image has few.
GRID_PX = 96
MINIMUM_STARS = 10
TINY_WIDTH, TINY_LEVELS, TINY_RAMP, TINY_TILE = 4, 3, 8, 128
TINY_STAMP_WIDTH, TINY_FEATURES = 4, 8
STRENGTHS = (0.46, 0.5)
ROLLOFF = 1.0
SETTINGS = {"nonstellar": True, "kernel_ratios": [1.1, 4.0], "floor_px": 1.5, "residual": "asinh"}
# The scaled pass a bundle's card has, which bins `scaled_image`'s stars,
# about 7 px across, 2 × 2. That image is larger than `image`, so that its
# binned half still holds stars enough for one PSF; the floor holds the
# binned run's rest above the 0.35 asked.
SCALED = {"stars_px": 3.5, "blend": 0.5, "rest": 0.35}
SCALED_WIDTH, SCALED_HEIGHT = 400, 300


def sampled(plane: np.ndarray, stride: int) -> list:
    """Every `stride`th pixel of a plane, NaN as null."""
    values = plane[::stride, ::stride].astype(np.float64).ravel()
    return [None if np.isnan(value) else float(value) for value in values]


def image() -> np.ndarray:
    """Stars that widen from left to right, three of them clipped, on a sky
    with a nebula at the top and a corner nothing covers."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    signal = 200.0 + 0.05 * x + 60.0 * np.exp(-((y - 40) ** 2) / (2 * 30.0**2))
    places = rng.uniform((8, 8), (HEIGHT - 8, WIDTH - 8), (80, 2))
    fluxes = 10 ** rng.uniform(3.6, 4.6, len(places))
    for (cy, cx), flux in zip(places, fluxes):
        sigma = 1.1 + 0.9 * cx / WIDTH
        signal += flux / (2 * np.pi * sigma**2) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    for cy, cx in ((60, 70), (150, 200), (210, 90)):
        signal += 4e5 / (2 * np.pi * 2.0) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 2.0))
    plane = signal + rng.normal(0.0, 1.0, (HEIGHT, WIDTH)) * np.sqrt(9.0 + 0.02 * (signal - 200.0))
    plane = np.minimum(plane, 6000.0).astype(np.float32)
    plane[-10:, :14] = np.nan
    return plane


def scaled_image() -> np.ndarray:
    """Stars about 6 to 8.5 px across, widening from left to right, on a
    sky with a nebula at the top and a corner nothing covers. None is
    clipped (`image` has those): a clipped core this wide widens the binned
    image's one PSF, which its few stars leave global."""
    rng = np.random.default_rng(13)
    y, x = np.mgrid[:SCALED_HEIGHT, :SCALED_WIDTH]
    signal = 200.0 + 0.05 * x + 60.0 * np.exp(-((y - 60) ** 2) / (2 * 45.0**2))
    places = rng.uniform((12, 12), (SCALED_HEIGHT - 12, SCALED_WIDTH - 12), (40, 2))
    fluxes = 10 ** rng.uniform(4.2, 5.2, len(places))
    for (cy, cx), flux in zip(places, fluxes):
        sigma = 2.6 + 1.0 * cx / SCALED_WIDTH
        signal += flux / (2 * np.pi * sigma**2) * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    plane = signal + rng.normal(0.0, 1.0, signal.shape) * np.sqrt(9.0 + 0.02 * (signal - 200.0))
    plane = np.minimum(plane, 6000.0).astype(np.float32)
    plane[-12:, :16] = np.nan
    return plane


class ImageField:
    """An image's PSF field and point sources, measured from it as the PSF
    pass measures a product's, and held as its record holds them."""

    def __init__(self, plane: np.ndarray) -> None:
        table, self.clip_level, _ = psf_fields.detections(plane)
        field, self.status, self.error = psf_fields.measure_field(plane)
        if field is None:
            raise ValueError(f"the image's PSF could not be measured: {self.error}")
        height, width = plane.shape
        xs, ys = psf_fields.grid_nodes(width), psf_fields.grid_nodes(height)
        stamps = np.stack([np.stack([field.stamp_at(float(x), float(y)) for x in xs]) for y in ys]).astype(np.float32)
        self.arrays = {"detections": table, "xs": xs.astype(np.float32), "ys": ys.astype(np.float32),
                       "stamps": stamps}
        self.stars, self.is_global = field.star_count, field.is_global

    def points(self) -> np.ndarray:
        return self.arrays["detections"]


def edged(plane: np.ndarray) -> np.ndarray:
    """A change the crate's tests can make again without a network: each
    pixel moved from its left neighbour by half as much again."""
    found = plane.copy()
    found[:, 1:] += np.float32(0.5) * (plane[:, 1:] - plane[:, :-1])
    return found


def answers(record: dict, model: torch.nn.Module, plane: np.ndarray, stride: int,
            strengths: tuple[float, float] = STRENGTHS) -> dict:
    """What the harness makes of `plane` with `model`."""
    field = ImageField(plane)
    runner = Sharpener(record, model, plane, field)
    runner.where = torch.device("cpu")
    ratios = runner.ratio(strengths)
    rendered = sharpen_model.field_point_map(runner.points, runner.ys, runner.xs, runner.node_stamps, plane.shape,
                                             runner.background)
    features = sharpen_model.node_features(model, runner.node_stamps, runner.where)
    tile, multiple = record["inference"]["tile"], record["inference"]["multiple"]
    corner = (multiple * 3, multiple * 5)
    output = runner.run(ratios)
    runner.rolloff = ROLLOFF
    rolled = runner.run(ratios)
    columns = psf_fields.DETECTION_COLUMNS
    return {
        "width": plane.shape[1],
        "height": plane.shape[0],
        "stride": stride,
        "strengths": list(strengths),
        "rolloff": ROLLOFF,
        "background": runner.background,
        "noise": runner.noise,
        "status": field.status,
        "stars": field.stars,
        "ys": runner.ys.tolist(),
        "xs": runner.xs.tolist(),
        "widths": [None if np.isnan(width) else float(width) for width in runner.widths.ravel()],
        "stamp_side": int(runner.node_stamps.shape[-1]),
        "stamp_sums": [float(stamp.sum(dtype=np.float64)) for stamp in runner.node_stamps.reshape(
            -1, *runner.node_stamps.shape[2:])],
        "stamp_peaks": [float(stamp.max()) for stamp in runner.node_stamps.reshape(-1, *runner.node_stamps.shape[2:])],
        "points": [[None if np.isnan(row[columns.index(name)]) else float(row[columns.index(name)])
                    for name in ("x", "y", "flux", "peak")] for row in runner.points],
        "ratio_stars": sampled(ratios[0], stride),
        "ratio_rest": sampled(ratios[1], stride),
        "point_map": sampled(rendered, stride),
        "vectors": features.reshape(-1, features.shape[-1]).astype(np.float64).tolist(),
        "corner": list(corner),
        "vector_map": sharpen_model.sampled_features(features, runner.ys, runner.xs, corner[0], corner[1], tile,
                                                     multiple).astype(np.float64).ravel().tolist(),
        "edged_rolled": sampled(sharpen_model.rolloff_step(edged(plane), plane, ROLLOFF, runner.noise_map), stride),
        "output": sampled(output, stride),
        "rolled": sampled(rolled, stride),
    }


def scaled_answers(record: dict, model: torch.nn.Module, plane: np.ndarray, stride: int, scaled: dict,
                   strengths: tuple[float, float], *, known: bool = True, rolled: bool = True) -> dict:
    """What the harness makes of `plane` with `model` and the scaled pass
    `scaled`, a card's section: the binned image's reading and the binned
    run's ratio maps, sampled every `stride`th binned pixel; with `known`,
    the `edged` change put together by scale; and the output, as it is and,
    with `rolled`, rolled off."""
    runner = Sharpener(record, model, plane, ImageField(plane))
    runner.where = torch.device("cpu")
    runner.scale_to, runner.scale_blend, runner.scale_rest = scaled["stars_px"], scaled["blend"], scaled["rest"]
    factor = runner.factor
    smaller = runner.smaller()
    smaller.where = runner.where
    ratios = runner.ratio(strengths)
    binned = runner.smaller_ratio(ratios, smaller)
    sigma = runner.scale_blend * float(np.nanmedian(runner.widths))
    found = {
        "scaled": scaled,
        "width": plane.shape[1],
        "height": plane.shape[0],
        "stride": stride,
        "strengths": list(strengths),
        "rolloff": ROLLOFF,
        "background": runner.background,
        "noise": runner.noise,
        "typical": float(np.nanmedian(runner.widths)),
        "factor": factor,
        "sigma": sigma,
        "binned": {
            "width": smaller.image.shape[1],
            "height": smaller.image.shape[0],
            "background": smaller.background,
            "noise": smaller.noise,
            "stars": smaller.field.record["star_count"],
            "ys": smaller.ys.tolist(),
            "xs": smaller.xs.tolist(),
            "widths": [None if np.isnan(width) else float(width) for width in smaller.widths.ravel()],
            "points": len(smaller.points),
            "image": sampled(smaller.image, stride),
            "ratio_stars": sampled(binned[0], stride),
            "ratio_rest": sampled(binned[1], stride),
        },
    }
    if known:
        change = edged(plane)
        up = brought_up(np.nan_to_num(block_mean(change, factor) - smaller.image), factor)
        up = up[:plane.shape[0], :plane.shape[1]]
        found["edged_up"] = sampled(up, stride)
        found["edged_blended"] = sampled(scale_blended(plane, change, up, sigma), stride)
    found["output"] = sampled(runner.run(ratios), stride)
    if rolled:
        runner.rolloff = ROLLOFF
        found["rolled"] = sampled(runner.run(ratios), stride)
    return found


def tiny(plane: np.ndarray) -> None:
    """A tiny bundle of random weights and the harness's answers with it."""
    torch.manual_seed(11)
    model = SharpeningUNet(inputs=len(sharpen_model.PLANES_NONSTELLAR), width=TINY_WIDTH, levels=TINY_LEVELS,
                           stamp_inputs=len(sharpen_model.STAMP_PLANES), stamp_width=TINY_STAMP_WIDTH,
                           features=TINY_FEATURES).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    # A residual of a few tenths, as a trained model's is on the sky.
    with torch.no_grad():
        model.last.weight.mul_(0.03)
        model.last.bias.mul_(0.03)
    margin = reach(TINY_LEVELS)
    tiling = {"tile": TINY_TILE, "overlap": 2 * margin + TINY_RAMP, "margin": margin, "multiple": model.multiple}
    record = {"config": {"settings": SETTINGS, "planes": list(sharpen_model.PLANES_NONSTELLAR)}, "inference": tiling}
    directory = HERE / "sharpen" / "tiny"
    written = sharpen_model.export_parts(model, directory)
    graphs = [{"precision": entry["precision"], "file": entry["file"], "sha256": sha256_of(directory / entry["file"])}
              for entry in written["body"]]
    encoder = {"file": written["encoder"]["file"], "sha256": sha256_of(directory / written["encoder"]["file"]),
               "features": written["features"]}
    card = sharpen_card("sharpen-tiny", record, graphs, encoder, ["cpu"], STRENGTHS, scaled=False)
    (directory / "card.json").write_text(json.dumps(card, indent=2) + "\n")
    expected = answers(record, model, plane, STRIDE)
    wide = scaled_image()
    wide.astype("<f4").tofile(HERE / "sharpen" / "scaled.f32")
    expected["scaled"] = scaled_answers(record, model, wide, STRIDE, SCALED, STRENGTHS)
    (HERE / "sharpen" / "expected.json").write_text(json.dumps(expected, indent=0) + "\n")


def trained(arguments: argparse.Namespace) -> None:
    """A trained run's bundle and the harness's answers with it on a real
    image, for the crate's ignored comparison."""
    store = SimpleNamespace(root=arguments.store)
    plane = np.load(arguments.image).astype(np.float32)
    arguments.out.mkdir(parents=True, exist_ok=True)
    bundle = arguments.out / "bundle"
    strengths = SCALED_FULL if arguments.scaled else STRENGTHS
    if not (bundle / "card.json").exists():
        write_sharpen_bundle(store, arguments.run, bundle, name=arguments.out.name, strengths=strengths,
                             scaled=arguments.scaled)
    record, model = sharpen_model.load_run(store, arguments.run)
    plane.astype("<f4").tofile(arguments.out / "image.f32")

    def kept(values: list, name: str) -> None:
        np.array([np.nan if value is None else value for value in values], dtype="<f4").tofile(
            arguments.out / f"{name}.f32")

    expected = answers(record, model, plane, 1, strengths)
    for name in ("output", "rolled", "edged_rolled", "point_map", "ratio_stars", "ratio_rest"):
        kept(expected.pop(name), name)
    if arguments.scaled:
        card = json.loads((bundle / "card.json").read_text())
        found = scaled_answers(record, model, plane, 1, card["sharpen"]["scaled"], strengths, known=False,
                               rolled=False)
        kept(found.pop("output"), "scaled-output")
        for name in ("image", "ratio_stars", "ratio_rest"):
            found["binned"].pop(name)
        expected["scaled"] = found
    (arguments.out / "expected.json").write_text(json.dumps(expected, indent=0) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scaled", action="store_true",
                        help="give the bundle's card the scaled pass, and record the output with it")
    arguments = parser.parse_args()
    if arguments.image:
        trained(arguments)
        return
    psf_fields.GRID_PX = GRID_PX
    psf_fields.MEASURE_OPTIONS = {**psf_fields.MEASURE_OPTIONS, "minimum_stars": MINIMUM_STARS}
    plane = image()
    (HERE / "sharpen").mkdir(exist_ok=True)
    plane.astype("<f4").tofile(HERE / "sharpen" / "image.f32")
    tiny(plane)
    card_path = HERE / "sharpen" / "tiny" / "card.json"
    card = json.loads(card_path.read_text())
    assert card["sharpen"]["psf"]["grid_px"] == GRID_PX
    assert card["sharpen"]["psf"]["minimum_stars"] == MINIMUM_STARS
    print(json.dumps({key: value for key, value in json.loads((HERE / "sharpen" / "expected.json").read_text()).items()
                      if key in ("background", "noise", "status", "stars", "ys", "xs", "widths")}))


if __name__ == "__main__":
    main()
