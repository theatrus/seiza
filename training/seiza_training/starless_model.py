"""The starless model's trainer (`docs/design/ml-star-removal.md`): a
residual U-Net (`unet.ResidualUNet`) trained on the composites of a starless
bank (`starless_bank`) to give an image's starless image.

Two ways, compared on one bank:

- `fit`: the network reads the image, the catalogue fit's starless estimate
  and its stars' model, the noise and the stars' places, and corrects the
  estimate: it heals what the fit leaves (remnants, clipped cores, spikes,
  halos) and fills where stars were.
- `image`: the network reads the image, the noise and the stars' places
  alone, and corrects the image to its starless one.

Either starts as its baseline, since the U-Net's last layer starts at zero:
the fit's estimate, or the image itself. A tile is read on the scale of its
own noise, its sky level taken off (`level`, the median of its valid
pixels; `scale`, its noise map's median), and every plane but the stars'
places is compressed by `asinh(x / ASINH)`, so a bright star's core and the
sky's noise both count. The network gives a residual in that space; the
starless image is the baseline's compressed value plus it, taken back
through `sinh`.

The loss is a Charbonnier penalty on the compressed starless image against
its truth, over the window's valid pixels, weighted `STAR_WEIGHT` times
where the image and its truth differ by more than the noise (where stars
were), where the healing is learned. Validation is the same loss over fixed
batches of a validation bank, with the baseline's alone beside it.

A run with `layers` gives two planes more: the stars' halos and rings, and
their diffraction spikes, each the light the bank keeps apart for it, read
about zero through the same compression and penalised alike beside the
starless image (`LAYER_WEIGHT`). The input is then four layers that add back
to it (`layers`): the starless image, the halos, the spikes, and the stars'
cores and wings, which is what is left and carries their photon noise. Of
the halos, a chosen share goes back into the starless image and the rest
with the stars (`kept`), so halo reduction is a setting.

The spikes are sparse: a pixel near a bright star holds one on some
telescopes and not on most. A penalty that seeks the median leaves such a
pixel at zero until the network can tell where spikes run, which it learns
only by first leaning toward them, so the spike layer can be penalised as
squared error up to an error of its own (`Settings.spike_quadratic`, as the
Charbonnier penalty's epsilon) and weighed up (`Settings.spike_weight`).
Few windows show spikes over the noise, and the layer learns only once it
has seen some tens of thousands of them, so a share of every batch can be
drawn from those that do (`train`'s `spike_share`, `spiky_windows`).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .store import Store
from .unet import ResidualUNet

# The planes each way reads, in order, of a starless bank's (`noise` is read
# as the ratio of the window's noise map to its median, `points` as it is).
READS = {"fit": ("input", "fit_starless", "fit_model", "noise", "points"),
         "image": ("input", "noise", "points")}
BASELINE = {"fit": "fit_starless", "image": "input"}
# Planes that hold no sky, read about zero rather than the sky's level.
SKYLESS = ("fit_model",)
# The compression: planes in units of the noise, through asinh(x / ASINH).
ASINH = 5.0
# Where the image and its truth differ by over the noise, the loss weighs
# this many times as much.
STAR_WEIGHT = 10.0
CHARBONNIER = 0.01
VALIDATION_STREAM = 1 << 20
# The planes of the stars' light a run with `layers` gives beside the
# starless image, as a starless bank keeps them, and how much each one's
# penalty weighs against the starless image's.
LAYERS = ("halos", "spikes")
LAYER_WEIGHT = 1.0
# A window shows spikes when its spike truth tops the noise over this many
# pixels.
SPIKY_PIXELS = 500


@dataclass(frozen=True)
class Settings:
    """What a run reads and how: `way` (`READS`), the tile's side, the
    network's width and levels, whether it gives the halos and spikes as
    layers of their own (`LAYERS`), and with them the spike layer's penalty:
    quadratic up to an error of `spike_quadratic` (compressed) and
    `spike_weight` times as heavy as the starless image's. The defaults are
    the penalty every other part has."""

    way: str = "fit"
    tile: int = 256
    width: int = 32
    levels: int = 5
    layers: bool = False
    spike_quadratic: float = CHARBONNIER
    spike_weight: float = LAYER_WEIGHT

    @property
    def outputs(self) -> int:
        """The planes the network gives: the starless image's residual, and
        each of `LAYERS` with `layers`."""
        return 1 + (len(LAYERS) if self.layers else 0)

    def __post_init__(self) -> None:
        if self.way not in READS:
            raise ValueError(f"a starless run reads one of {sorted(READS)}, not {self.way}")
        if self.tile % 2 ** (self.levels - 1):
            raise ValueError(f"a tile of {self.tile} is not a multiple of {2 ** (self.levels - 1)}")
        if self.spike_quadratic <= 0.0 or self.spike_weight < 0.0:
            raise ValueError(f"a spike penalty quadratic to {self.spike_quadratic} and weighed "
                             f"{self.spike_weight} times is not one")


class Bank:
    """A starless bank's planes, memory-mapped, and its windows' records."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.done = json.loads((self.directory / "done.json").read_text())
        self.count = int(self.done["windows"])
        self.planes = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r") for name in self.done["planes"]}
        self.records = json.loads((self.directory / "sources.json").read_text())


def compressed(plane: np.ndarray, level: float, scale: float) -> np.ndarray:
    return np.arcsinh((plane - level) / (scale * ASINH)).astype(np.float32)


def released(plane: np.ndarray, level: float, scale: float) -> np.ndarray:
    """`compressed`'s inverse."""
    return (np.sinh(plane) * scale * ASINH + level).astype(np.float32)


def network_reads(planes: dict[str, np.ndarray], way: str) -> tuple[list[np.ndarray], float, float]:
    """The planes a way's network reads, and the level and scale they are
    read on: the input's median, as a runtime knows it with no mask, and
    the noise map's median. A plane with sky is read less the level, a
    plane without (`SKYLESS`) as it is, both compressed."""
    level = float(np.nanmedian(planes["input"]))
    scale = float(np.median(planes["noise"])) or 1.0
    read = []
    for name in READS[way]:
        if name == "noise":
            read.append(planes["noise"] / scale)
        elif name == "points":
            read.append(planes["points"])
        else:
            read.append(compressed(planes[name], 0.0 if name in SKYLESS else level, scale))
    return read, level, scale


def example(bank: Bank, index: int, settings: Settings, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A tile of window `index`: the network's planes, the baseline and the
    truth compressed, the loss's weights, and with `layers` each of `LAYERS`
    compressed about zero; flipped and turned at random."""
    side = bank.planes["input"].shape[1]
    tile = settings.tile
    top, left = (int(rng.integers(0, side - tile + 1)) for _ in range(2))
    cut = (index, slice(top, top + tile), slice(left, left + tile))
    planes = {name: np.asarray(bank.planes[name][cut], dtype=np.float32) for name in bank.planes}
    valid = planes["valid"] > 0
    read, level, scale = network_reads(planes, settings.way)
    baseline = compressed(planes[BASELINE[settings.way]], level, scale)
    truth = compressed(planes["starless"], level, scale)
    weight = valid.astype(np.float32) * (1.0 + (STAR_WEIGHT - 1.0)
                                         * (np.abs(planes["input"] - planes["starless"]) > planes["noise"]))
    extras = {"baseline": baseline, "truth": truth, "weight": weight}
    if settings.layers:
        extras.update({name: compressed(planes[name], 0.0, scale) for name in LAYERS})
    stacked = np.stack(read + list(extras.values()))
    turns = int(rng.integers(0, 4))
    stacked = np.rot90(stacked, turns, axes=(1, 2))
    if rng.random() < 0.5:
        stacked = stacked[:, :, ::-1]
    stacked = np.ascontiguousarray(stacked, dtype=np.float32)
    found = {"planes": stacked[:len(read)]}
    found.update({key: stacked[len(read) + offset] for offset, key in enumerate(extras)})
    return found


def batch_from(bank: Bank, indices: np.ndarray, settings: Settings, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    examples = [example(bank, int(index), settings, rng) for index in indices]
    return {key: torch.from_numpy(np.stack([found[key] for found in examples])) for key in examples[0]}


def spiky_windows(bank: Bank, pixels: int = SPIKY_PIXELS) -> np.ndarray:
    """The windows of `bank` whose spike truth tops the noise over at least
    `pixels` pixels, in order."""
    return np.array([index for index in range(bank.count)
                     if int((np.asarray(bank.planes["spikes"][index]) > np.asarray(bank.planes["noise"][index])).sum())
                     >= pixels], dtype=np.int64)


def drawn(count: int, spiky: np.ndarray, share: float, batch: int, rng: np.random.Generator) -> np.ndarray:
    """A batch's windows: `share` of them (rounded) from `spiky`, the rest
    from all `count`; with no share, as every run before drew them."""
    some = round(share * batch) if len(spiky) else 0
    if not some:
        return rng.integers(0, count, batch)
    return np.concatenate([rng.integers(0, count, batch - some), rng.choice(spiky, some)])


def loss_of(model: torch.nn.Module, batch: dict[str, torch.Tensor], settings: Settings | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """The run's loss on `batch`, its baseline's (the residual 0, and no
    halos or spikes), and their parts, detached: the starless image's
    penalty and, with layers, each of `LAYERS`' (before its weight), each
    beside its baseline's (`<part>_baseline`). The spike layer's penalty
    and weight are `settings`' (the others' without)."""
    settings = settings or Settings()
    out = model(batch["planes"])
    weight = batch["weight"]
    total = weight.sum().clamp_min(1.0)

    def penalty(estimate: torch.Tensor, truth: torch.Tensor, quadratic: float = CHARBONNIER) -> torch.Tensor:
        # Charbonnier, quadratic up to `quadratic`; less its own floor and
        # plus the usual one, so every part reads from the same floor.
        return (weight * (torch.sqrt((estimate - truth) ** 2 + quadratic**2) - quadratic + CHARBONNIER)).sum() / total

    run = penalty(batch["baseline"] + out[:, 0], batch["truth"])
    base = penalty(batch["baseline"], batch["truth"]).detach()
    parts = {"starless": run.detach(), "starless_baseline": base}
    for channel, name in enumerate(LAYERS[:out.shape[1] - 1], start=1):
        quadratic, heavy = ((settings.spike_quadratic, settings.spike_weight) if name == "spikes"
                            else (CHARBONNIER, LAYER_WEIGHT))
        part = penalty(out[:, channel], batch[name], quadratic)
        part_base = penalty(torch.zeros_like(batch[name]), batch[name], quadratic).detach()
        parts[name], parts[f"{name}_baseline"] = part.detach(), part_base
        run = run + heavy * part
        base = base + heavy * part_base
    return run, base, parts


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_directory(store: Store, name: str) -> Path:
    return store.root / "models" / "starless" / name


def train(store: Store, *, name: str, bank: Path, validation_bank: Path | None, settings: Settings,
          steps: int = 10_000, batch: int = 8, learning_rate: float = 3e-4, seed: int = 0, log_every: int = 250,
          checkpoint_every: int = 1000, validation_batches: int = 8, where: torch.device | None = None,
          init_from: str | None = None, spike_share: float = 0.0) -> dict[str, Any]:
    """Train a starless run on `bank` and keep it under
    `STORE/models/starless/<name>/`: its weights, its configuration and
    its log of the loss, the validation loss and the baseline's. With
    `spike_share`, that share of every batch is drawn from the windows that
    show spikes (`spiky_windows`)."""
    if not 0.0 <= spike_share <= 1.0 or (spike_share and not settings.layers):
        raise ValueError(f"a share of {spike_share} of windows with spikes is not one for this run")
    where = where or device()
    torch.manual_seed(seed)
    train_bank = Bank(bank)
    spiky = spiky_windows(train_bank) if spike_share else np.zeros(0, dtype=np.int64)
    if spike_share and not len(spiky):
        raise ValueError(f"{bank} has no windows with spikes to draw a share from")
    held = Bank(validation_bank) if validation_bank else None
    model = ResidualUNet(inputs=len(READS[settings.way]), outputs=settings.outputs, width=settings.width,
                         levels=settings.levels).to(where)
    directory = run_directory(store, name)
    directory.mkdir(parents=True, exist_ok=True)
    if init_from:
        model.load_state_dict(torch.load(run_directory(store, init_from) / "weights.pt", map_location=where))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps,
                                                   pct_start=0.05)
    config = {"name": name, "settings": asdict(settings), "bank": str(bank),
              "validation_bank": str(validation_bank) if validation_bank else None, "steps": steps, "batch": batch,
              "learning_rate": learning_rate, "seed": seed, "init_from": init_from, "reads": READS[settings.way],
              "asinh": ASINH, "star_weight": STAR_WEIGHT, "windows": train_bank.count,
              "layers": list(LAYERS) if settings.layers else [], "layer_weight": LAYER_WEIGHT,
              "spike_share": spike_share, "spiky_windows": len(spiky),
              "parameters": sum(p.numel() for p in model.parameters())}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    checks = []
    if held is not None and held.count:
        for index in range(validation_batches):
            rng = np.random.default_rng([seed, VALIDATION_STREAM, index])
            checks.append(batch_from(held, rng.integers(0, held.count, batch), settings, rng))

    def validation() -> dict[str, float]:
        totals: dict[str, float] = {}
        with torch.no_grad():
            for found in checks:
                moved = {key: value.to(where) for key, value in found.items()}
                run_loss, base_loss, parts = loss_of(model, moved, settings)
                for key, value in [("validation", float(run_loss)), ("validation_baseline", float(base_loss))] \
                        + [(f"validation_{part}", float(value)) for part, value in parts.items()]:
                    totals[key] = totals.get(key, 0.0) + value / len(checks)
        return totals

    history, began = [], time.monotonic()
    running: dict[str, float] = {}
    for step in range(1, steps + 1):
        rng = np.random.default_rng([seed, step])
        found = batch_from(train_bank, drawn(train_bank.count, spiky, spike_share, batch, rng), settings, rng)
        moved = {key: value.to(where) for key, value in found.items()}
        run_loss, base_loss, parts = loss_of(model, moved, settings)
        optimizer.zero_grad(set_to_none=True)
        run_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        for key, value in [("loss", float(run_loss)), ("baseline", float(base_loss))] \
                + [(f"loss_{part}", float(value)) for part, value in parts.items()]:
            running[key] = running.get(key, 0.0) + value
        if step % log_every == 0 or step == steps:
            window = log_every if step % log_every == 0 else step % log_every
            entry: dict[str, float] = {"step": step}
            entry.update({key: value / window for key, value in running.items()})
            entry["seconds"] = round(time.monotonic() - began, 1)
            if checks:
                entry.update(validation())
            history.append(entry)
            shown = "".join(f", {part} {entry[f'validation_{part}']:.4f} ({entry[f'validation_{part}_baseline']:.4f})"
                            for part in ("starless", *LAYERS)) if settings.layers and checks else ""
            print(f"starless {name}: step {step} loss {entry['loss']:.4f} (baseline {entry['baseline']:.4f})"
                  + (f", validation {entry['validation']:.4f} (baseline {entry['validation_baseline']:.4f})"
                     if checks else "") + shown + f" ({entry['seconds']:.0f} s)", flush=True)
            running = {}
        if step % checkpoint_every == 0 or step == steps:
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                       directory / "weights.tmp")
            (directory / "weights.tmp").replace(directory / "weights.pt")
    record = {"config": config, "history": history}
    (directory / "run.json").write_text(json.dumps(record, indent=2))
    return record


def starless(model: torch.nn.Module, planes: dict[str, np.ndarray], settings: Settings,
             where: torch.device | None = None) -> np.ndarray:
    """A window's starless image from a trained `model`, its planes as a
    bank keeps them, read as one tile (its side a multiple of the network's
    `multiple`)."""
    return layers(model, planes, settings, where)["starless"]


def layers(model: torch.nn.Module, planes: dict[str, np.ndarray], settings: Settings,
           where: torch.device | None = None) -> dict[str, np.ndarray]:
    """A window's four layers from a trained `model`, which add back to its
    input: the starless image, the halos and rings, the diffraction spikes
    (both none from a run without `layers`), and the stars' cores and wings
    (`stars`, what is left, with the stars' photon noise)."""
    where = where or next(model.parameters()).device
    read, level, scale = network_reads(planes, settings.way)
    baseline = compressed(planes[BASELINE[settings.way]], level, scale)
    with torch.no_grad():
        out = model(torch.from_numpy(np.stack(read)[None].astype(np.float32)).to(where))[0].cpu().numpy()
    found = {"starless": released(baseline + out[0], level, scale)}
    for channel, name in enumerate(LAYERS, start=1):
        found[name] = (released(out[channel], 0.0, scale) if channel < out.shape[0]
                       else np.zeros_like(found["starless"]))
    found["stars"] = (planes["input"] - found["starless"] - sum(found[name] for name in LAYERS)).astype(np.float32)
    return found


def kept(found: dict[str, np.ndarray], halo_kept: float) -> dict[str, np.ndarray]:
    """`layers`' four as a user takes them, with `halo_kept` (0 to 1) of
    the halos left in the starless image and the rest with the stars: the
    starless image, the stars (cores, wings and the halos' other share) and
    the spikes, which still add back to the input."""
    if not 0.0 <= halo_kept <= 1.0:
        raise ValueError(f"the share of the halos kept is between 0 and 1, not {halo_kept}")
    return {"starless": (found["starless"] + halo_kept * found["halos"]).astype(np.float32),
            "stars": (found["stars"] + (1.0 - halo_kept) * found["halos"]).astype(np.float32),
            "spikes": found["spikes"]}


def load_run(store: Store, name: str, where: torch.device | None = None) -> tuple[ResidualUNet, Settings]:
    """A starless run's network, with its weights, and its settings."""
    directory = run_directory(store, name)
    config = json.loads((directory / "config.json").read_text())
    settings = Settings(**config["settings"])
    model = ResidualUNet(inputs=len(READS[settings.way]), outputs=settings.outputs, width=settings.width,
                         levels=settings.levels)
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
    return model.to(where or device()).eval(), settings


# Where a star buried the sky's noise, a network trained to the median gives
# a smooth fill. After it, noise to match is drawn there (`filled`). A
# starless image's grain is read as its variance about a 3 px box mean over a
# `FILL_WINDOW` px box, against that variance's median over `FILL_BLOCK` px
# blocks of the same image: where it falls short by `FILL_START` of it or more
# (fully from `FILL_FULL`), what it lacks of the image's own noise (read the
# same way, about a `FILL_BLUR` px box mean) is drawn, shaped to its grain
# (its neighbouring pixels' correlation, read where nothing is lacking), and
# the same is taken out of the stars layer, so the layers still add to the
# image. Nothing but the image is read: a noise map is raised about a bright
# star by the star itself (23 times the sky's at one core), and a network's
# fill keeps blotches of 5 to 10 px that a wider box reads as noise, though
# its grain is gone.
FILL_BLUR = 9
FILL_WINDOW = 15
FILL_BLOCK = 96
FILL_START = 0.3
FILL_FULL = 0.5


def box_mean(plane: np.ndarray, size: int) -> np.ndarray:
    """Each pixel's mean over the `size` × `size` box about it (size odd),
    over the part of the box within the plane."""
    half = size // 2
    padded = np.pad(np.asarray(plane, dtype=np.float64), ((half + 1, half), (half + 1, half)))
    ones = np.pad(np.ones(plane.shape), ((half + 1, half), (half + 1, half)))
    sums = padded.cumsum(0).cumsum(1)
    counts = ones.cumsum(0).cumsum(1)

    def boxed(table: np.ndarray) -> np.ndarray:
        return (table[size:, size:] - table[:-size, size:] - table[size:, :-size] + table[:-size, :-size])

    return boxed(sums) / boxed(counts)


def local_noise(plane: np.ndarray, blur: int = FILL_BLUR) -> np.ndarray:
    """A plane's local noise variance: its variance about a `blur` box mean
    over a `FILL_WINDOW` box, scaled for what the box mean takes of white
    noise."""
    away = plane - box_mean(plane, blur)
    return box_mean(away**2, FILL_WINDOW) / (1.0 - 1.0 / blur**2)


def block_median(plane: np.ndarray, where: np.ndarray | None = None, block: int = FILL_BLOCK) -> np.ndarray:
    """Each pixel's block's median over the pixels `where` marks (all when
    None), the blocks `block` px a side (the last of each row and column
    taking what is left), smoothed over a block; a block with under a
    quarter of its pixels marked takes the median of the others."""
    height, width = plane.shape
    rows = list(range(0, height, block))
    columns = list(range(0, width, block))
    found = np.empty(plane.shape)
    for top in rows:
        for left in columns:
            part = plane[top:top + block, left:left + block]
            marked = np.isfinite(part) if where is None else np.isfinite(part) & where[top:top + block,
                                                                                       left:left + block]
            values = part[marked]
            found[top:top + block, left:left + block] = np.median(values) if values.size >= part.size / 4 \
                else np.nan
    found = np.where(np.isfinite(found), found, np.nanmedian(found) if np.isfinite(found).any() else 0.0)
    return box_mean(found, block | 1)


def grain(plane: np.ndarray, where: np.ndarray) -> tuple[float, float]:
    """The correlation of neighbouring pixels of a plane's noise, across and
    down, read where `where` marks, about a `FILL_BLUR` box mean."""
    away = plane - box_mean(plane, FILL_BLUR)
    found = []
    for axis in (1, 0):
        a, b = (away[:, :-1], away[:, 1:]) if axis == 1 else (away[:-1, :], away[1:, :])
        both = (where[:, :-1] & where[:, 1:]) if axis == 1 else (where[:-1, :] & where[1:, :])
        if both.sum() < 100:
            found.append(0.0)
            continue
        found.append(float(np.clip(np.corrcoef(a[both], b[both])[0, 1], 0.0, 0.45)))
    return found[0], found[1]


def grained(shape: tuple[int, int], across: float, down: float, rng: np.random.Generator) -> np.ndarray:
    """Noise of unit variance whose neighbouring pixels correlate by `across`
    and `down`: white noise through a [c, 1, c] kernel each way."""
    noise = rng.standard_normal((shape[0] + 2, shape[1] + 2))
    for axis, rho in ((1, across), (0, down)):
        c = (1.0 - np.sqrt(1.0 - 2.0 * rho**2)) / (2.0 * rho) if rho > 1e-6 else 0.0
        noise = (noise + c * (np.roll(noise, 1, axis) + np.roll(noise, -1, axis))) / np.sqrt(1.0 + 2.0 * c**2)
    return noise[1:-1, 1:-1]


def filled(image: np.ndarray, starless: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A starless image with noise drawn where it lacks it (`FILL_*`), and
    the stars layer that adds back to `image` with it."""
    starless64 = np.nan_to_num(np.asarray(starless, dtype=np.float64))
    fine = local_noise(starless64, 3)
    # The image's own grain, read twice: the second time away from where the
    # first found it lacking, so a wide fill does not lower its own measure.
    quiet = None
    for _ in range(2):
        lacking = np.clip(1.0 - fine / np.maximum(block_median(fine, quiet), 1e-30), 0.0, 1.0)
        lacking = box_mean(lacking, FILL_WINDOW)
        quiet = lacking < FILL_START / 2
    ramp = np.clip((lacking - FILL_START) / (FILL_FULL - FILL_START), 0.0, 1.0)
    variance = block_median(local_noise(starless64), quiet)
    across, down = grain(starless64, lacking < 0.2)
    drawn = grained(starless64.shape, across, down, np.random.default_rng(seed))
    out = starless64 + drawn * np.sqrt(ramp * lacking * np.maximum(variance, 0.0))
    out = np.where(np.isfinite(starless), out, starless).astype(np.float32)
    return out, (np.asarray(image, dtype=np.float64) - out).astype(np.float32)


def measures(estimate: np.ndarray, planes: dict[str, np.ndarray]) -> dict[str, float | None]:
    """How an estimate of a window's starless image meets its truth, over
    its valid pixels and in units of its noise map's median: the error's
    root mean square where stars were (where the input and the truth differ
    by more than the noise) and on the sky elsewhere; its mean on the stars'
    cores (their light over ten times the noise: below 0 a hole, above a
    remnant); and the share of the stars' light left in the estimate. A
    bright core's error is thousands of times the noise, so beside these,
    the mean absolute error as the trainer reads images (`network_reads`'
    asinh, closer to a stretched image's eye) where stars were and on the
    sky; and how much of the truth's noise the estimate keeps where stars
    were (its local noise over the truth's: under 1 a smooth fill)."""
    scale = float(np.median(planes["noise"])) or 1.0
    level = float(np.nanmedian(planes["input"]))
    valid = planes["valid"] > 0
    light = planes["input"].astype(np.float64) - planes["starless"]
    stars = valid & (np.abs(light) > planes["noise"])
    cores = valid & (light > 10.0 * planes["noise"])
    sky = valid & ~stars
    error = (estimate.astype(np.float64) - planes["starless"]) / scale
    seen = np.abs(compressed(estimate, level, scale) - compressed(planes["starless"], level, scale))

    def rms(where: np.ndarray) -> float | None:
        return float(np.sqrt(np.mean(error[where] ** 2))) if where.any() else None

    def stretched(where: np.ndarray) -> float | None:
        return float(np.mean(seen[where])) if where.any() else None

    total = float(light[stars].sum())
    kept = None
    if stars.any():
        truth_noise = float(np.mean(local_noise(np.asarray(planes["starless"], dtype=np.float64), 3)[stars]))
        if truth_noise > 0:
            kept = math.sqrt(float(np.mean(local_noise(np.nan_to_num(estimate.astype(np.float64)), 3)[stars]))
                             / truth_noise)
    return {"stars_rms": rms(stars), "sky_rms": rms(sky),
            "cores_mean": float(error[cores].mean()) if cores.any() else None,
            "light_left": float((error[stars] * scale).sum()) / total if total > 0 else None,
            "stars_stretched": stretched(stars), "sky_stretched": stretched(sky), "noise_kept": kept}


def layer_measures(found: dict[str, np.ndarray], planes: dict[str, np.ndarray]) -> dict[str, float | None]:
    """How a run's halos and spikes (`layers`) meet the bank's, in units of
    the noise map's median, over the valid pixels: for each, the error's
    root mean square where the truth's light tops the noise; how much of
    the truth the layer holds there, as the scale of the truth that best
    meets it (1 all, 0 none); and the error's root mean square where the
    truth holds under a tenth of the noise (light put in the layer that is
    not its own: the sky, the nebula or the stars'). A spike's light is
    signed, as the vanes take from the core the light they throw out along
    the spikes; the dip at the core is not what shows, and is read by
    neither."""
    scale = float(np.median(planes["noise"])) or 1.0
    valid = planes["valid"] > 0
    out: dict[str, float | None] = {}
    for name in LAYERS:
        truth = planes[name].astype(np.float64)
        estimate = found[name].astype(np.float64)
        inside = valid & (truth > planes["noise"])
        outside = valid & (np.abs(truth) < 0.1 * planes["noise"])
        power = float((truth[inside] ** 2).sum())
        out[f"{name}_rms"] = float(np.sqrt(np.mean(((estimate - truth)[inside] / scale) ** 2))) \
            if inside.any() else None
        out[f"{name}_found"] = float((estimate[inside] * truth[inside]).sum()) / power if inside.any() else None
        out[f"{name}_stray"] = float(np.sqrt(np.mean(((estimate - truth)[outside] / scale) ** 2))) \
            if outside.any() else None
    return out


def mean_of(found: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    """Each measure's median and mean over the windows that have it: the
    median, as a few windows' bright cores can carry the mean."""
    out: dict[str, dict[str, float | None]] = {"median": {}, "mean": {}}
    for key in (found[0] if found else {}):
        values = [entry[key] for entry in found if entry[key] is not None]
        out["median"][key] = float(np.median(values)) if values else None
        out["mean"][key] = float(np.mean(values)) if values else None
    return out


def evaluate(store: Store, name: str, bank: Path, *, count: int = 64, sheet_rows: int = 6,
             where: torch.device | None = None) -> dict[str, Any]:
    """A run's `measures` on `count` windows of a bank, evenly spread, beside
    its baseline's and those of its output with noise drawn where it lacks
    it (`filled`), over all and by telescope kind; and a sheet of the first
    `sheet_rows` of them: the input, the baseline, the network, the network
    filled, the truth and the filled output's stars layer, stretched alike.
    A run with `layers` is measured on its halos and spikes too
    (`layer_measures`), and its sheet shows them beside the bank's, its
    stars layer then the cores and wings alone. Both are kept under the
    run's `evaluation/`, named for the bank."""
    model, settings = load_run(store, name, where)
    held = Bank(bank)
    indices = np.unique(np.linspace(0, held.count - 1, min(count, held.count)).round().astype(int))
    results: dict[str, list[dict[str, float | None]]] = {}
    rows = []
    for index in indices:
        planes = {plane: np.asarray(held.planes[plane][index], dtype=np.float32) for plane in held.planes}
        found = layers(model, planes, settings)
        estimate = found["starless"]
        baseline = planes[BASELINE[settings.way]]
        fill, stars = filled(planes["input"], estimate, seed=int(index))
        stars = stars - found["halos"] - found["spikes"]
        measured = measures(estimate, planes)
        if settings.layers:
            measured.update(layer_measures(found, planes))
        kind = held.records[index].get("telescope_kind", "all")
        for group in ("all", kind):
            results.setdefault(f"{group}/run", []).append(measured)
            results.setdefault(f"{group}/filled", []).append(measures(fill, planes))
            results.setdefault(f"{group}/baseline", []).append(measures(baseline, planes))
        if len(rows) < sheet_rows:
            tiles = [planes["input"], baseline, estimate, fill, planes["starless"], stars]
            if settings.layers:
                tiles += [found["halos"], planes["halos"], found["spikes"], planes["spikes"]]
            rows.append((held.records[index], tiles))
    summary = {"run": name, "bank": str(bank), "windows": int(len(indices)),
               "measures": {key: mean_of(found) for key, found in sorted(results.items())}}
    directory = run_directory(store, name) / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    label = Path(bank).name
    (directory / f"{label}.json").write_text(json.dumps(summary, indent=2))
    if rows:
        write_sheet(directory / f"{label}.png", rows, settings.way)
    return summary


# A sheet's columns: the starless images, then the layers of the stars'
# light, which a layered run adds.
SHEET_IMAGES = ("input", "baseline", "network", "network, noise filled", "truth")
SHEET_LAYERS = ("stars", "halos", "halos, truth", "spikes", "spikes, truth")


def write_sheet(path: Path, rows: list[tuple[dict[str, Any], list[np.ndarray]]], way: str) -> None:
    """A starless sheet: a row a window (input, baseline, network, network
    filled, truth, stars layer, and a layered run's halos and spikes beside
    the bank's), each stretched on its truth's levels, the layers of the
    stars' light from the truth's floor."""
    from PIL import Image, ImageDraw

    images = len(SHEET_IMAGES)
    titles = list(SHEET_IMAGES + SHEET_LAYERS)[:len(rows[0][1])]
    titles[1] = "fit's starless" if way == "fit" else "input (baseline)"
    if len(titles) > images + 1:
        titles[images] = "stars, cores and wings"
    strips = []
    for record, tiles in rows:
        low, high = np.percentile(np.nan_to_num(tiles[images - 1]), (1.0, 99.5))

        def look(plane: np.ndarray, low: float = low, high: float = high) -> np.ndarray:
            t = np.clip((np.nan_to_num(plane) - low) / max(high - low, 1e-12), 0.0, 1.0)
            return (np.arcsinh(t * 30.0) / np.arcsinh(30.0) * 255.0).astype(np.uint8)

        shown = [look(tile) for tile in tiles[:images]] + [look(tile + low) for tile in tiles[images:]]
        gap = np.full((shown[0].shape[0], 6), 40, np.uint8)
        strip = np.concatenate([part for tile in shown for part in (tile, gap)][:-1], axis=1)
        strips.append((f"{record.get('product', '')} draw {record.get('draw', '')}: {record.get('telescope', '')}",
                       strip))
    width = max(strip.shape[1] for _, strip in strips)
    canvas = Image.new("L", (width, 18 + sum(strip.shape[0] + 18 for _, strip in strips)), 20)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), " | ".join(titles), fill=230)
    top = 18
    for label, strip in strips:
        draw.text((4, top + 2), label, fill=200)
        canvas.paste(Image.fromarray(strip), (0, top + 18))
        top += strip.shape[0] + 18
    canvas.save(path)
