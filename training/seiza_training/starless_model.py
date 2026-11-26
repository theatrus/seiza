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
"""

from __future__ import annotations

import json
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
# The compression: planes in units of the noise, through asinh(x / ASINH).
ASINH = 5.0
# Where the image and its truth differ by over the noise, the loss weighs
# this many times as much.
STAR_WEIGHT = 10.0
CHARBONNIER = 0.01
VALIDATION_STREAM = 1 << 20


@dataclass(frozen=True)
class Settings:
    """What a run reads and how: `way` (`READS`), the tile's side, the
    network's width and levels."""

    way: str = "fit"
    tile: int = 256
    width: int = 32
    levels: int = 5

    def __post_init__(self) -> None:
        if self.way not in READS:
            raise ValueError(f"a starless run reads one of {sorted(READS)}, not {self.way}")
        if self.tile % 2 ** (self.levels - 1):
            raise ValueError(f"a tile of {self.tile} is not a multiple of {2 ** (self.levels - 1)}")


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


def example(bank: Bank, index: int, settings: Settings, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A tile of window `index`: the network's planes, the baseline and the
    truth compressed, and the loss's weights; flipped and turned at random."""
    side = bank.planes["input"].shape[1]
    tile = settings.tile
    top, left = (int(rng.integers(0, side - tile + 1)) for _ in range(2))
    cut = (index, slice(top, top + tile), slice(left, left + tile))
    planes = {name: np.asarray(bank.planes[name][cut], dtype=np.float32) for name in bank.planes}
    valid = planes["valid"] > 0
    level = float(np.median(planes["input"][valid])) if valid.any() else float(np.median(planes["input"]))
    scale = float(np.median(planes["noise"])) or 1.0
    read = []
    for name in READS[settings.way]:
        if name == "noise":
            read.append(planes["noise"] / scale)
        elif name == "points":
            read.append(planes["points"])
        else:
            read.append(compressed(planes[name], level, scale))
    baseline = compressed(planes[BASELINE[settings.way]], level, scale)
    truth = compressed(planes["starless"], level, scale)
    weight = valid.astype(np.float32) * (1.0 + (STAR_WEIGHT - 1.0)
                                         * (np.abs(planes["input"] - planes["starless"]) > planes["noise"]))
    stacked = np.stack(read + [baseline, truth, weight])
    turns = int(rng.integers(0, 4))
    stacked = np.rot90(stacked, turns, axes=(1, 2))
    if rng.random() < 0.5:
        stacked = stacked[:, :, ::-1]
    stacked = np.ascontiguousarray(stacked, dtype=np.float32)
    return {"planes": stacked[:-3], "baseline": stacked[-3], "truth": stacked[-2], "weight": stacked[-1]}


def batch_from(bank: Bank, indices: np.ndarray, settings: Settings, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    examples = [example(bank, int(index), settings, rng) for index in indices]
    return {key: torch.from_numpy(np.stack([found[key] for found in examples])) for key in examples[0]}


def loss_of(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The run's loss on `batch`, and its baseline's (the residual 0)."""
    residual = model(batch["planes"])[:, 0]
    weight = batch["weight"]
    total = weight.sum().clamp_min(1.0)

    def penalty(estimate: torch.Tensor) -> torch.Tensor:
        return (weight * torch.sqrt((estimate - batch["truth"]) ** 2 + CHARBONNIER**2)).sum() / total

    return penalty(batch["baseline"] + residual), penalty(batch["baseline"]).detach()


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
          init_from: str | None = None) -> dict[str, Any]:
    """Train a starless run on `bank` and keep it under
    `STORE/models/starless/<name>/`: its weights, its configuration and
    its log of the loss, the validation loss and the baseline's."""
    where = where or device()
    torch.manual_seed(seed)
    train_bank = Bank(bank)
    held = Bank(validation_bank) if validation_bank else None
    model = ResidualUNet(inputs=len(READS[settings.way]), outputs=1, width=settings.width,
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
              "parameters": sum(p.numel() for p in model.parameters())}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    checks = []
    if held is not None and held.count:
        for index in range(validation_batches):
            rng = np.random.default_rng([seed, VALIDATION_STREAM, index])
            checks.append(batch_from(held, rng.integers(0, held.count, batch), settings, rng))

    def validation() -> dict[str, float]:
        totals = [0.0, 0.0]
        with torch.no_grad():
            for found in checks:
                moved = {key: value.to(where) for key, value in found.items()}
                run_loss, base_loss = loss_of(model, moved)
                totals[0] += float(run_loss) / len(checks)
                totals[1] += float(base_loss) / len(checks)
        return {"validation": totals[0], "validation_baseline": totals[1]}

    history, running, began = [], [0.0, 0.0], time.monotonic()
    for step in range(1, steps + 1):
        rng = np.random.default_rng([seed, step])
        found = batch_from(train_bank, rng.integers(0, train_bank.count, batch), settings, rng)
        moved = {key: value.to(where) for key, value in found.items()}
        run_loss, base_loss = loss_of(model, moved)
        optimizer.zero_grad(set_to_none=True)
        run_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        running[0] += float(run_loss)
        running[1] += float(base_loss)
        if step % log_every == 0 or step == steps:
            window = log_every if step % log_every == 0 else step % log_every
            entry = {"step": step, "loss": running[0] / window, "baseline": running[1] / window,
                     "seconds": round(time.monotonic() - began, 1)}
            if checks:
                entry.update(validation())
            history.append(entry)
            print(f"starless {name}: step {step} loss {entry['loss']:.4f} (baseline {entry['baseline']:.4f})"
                  + (f", validation {entry['validation']:.4f} (baseline {entry['validation_baseline']:.4f})"
                     if checks else "") + f" ({entry['seconds']:.0f} s)", flush=True)
            running = [0.0, 0.0]
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
    where = where or next(model.parameters()).device
    valid = planes.get("valid", np.ones_like(planes["input"])) > 0
    level = float(np.median(planes["input"][valid]))
    scale = float(np.median(planes["noise"])) or 1.0
    read = []
    for name in READS[settings.way]:
        if name == "noise":
            read.append(planes["noise"] / scale)
        elif name == "points":
            read.append(planes["points"])
        else:
            read.append(compressed(planes[name], level, scale))
    baseline = compressed(planes[BASELINE[settings.way]], level, scale)
    with torch.no_grad():
        residual = model(torch.from_numpy(np.stack(read)[None].astype(np.float32)).to(where))[0, 0].cpu().numpy()
    return released(baseline + residual, level, scale)


def charbonnier_mean(estimate: np.ndarray, truth: np.ndarray, valid: np.ndarray, scale: float) -> float:
    """The mean Charbonnier distance of two images in units of `scale`."""
    difference = (estimate - truth)[valid] / scale
    return float(np.mean(np.sqrt(difference**2 + CHARBONNIER**2)))
