"""Benchmark forced photometry on the five pinned V1337 Cas frames.

Use the same optimized build profile and this script for both revisions.
FITS decoding, green extraction, downloads and hashing are outside timings.
This is a real-pixel performance/parity check, not Kai's full ALP reduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import seiza

MANIFEST = Path(__file__).with_name("v1337-cas-v1.json")
FIELDS = (
    "x", "y", "flux", "flux_error", "magnitude", "magnitude_error",
    "background", "background_noise", "aperture_sum", "aperture_area",
    "annulus_pixels", "peak",
)
FLAGS = ("saturated", "missing_coverage", "unusable")


def verified_file(directory, entry, base_url, download):
    path = directory / entry["file"]

    def verify(candidate):
        if candidate.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {candidate}")
        with candidate.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {candidate}")

    if path.exists():
        verify(path)
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}; use --download to fetch the dataset")
    directory.mkdir(parents=True, exist_ok=True)
    url = base_url.rstrip("/") + "/" + quote(entry["file"], safe="")
    print(f"Downloading {entry['file']}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as dest:
        temporary = Path(dest.name)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as dest:
            shutil.copyfileobj(response, dest)
        verify(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_green(path):
    header = seiza.read_image_header(path)["headers"]
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    if (width, height, header.get("BAYERPAT", "").strip()) != (2160, 3840, "GRBG"):
        raise ValueError(f"Unexpected sensor layout: {path}")
    raw = np.empty((height, width), np.float32)
    # read_image_window accepts sides up to 1024, in stored row order.
    for y in range(0, height, 1024):
        for x in range(0, width, 1024):
            tile = seiza.read_image_window(
                path, x, y, min(1024, width-x), min(1024, height-y))
            raw[y:y+tile.height, x:x+tile.width] = tile.values.reshape(tile.height, tile.width)
    return np.ascontiguousarray((raw[0::2, 0::2] + raw[1::2, 1::2]) * 0.5)


def compare(reference, result):
    for name in ("dataset", "geometry", "fields", "frames"):
        if reference[name] != result[name]:
            raise ValueError(f"Comparison inputs differ: {name}")
    if len(reference["rows"]) != len(result["rows"]):
        raise ValueError("Comparison scenario counts differ")
    ratios = []
    for old, new in zip(reference["rows"], result["rows"]):
        for name in ("file", "stars", "layout", "strides", "values"):
            if old[name] != new[name]:
                raise ValueError(f"Comparison differs in {name}: {new['file']}, "
                                 f"{new['stars']} stars, {new['layout']}")
        ratios.append(dict(file=new["file"], stars=new["stars"], layout=new["layout"],
                           speedup=old["ms"] / new["ms"]))
    return dict(exact_parity=True, speedups=ratios)


def benchmark(args, manifest):
    result = dict(dataset=manifest["dataset"], geometry=manifest["geometry"],
                  fields=[*FIELDS, *FLAGS], frames=manifest["frames"],
                  version=seiza.__version__, module=seiza.__file__,
                  python=sys.version, numpy=np.__version__, platform=platform.platform(),
                  iterations=args.iterations, rounds=args.rounds, rows=[])
    for entry in manifest["frames"]:
        path = verified_file(args.data_dir, entry, manifest["base_url"], args.download)
        green = read_green(path)
        rgb = np.stack([green, green, green], axis=-1)
        mask = np.zeros(green.shape, np.uint8)
        uncertainty = np.full_like(green, 3)
        for count in (1, 8, 9, 100):
            positions = entry["positions"][:count]
            for layout, plane, maps in (
                ("mono", green, {}), ("green", rgb[..., 1], {}),
                ("mapped", green, dict(mask=mask, saturation_mask=mask, uncertainty=uncertainty)),
            ):
                def run():
                    return seiza.aperture_photometry(plane, positions, **manifest["geometry"], **maps)

                values = [[getattr(m, f) for f in FIELDS] +
                          [getattr(m.flags, f) for f in FLAGS] for m in run()]
                samples = []
                for _ in range(args.rounds):
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        run()
                    samples.append((time.perf_counter()-start)*1000/args.iterations)
                result["rows"].append(dict(file=entry["file"], stars=count, layout=layout,
                                           strides=list(plane.strides), values=values,
                                           ms=statistics.median(samples), rounds_ms=samples))
        print(f"Measured {entry['file']}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download missing files (about 83 MB)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Check exact parity with a previous output JSON")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        parser.error("iterations and rounds must be positive")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = benchmark(args, manifest)
    if args.compare:
        result["comparison"] = compare(
            json.loads(args.compare.read_text(encoding="utf-8")), result)
        print("Exact numerical parity passed", file=sys.stderr)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
