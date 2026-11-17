"""Benchmark sparse photometry on synthetic mono and strided green planes.

Build both revisions with the same optimized maturin profile and interpreter.
Run this script on each, redirecting stdout to before.json and after.json.
Each record includes all results so numerical parity can be checked separately
from timing. This does not model a particular camera or benchmark Photutils.
"""
import argparse
import json
import statistics
import time

import numpy as np
import seiza


def benchmark(iterations, rounds):
    rng = np.random.default_rng(479)
    rows = []
    fields = (
        "x", "y", "flux", "flux_error", "magnitude", "magnitude_error",
        "background", "background_noise", "aperture_sum", "aperture_area",
        "annulus_pixels", "peak",
    )
    for width, height in ((512, 384), (1920, 1080)):
        image = rng.normal(100, 3, (height, width)).astype(np.float32)
        rgb = np.stack([image, image, image], axis=-1)
        mask = np.zeros(image.shape, np.uint8)
        uncertainty = np.full_like(image, 3)
        for count in (1, 9, 100):
            positions = [
                (32 + (i * 173) % (width - 64) + .3,
                 32 + (i * 97) % (height - 64) + .7)
                for i in range(count)
            ]
            layouts = (
                ("mono", image, {}),
                ("green", rgb[..., 1], {}),
                ("mapped", image, dict(mask=mask, saturation_mask=mask,
                                      uncertainty=uncertainty)),
            )
            for layout, plane, maps in layouts:
                def run():
                    return seiza.aperture_photometry(
                        plane, positions, aperture_radius=6, annulus_inner=10,
                        annulus_outer=16, **maps)

                results = run()  # Warm up before timing.
                values = [
                    [getattr(result, field) for field in fields]
                    + [result.flags.saturated, result.flags.missing_coverage,
                       result.flags.unusable]
                    for result in results
                ]
                times = []
                for _ in range(rounds):
                    start = time.perf_counter()
                    for _ in range(iterations):
                        run()
                    times.append((time.perf_counter() - start) * 1000 / iterations)
                rows.append(dict(width=width, height=height, stars=count,
                                 layout=layout, ms=statistics.median(times),
                                 values=values))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        parser.error("iterations and rounds must be positive")
    print(json.dumps(benchmark(args.iterations, args.rounds)))
