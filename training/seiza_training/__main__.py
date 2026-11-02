"""`python -m seiza_training <pass> --store DIR ...`: run one corpus pass.

The passes run in order — crawl, calibrate, stacks, split, measure,
build, review — and each can be rerun alone. `process` measures and
builds one set at a time, which reads each frame over the network once.
`status` prints what the manifest holds; `decide` records a review
decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import Manifest
from .store import Store, versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m seiza_training", description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="corpus store directory")
    parser.add_argument(
        "--scratch", type=Path,
        help="where build keeps registered frames, on local disk (default ~/.cache/seiza-training/scratch)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    crawl = commands.add_parser("crawl", help="read every header and group lights into sets")
    crawl.add_argument("--corpus", type=Path, help="the telescope corpus (remembered)")
    crawl.add_argument("--library", type=Path, help="the calibration-master library (remembered)")
    crawl.add_argument("--calibration", type=Path, help="raw calibration frames, darks among them (remembered)")
    crawl.add_argument("--workers", type=int, default=16)

    for name, text in (
        ("calibrate", "choose calibration masters for each set"),
        ("measure", "measure every frame of the calibratable sets"),
        ("build", "register each set once and build its products"),
        ("process", "measure, then build, one set at a time"),
        ("review", "write the review pages"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--set", action="append", dest="sets", help="only this set id")
        if name in ("measure", "build"):
            command.add_argument("--again", action="store_true", help="redo sets already done")
        if name in ("measure", "build", "process"):
            command.add_argument("--limit", type=int, help="at most this many sets")
        if name in ("measure", "process"):
            command.add_argument("--catalog-dir", help="catalogs for plate solving")
        if name in ("build", "process"):
            command.add_argument("--keep-scratch", action="store_true")

    stacks = commands.add_parser("stacks", help="copy in the finished stacks PSF Guard made from the corpus")
    stacks.add_argument("--psf-guard", type=Path, help="PSF Guard's stacks directory (remembered)")
    stacks.add_argument("--again", action="store_true", help="copy stacks already copied")
    stacks.add_argument("--limit", type=int, help="at most this many stacks")

    dark = commands.add_parser("dark-noise", help="make noise pairs with a known clean image from dark frames")
    dark.add_argument("--set", action="append", dest="sets", help="only this set id")
    dark.add_argument("--size", type=int, action="append", dest="sizes",
                      help="the A half whose frames lend their registration, by size (default 8 and 16)")
    dark.add_argument("--again", action="store_true", help="make pairs already made")
    dark.add_argument("--limit", type=int, help="at most this many pairs")

    train = commands.add_parser("train-noise", help="train a noise model on the train split's noise pairs")
    train.add_argument("--name", required=True, help="the run's name, under STORE/models/noise/")
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--levels", type=int, default=4)
    train.add_argument("--tile", type=int, default=256)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--set", action="append", dest="sets", help="only this train set id")
    train.add_argument("--validation-set", action="append", dest="validation_sets",
                       help="measure on only this validation set id")
    train.add_argument("--loss-weight", choices=("core", "variance"), default="core")
    train.add_argument("--map-jitter", type=float, default=0.0,
                       help="scale the noise map the network reads by up to exp(±this)")
    train.add_argument("--noise-map", choices=("raw", "model"), default="raw",
                       help="the stack's per-pixel noise map, or the modelled one")
    train.add_argument("--huber", type=float, default=0.0,
                       help="square each pixel's weighted error only within this of zero")
    train.add_argument("--huber-above", type=float, default=0.0,
                       help="make only pixels whose target is this many σ̄ above the background robust")
    train.add_argument("--structure-sampling", type=float, default=0.0,
                       help="draw windows 1 + this × their share of bright pixels as often")
    train.add_argument("--dark-noise-fraction", type=float, default=0.0,
                       help="this share of every batch from the dark-noise pairs, whose target is clean")

    evaluate = commands.add_parser("evaluate-noise", help="measure a trained noise model on a split")
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--limit", type=int, help="at most this many pairs")
    evaluate.add_argument("--set", action="append", dest="sets", help="only this set id")

    split = commands.add_parser("split", help="assign sky groups and train/validation/test splits")
    split.add_argument("--hold-out", help="a telescope whose groups are all test")

    decide = commands.add_parser("decide", help="record a review decision")
    decide.add_argument("subject", help="a set id")
    decide.add_argument("kind", choices=("set", "starless"))
    decide.add_argument("decision", choices=("accept", "reject"))
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--note")

    commands.add_parser("status", help="summarize the manifest")

    arguments = parser.parse_args(argv)
    store = Store(arguments.store, arguments.scratch)
    only = set(arguments.sets) if getattr(arguments, "sets", None) else None
    with Manifest(store.manifest_path) as manifest:
        if arguments.command == "decide":
            manifest.record_review(
                arguments.subject, arguments.kind, arguments.decision, arguments.reviewer,
                arguments.note,
            )
            return 0
        if arguments.command == "status":
            print(json.dumps(status(manifest), indent=2))
            return 0
        recorded = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
            if key not in ("store", "scratch", "command")
        }
        with manifest.pass_run(arguments.command, versions(), recorded) as summary:
            summary.update(run(arguments, manifest, store, only))
        print(json.dumps(summary, indent=2, default=str))
    return 0


def run(arguments: argparse.Namespace, manifest: Manifest, store: Store, only: set[str] | None) -> dict:
    command = arguments.command
    if command == "crawl":
        from .crawl import crawl

        roots = manifest.meta("roots") or {}
        corpus = arguments.corpus or (Path(roots["corpus"]) if roots.get("corpus") else None)
        library = arguments.library or (Path(roots["library"]) if roots.get("library") else None)
        calibration = arguments.calibration or (Path(roots["calibration"]) if roots.get("calibration") else None)
        if corpus is None and library is None:
            raise SystemExit("crawl needs --corpus or --library the first time")
        manifest.set_meta(
            "roots",
            {**roots, "corpus": str(corpus) if corpus else None, "library": str(library) if library else None,
             "calibration": str(calibration) if calibration else None},
        )
        return crawl(manifest, corpus, library, calibration=calibration, workers=arguments.workers)
    if command == "calibrate":
        from .calibrate import calibrate

        return calibrate(manifest, only=only)
    if command == "measure":
        from .measure import measure

        return measure(
            manifest, only=only, again=arguments.again, catalog_dir=arguments.catalog_dir,
            limit=arguments.limit,
        )
    if command == "stacks":
        from .stacks import import_stacks

        roots = manifest.meta("roots") or {}
        stacks_root = arguments.psf_guard or (Path(roots["psf_guard"]) if roots.get("psf_guard") else None)
        if stacks_root is None:
            raise SystemExit("stacks needs --psf-guard the first time")
        if not roots.get("corpus"):
            raise SystemExit("stacks needs the corpus crawled first")
        manifest.set_meta("roots", {**roots, "psf_guard": str(stacks_root)})
        return import_stacks(
            manifest, store, stacks_root, Path(roots["corpus"]), again=arguments.again,
            limit=arguments.limit,
        )
    if command == "dark-noise":
        from .dark_noise import SIZES
        from .dark_noise import run as make_dark_noise

        return make_dark_noise(
            manifest, store, sets=only, sizes=tuple(arguments.sizes or SIZES), again=arguments.again,
            limit=arguments.limit,
        )
    if command == "train-noise":
        from .noise_model import dark_noise_bank, train

        extra = {}
        if arguments.dark_noise_fraction:
            extra = {
                "extra": dark_noise_bank(store, manifest, sets=only, seed=arguments.seed, noise_map=arguments.noise_map),
                "extra_fraction": arguments.dark_noise_fraction,
                "extra_label": "dark-noise",
            }
        return train(
            store, manifest, name=arguments.name, steps=arguments.steps, width=arguments.width,
            levels=arguments.levels, tile=arguments.tile, batch=arguments.batch,
            learning_rate=arguments.learning_rate, sets=only, seed=arguments.seed,
            loss_weight=arguments.loss_weight, map_jitter=arguments.map_jitter,
            validation_sets=set(arguments.validation_sets) if arguments.validation_sets else None,
            noise_map=arguments.noise_map, huber=arguments.huber, huber_above=arguments.huber_above,
            structure_sampling=arguments.structure_sampling, **extra,
        )
    if command == "evaluate-noise":
        from .noise_model import evaluate_saved

        return evaluate_saved(store, manifest, arguments.name, split=arguments.split, limit=arguments.limit, sets=only)
    if command == "split":
        from .splits import assign_splits

        return assign_splits(manifest, hold_out=arguments.hold_out)
    if command == "build":
        from .build import build

        return build(
            manifest, store, only=only, again=arguments.again, limit=arguments.limit,
            keep_scratch=arguments.keep_scratch,
        )
    if command == "process":
        from .build import process

        return process(
            manifest, store, only=only, limit=arguments.limit, catalog_dir=arguments.catalog_dir,
            keep_scratch=arguments.keep_scratch,
        )
    if command == "review":
        from .review import review

        return review(manifest, store, only=only)
    raise AssertionError(command)


def status(manifest: Manifest) -> dict:
    def counts(query: str) -> dict:
        return {str(row[0]): row[1] for row in manifest.rows(query)}

    return {
        "frames": counts("SELECT origin || ':' || COALESCE(role, 'unread'), COUNT(*) FROM frames GROUP BY 1"),
        "sets": manifest.rows("SELECT COUNT(*) FROM sets")[0][0],
        "calibration": counts("SELECT calibration_status, COUNT(*) FROM sets GROUP BY 1"),
        "measurement": counts("SELECT measurement_status, COUNT(*) FROM sets GROUP BY 1"),
        "split": counts("SELECT split, COUNT(*) FROM sets GROUP BY 1"),
        "build": counts("SELECT build_status, COUNT(*) FROM sets GROUP BY 1"),
        "stacks": counts("SELECT telescope || ':' || status || ':' || COALESCE(split, 'unsplit'), COUNT(*) FROM stacks GROUP BY 1"),
        "products": counts("SELECT kind || ':' || status, COUNT(*) FROM products GROUP BY 1"),
        "reviews": counts("SELECT kind || ':' || decision, COUNT(*) FROM reviews GROUP BY 1"),
        "passes": [
            dict(row) for row in manifest.rows(
                "SELECT name, started, finished FROM passes ORDER BY started DESC LIMIT 10"
            )
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
