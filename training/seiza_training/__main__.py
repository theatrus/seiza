"""`python -m seiza_training <pass> --store DIR ...`: run one corpus pass.

The passes run in order — crawl, calibrate, split, measure, build,
review — and each can be rerun alone. `process` measures and
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
        if corpus is None and library is None:
            raise SystemExit("crawl needs --corpus or --library the first time")
        manifest.set_meta(
            "roots",
            {"corpus": str(corpus) if corpus else None, "library": str(library) if library else None},
        )
        return crawl(manifest, corpus, library, workers=arguments.workers)
    if command == "calibrate":
        from .calibrate import calibrate

        return calibrate(manifest, only=only)
    if command == "measure":
        from .measure import measure

        return measure(
            manifest, only=only, again=arguments.again, catalog_dir=arguments.catalog_dir,
            limit=arguments.limit,
        )
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
