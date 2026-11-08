"""Measure allocations made by the native Tiling.tiles API."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="seiza-tile-count-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        (work / "src/main.rs").write_text(Path(__file__).with_name("probe.rs").read_text())
        (work / "Cargo.toml").write_text(
            '[package]\nname = "seiza-geometry-probe"\nversion = "0.1.0"\n'
            'edition = "2024"\n[dependencies]\nseiza-model = { path = '
            + json.dumps(str(root / "seiza-model"))
            + " }\n"
        )
        with args.output.open("w") as output:
            subprocess.run(
                ["cargo", "run", "--offline", "--quiet", "--manifest-path", str(work / "Cargo.toml")],
                stdout=output,
                check=True,
            )


if __name__ == "__main__":
    main()
