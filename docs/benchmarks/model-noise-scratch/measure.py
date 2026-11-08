"""Measure native model-noise preprocessing, without an inference runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--side", type=int, default=2048)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--count-allocations", action="store_true")
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="seiza-noise-probe-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        (work / "src/main.rs").write_text(
            Path(__file__).with_name("probe.rs").read_text().replace("@ROOT@", str(root)).replace("@COUNT@", str(args.count_allocations).lower())
        )
        (work / "Cargo.toml").write_text(
            '[package]\nname = "seiza-noise-probe"\nversion = "0.1.0"\nedition = "2024"\n'
            '[dependencies]\nrayon = "1"\nseiza-model = { path = '
            + json.dumps(str(root / "seiza-model")) + ' }\n'
        )
        target = args.target_dir.resolve() if args.target_dir else work / "target"
        subprocess.run(
            ["cargo", "build", "--release", "--offline", "--quiet", "--manifest-path", str(work / "Cargo.toml"), "--target-dir", str(target)],
            check=True,
        )
        command = [str(target / "release/seiza-noise-probe"), str(args.side)]
        if args.resources:
            command = ["/usr/bin/time", "-v", "-o", str(args.resources.resolve()), *command]
        with args.output.open("w") as output:
            subprocess.run(command, check=True, stdout=output)



if __name__ == "__main__":
    main()
