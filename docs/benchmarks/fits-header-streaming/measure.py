"""Measure raw FITS header retention in a separate process (Linux)."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="seiza-header-probe-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        (work / "Cargo.toml").write_text(
            '[package]\nname = "seiza-header-probe"\nversion = "0.1.0"\n'
            'edition = "2024"\n[dependencies]\nseiza-fits = { path = '
            + json.dumps(str(root / "seiza-fits"))
            + " }\n"
        )
        (work / "src/main.rs").write_text(
            'fn main() {\n'
            '    let path = std::env::args_os().nth(1).unwrap();\n'
            '    println!("{:?}", seiza_fits::read_header(std::path::Path::new(&path)));\n'
            '}\n'
        )
        subprocess.run(
            [
                "cargo", "build", "--offline", "--manifest-path", str(work / "Cargo.toml"),
                "--target-dir", str(work / "target"),
            ],
            check=True,
        )
        first = bytearray(b" " * 2880)
        for index, card in enumerate(
            ["SIMPLE  =                    T", "BITPIX  =                    8", "NAXIS   =                    0"]
        ):
            first[index * 80 : index * 80 + len(card)] = card.encode("ascii")
        for label, complete in [("valid", True), ("unfinished", False)]:
            fixture = work / f"{label}.fits"
            with fixture.open("wb") as stream:
                stream.write(first)
                for _ in range(32767):
                    stream.write(b" " * 2880)
                if complete:
                    stream.write(b"END" + b" " * 2877)
            with (args.output / f"{label}-result.txt").open("w") as result:
                subprocess.run(
                    [
                        "/usr/bin/time", "-v", "-o",
                        str(args.output / f"{label}-resources.txt"),
                        str(work / "target/debug/seiza-header-probe"), str(fixture),
                    ],
                    stdout=result,
                    check=True,
                )


            expected = (
                'Ok([("SIMPLE", Logical(true)), ("BITPIX", Integer(8)), ("NAXIS", Integer(0))])\n'
                if complete
                else 'Err(Malformed("header runs past EOF"))\n'
            )
            actual = (args.output / f"{label}-result.txt").read_text()
            if actual != expected:
                raise RuntimeError(f"{label}: unexpected header result: {actual!r}")


if __name__ == "__main__":
    main()
