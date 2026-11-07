"""Measure a mapped one-pixel FITS window against increasing tile counts."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import tempfile


PROBE = """
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let width: usize = std::env::args().nth(2).unwrap().parse().unwrap();
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let window = seiza_fits::read_window(
            std::path::Path::new(&path), width - 1, 0, 1, 1,
        ).unwrap();
        assert_eq!(window.values, [42.0]);
        println!("{}", start.elapsed().as_nanos());
    }
}
"""


def header(cards):
    raw = "".join(f"{card:<80}" for card in cards + ["END"]).encode("ascii")
    return raw.ljust((len(raw) + 2879) // 2880 * 2880, b" ")


def fixture(path, tiles):
    compressed = gzip.compress(struct.pack(">f", 42.0))

    def card(key, value):
        return f"{key:<8}= {value:>20}"

    primary = header([card("SIMPLE", "T"), card("BITPIX", 8), card("NAXIS", 0)])
    table = header([
        "XTENSION= 'BINTABLE'", card("BITPIX", 8), card("NAXIS", 2),
        card("NAXIS1", 8), card("NAXIS2", tiles), card("PCOUNT", len(compressed)),
        card("GCOUNT", 1), card("TFIELDS", 1), "TTYPE1  = 'COMPRESSED_DATA'",
        "TFORM1  = '1PB'", card("ZIMAGE", "T"), card("ZBITPIX", -32),
        card("ZNAXIS", 2), card("ZNAXIS1", tiles), card("ZNAXIS2", 1),
        card("ZTILE1", 1), card("ZTILE2", 1), "ZCMPTYPE= 'GZIP_2'",
    ])
    with path.open("wb") as stream:
        stream.write(primary + table)
        # Every tile references the same valid four-byte sample in the heap.
        stream.write(struct.pack(">ii", len(compressed), 0) * tiles)
        stream.write(compressed)
        stream.truncate((stream.tell() + 2879) // 2880 * 2880)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    results = {}
    with tempfile.TemporaryDirectory(prefix="seiza-tile-probe-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        (work / "src/main.rs").write_text(PROBE)
        (work / "Cargo.toml").write_text(
            '[package]\nname = "seiza-tile-probe"\nversion = "0.1.0"\n'
            'edition = "2024"\n[dependencies]\nseiza-fits = { path = '
            + json.dumps(str(root / "seiza-fits")) + " }\n"
        )
        subprocess.run([
            "cargo", "build", "--release", "--offline", "--manifest-path",
            str(work / "Cargo.toml"), "--target-dir", str(work / "target"),
        ], check=True)
        for count in [1000, 100_000, 1_000_000]:
            path = work / f"tiles-{count}.fits"
            fixture(path, count)
            run = subprocess.run([
                str(work / "target/release/seiza-tile-probe"), str(path), str(count),
            ], check=True, text=True, capture_output=True)
            samples = list(map(int, run.stdout.split()))
            results[count] = {"nanoseconds": samples, "median_ns": statistics.median(samples)}
    report = {
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "platform": platform.platform(),
        "measurements": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
