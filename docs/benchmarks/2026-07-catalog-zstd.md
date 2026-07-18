# Catalog zstd transport benchmark

Measured on the July 2026 v4 catalog artifacts on local disk with zstd 1.5.7
at maximum level 22. These are transport sizes only: clients unpack into the
unchanged mmap-ready files on disk.

| Artifact | Uncompressed | Zstd level 22 | Saved |
| --- | ---: | ---: | ---: |
| `objects.bin` | 241,268,328 | 35,374,379 | 85.3% |
| `stars-lite-tycho2.bin` | 25,473,888 | 21,850,220 | 14.2% |
| `minor-bodies.bin` | 16,451,069 | 7,417,091 | 54.9% |
| `transients.bin` | 2,034,379 | 408,326 | 79.9% |
| `stars-lite-tycho2.ids.bin` | 100,435,815 | 38,887,683 | 61.3% |
| `stars-gaia.bin` | 367,311,952 | 298,423,494 | 18.8% |
| `stars-deep-gaia17.bin` | 1,541,288,228 | 1,177,579,832 | 23.6% |
| `blind-gaia16.idx` | 1,633,535,564 | 1,320,304,795 | 19.2% |
| **Complete bundle** | **3,927,799,223** | **2,900,245,820** | **26.2%** |

The setup presets benefit differently because object and identity metadata
compress much more than already-dense star positions and blind patterns:

| Preset | Uncompressed | Zstd level 22 | Saved |
| --- | ---: | ---: | ---: |
| Solver lite | 285,227,664 | 65,050,016 | 77.2% |
| Solver Gaia | 627,065,728 | 341,623,290 | 45.5% |
| Blind deep | 3,434,577,568 | 2,541,084,423 | 26.0% |
| Complete | 3,927,799,223 | 2,900,245,820 | 26.2% |

Sequential validation/decompression of all eight encoded artifacts took 3.74
seconds on the development Mac. Maximum compression is intentionally a
publication-time cost: the eight CLI encodes took about nine minutes in total,
while the release publisher encoded `objects.bin` in 80.9 seconds. Its frame
was 35,373,879 bytes and decoded byte-for-byte to the original SHA-256; the
500-byte difference from the table's direct zstd CLI run is due to frame
settings and does not materially change the ratio.

Level 3 produced a 3,036,954,261-byte complete bundle (22.7% saved). Level 22
therefore avoids another 136,708,441 transfer bytes per complete installation,
which justifies the occasional additional publication CPU time.
