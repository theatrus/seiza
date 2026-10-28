# Completed-stack report benchmarks

Measured 2026-10-27 on an AMD Ryzen 9 5950X (16 cores / 32 threads), Windows,
with an RTX 2080 SUPER available and default GPU settings. Release build;
`RAYON_NUM_THREADS=8`, two preparation workers, 2048 MiB pipeline/reintegration
budget. Eight Heart Nebula sulfur exposures, each 9576 × 6388 mono (61.17 MP),
978,808,320 total source bytes, staged on local storage. All eight were admitted.

The measured command stacks and reintegrates, writes the same FITS, and writes
JSON plus HTML when enabled. Rich also fingerprints sources/output and runs the
existing source-analysis pass. No concurrent builds or test suites ran during
these measurements. None and cheap warm-up runs preceded three repetitions per
mode, rotating order. Filesystem caches were warm; this is **not** a cold-NAS
benchmark. These results describe this field, resolution, machine and settings,
not a universal overhead percentage.

| Mode | Median total wall time | Range (three runs) |
| --- | ---: | ---: |
| none | 5.88 s | 5.81–6.10 s |
| cheap | 6.80 s | 6.59–6.83 s |
| rich | 198.35 s | 197.56–198.66 s |

Cheap adds about **0.91 s** over the no-report median (roughly 16% on this short,
fast stack). Rich adds about **192.47 s**. All eleven runs, including warm-ups,
produced byte-identical FITS files, verified by SHA-256 after the timed interval.
HTML was about 576 KB for cheap and 584 KB for rich.

Rich analysis stage medians (wall seconds, included in rich total):

| Stage | Median |
| --- | ---: |
| measurement_seconds | 187.419 s |
| read_seconds | 0.255 s |
| registration_seconds | 1.833 s |
| replay_seconds | 0.041 s |
| sampling_seconds | 0.809 s |
| total_seconds | 190.422 s |

Measurement includes conversion, quality star detection/PSF fits, native
background/noise and metadata. It dominates the additional work. Reusing an
already-loaded image could remove the extra reads, but would not eliminate
these measurements. Registration and sampled depth work are comparatively small
in this case. Source-analysis noise is an equal-weight diagnostic; it does not
reproduce final-stack calibration, weighting or rejection. Do not extrapolate
these timings blindly to every filter or to an entire NAS-backed project.

Rich is explicitly opt-in. Cheap does not hash or reread image pixels, run star
detection, or compute source-analysis curves. No-report mode skips report
construction and file output entirely. The CLI provenance envelope is schema 2:
`sha256` is null when not computed; the nested post-stack schema remains 1.

Reproduce with a release CLI and a text file of representative source paths:

```sh
cargo build --release -p seiza-cli --locked
python seiza-analysis/benches/report_modes.py --seiza target/release/seiza --inputs inputs.txt --out new-benchmark-directory --repeats 3
```

On Windows, use `target/release/seiza.exe`. The harness refuses an existing
output directory, preserves generated artifacts, checks FITS equality, and saves
per-run timings. [Raw measurements](benchmarks/report-modes-heart.json) include
the exact benchmark executable hash.
