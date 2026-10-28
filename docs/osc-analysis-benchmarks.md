# OSC analysis validation and timings

These are deterministic **synthetic fixtures**, not camera datasets. The example
uses 64 Gaussian stars, independent SplitMix64/Box–Muller noise, known component
backgrounds and noise scales, and six 60-second frames per representation.
The CFA and RGB fixtures represent the same underlying scene, but RGB is
synthetic native color data, not a demosaiced version of the CFA files.

Measured on Windows with an AMD Ryzen 9 5950X, release build and
`RAYON_NUM_THREADS=8`. Each source is 800 × 800 pixels. Raw CFA yields four
400 × 400 component lattices; RGB yields three 800 × 800 planes. Analysis uses
32,768 requested samples per component (100 available patches for CFA; 128 for
RGB). One warm-up per representation precedes three alternating repetitions.
No concurrent builds or test suites ran during timing. Timings include source
reads and analysis, but exclude fixture generation and JSON/HTML writing.

| Input | Source files | Component records | Median analysis | Range |
| --- | ---: | ---: | ---: | ---: |
| Raw CFA | 6 | 24 | 6.247 s | 6.246–6.319 s |
| RGB | 6 | 18 | 7.274 s | 7.269–7.337 s |

All groups included all six inputs. The CFA red split-noise measurement fell
from 4.227 at two frames to 2.426 at six; the expected values for the planted
sigma of 6 are 4.243 and 2.449. The other components also retain their distinct
noise scales. These small cases demonstrate functionality and repeatability;
they do **not** predict the cost of large camera files or NAS storage.

Reproduce with a new output directory:

```sh
cargo run --release -p seiza-analysis --example osc_preview -- NEW_DIRECTORY
```

Set `RAYON_NUM_THREADS=8` to match the measured run. The example writes source
FITS files via `seiza-image-io`, per-layout JSON/HTML reports and `timings.json`.
It refuses an existing output directory. [Recorded timings](benchmarks/osc-analysis.json).

Scientific tests additionally cover all four Bayer patterns, both row orders,
origin offsets, odd dimensions, exact native sample preservation, malformed
CFA metadata, RGB separation, mixed FITS/XISF inputs, independently normalized
component curves and original-image pixel units. Mono regressions remain covered.
Browser verification covers component selection, six points per component,
three balanced depth points, source/component counts, exposure units and
both standalone and enhanced post-stack rendering. An actual six-frame CFA
stack was reintegrated and rendered with its four source-analysis components.
