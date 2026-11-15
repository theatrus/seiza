# Mosaic matching: PR 477 real-data validation

Tested revision `5103dd144b678472ee972eed596a0d7644a6eb7b` using a newly
built release CLI and the original author's frame lists, calibration masters,
measurement scripts, and saved `before` FITS outputs. The baseline is the
author's main comparison, not an independently rebuilt current `origin/main`.

All 20 new stacks completed successfully: ten datasets/filter combinations,
each with matching disabled and enabled. Every run used
`--reference expand --reintegrate`; enabled runs added
`--match-mosaic-backgrounds`. IC 1318 and Elephant's Trunk also used
`--max-canvas-megasamples 1000`. The calibrated IC 1318 run used the supplied
NIGHT_2 bias and H-alpha flat for every panel and night.

Every disabled output is sample-for-sample identical to its saved baseline,
including NaNs. There were no changes in dimensions, accepted frames, or
rejected frames. IC 1318 accepted 83/83, M31 accepted 121/121, Elephant's Trunk
accepted 127/134 (the identical seven Panel 3 rejection messages), and each
LMC filter accepted 10/10. All source datasets were read-only.

## Enabled results

Values are baseline -> enabled, in ADU. Sky spread is the author's robust
sigma of 128-pixel cell medians over the baseline's dim half. Mottle is the
robust sigma after a 15-cell median filter of 32-pixel cell medians, using
each variant's dim 60 percent. RGB entries are in R/G/B order.

| Dataset | Sky spread | Mottle |
| --- | --- | --- |
| IC 1318 calibrated | 10.44 -> 8.45 | 4.82 -> 4.77 |
| IC 1318 raw | 10.44 -> 11.28 | 4.62 -> 4.64 |
| M31 S50 | 13.46/9.25/11.88 -> 17.42/16.32/22.41 | 7.64/4.26/7.03 -> 7.35/4.55/7.93 |
| Elephant's Trunk | 10.12 -> 9.98 | 1.85 -> 1.89 |
| LMC Red | 4.59 -> 4.95 | 1.49 -> 1.50 |
| LMC Green | 7.19 -> 7.66 | 2.21 -> 2.20 |
| LMC Blue | 10.21 -> 11.00 | 2.78 -> 2.77 |
| LMC Ha | 1.06 -> 1.06 | 0.40 -> 0.41 |
| LMC OIII | 1.63 -> 1.85 | 0.48 -> 0.49 |
| LMC SII | 0.68 -> 0.83 | 0.36 -> 0.37 |

Calibrated IC 1318's bottom/columns seam magnitude decreased from 1.66 to
0.50 ADU. Its other three headline seam measurements changed by at most
0.03 ADU. Raw IC 1318 showed no meaningful headline seam benefit.

M31's blue-minus-green spread increased from 3.77 to 10.87 ADU, despite
red-minus-green spread decreasing from 5.92 to 4.18. This is a substantial
colour regression. One green-channel sample at zero-based (x=2535, y=947)
changed from finite to NaN. The enabled runs change sample rejection totals;
output finite masks are not guaranteed invariant. All other datasets kept
their baseline finite masks.

The LMC seam profiles show benefits and trade-offs. Red's selected steps
changed +0.69 -> +0.03 and -0.17 -> -0.37 ADU; Green's changed +0.79 -> +1.21
and -2.43 -> -0.18; Blue's changed -0.38 -> +0.02 and -1.06 -> +0.60;
Ha's changed +0.47 -> +0.12 and +0.02 -> +0.10; OIII's changed +0.34 -> +0.13
and -0.12 -> -0.19; SII's changed +0.38 -> +0.17 and -0.03 -> +0.04. These are heuristic row profiles, not manually annotated
panel boundaries; the original script selects rows using the largest
baseline-to-enabled differences. They do not establish an overall improvement.

## Interpretation and reproducibility

Keep matching opt-in. It helps calibrated IC 1318 and some LMC seams, but
is unsuitable for the supplied raw IC 1318 and uncalibrated M31 inputs.
Calibrated inputs are a prerequisite, not a guarantee of improvement.
Elephant's Trunk shows little benefit. Inspect the actual stack before
adopting a correction; sky spread includes real emission, and these metrics
do not establish absolute photometric accuracy or preservation of all signal.

The local validation bundle is `target/pr477-real-data/`: exact frame lists,
original measurement scripts, baseline and revised binaries, all outputs,
previews, logs, `manifest.json` with complete commands and timings, and
`disabled-comparison.json` with exhaustive pixel comparisons. Baseline FITS
files are hard-linked into the bundle, preserving them independently of the
author's temporary directory. The bundle is ignored by Git and will be lost
if `target` is cleaned; this report retains the conclusions in the repository.

Binary SHA-256 hashes:

- Author's baseline: `7d5ae2107f6b3a363dda168896e68f8cfc5f3d78d71ca3448f3a0c3363c707d8`
- Revised release: `320bce9effa64290ac843352fd64f8ad6dd327b19e1927c7c89ab924936dfa3c`
