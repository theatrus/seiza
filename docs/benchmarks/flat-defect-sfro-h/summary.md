# H-flat spatial suppression evaluation

Run status: complete; 80 configuration records.

No bias or dark frames were used. These are uncalibrated normalized response comparisons.

| Flats | Configuration | Spatial σ | Selections | Held-out RMS response difference | Master corrections | Integration + mask (s) |
|---|---|---|---|---|---|---|
| 5 | A | — | 3 | 0.00607759 ± 5.86e-05 | 0 ± 0 | 5.27054 ± 4.81 |
| 5 | B | — | 3 | 0.00607759 ± 5.86e-05 | 0 ± 0 | 5.03958 ± 1.52 |
| 5 | C | — | 3 | 0.00607005 ± 5.85e-05 | 18971.7 ± 35.5 | 6.06707 ± 3.92 |
| 5 | D | — | 3 | 0.00607005 ± 5.85e-05 | 18971.7 ± 35.5 | 5.92164 ± 0.925 |
| 5 | E | 6.0 | 3 | 0.00607023 ± 5.86e-05 | 0 ± 0 | 2.69279 ± 0.309 |
| 5 | E | 10.0 | 3 | 0.00607161 ± 5.86e-05 | 0 ± 0 | 2.39032 ± 0.219 |
| 5 | F | 6.0 | 3 | 0.00606868 ± 5.85e-05 | 18935 ± 41.1 | 6.18851 ± 0.705 |
| 5 | F | 10.0 | 3 | 0.00607001 ± 5.85e-05 | 18948 ± 34.7 | 6.37266 ± 0.811 |
| 10 | A | — | 3 | 0.00459539 ± 9.03e-06 | 0 ± 0 | 5.12902 ± 0.144 |
| 10 | B | — | 3 | 0.00459539 ± 9.03e-06 | 0 ± 0 | 8.04929 ± 0.0164 |
| 10 | C | — | 3 | 0.0045879 ± 8.87e-06 | 18091.3 ± 206 | 6.34462 ± 0.0736 |
| 10 | D | — | 3 | 0.0045879 ± 8.87e-06 | 18091.3 ± 206 | 9.60139 ± 0.356 |
| 10 | E | 6.0 | 3 | 0.00458648 ± 8.88e-06 | 0 ± 0 | 5.21519 ± 0.266 |
| 10 | E | 10.0 | 3 | 0.00458742 ± 8.93e-06 | 0 ± 0 | 4.46882 ± 0.369 |
| 10 | F | 6.0 | 3 | 0.00458699 ± 8.81e-06 | 18040.3 ± 211 | 9.21884 ± 0.0661 |
| 10 | F | 10.0 | 3 | 0.00458788 ± 8.86e-06 | 18064.3 ± 204 | 9.39644 ± 0.136 |
| 20 | A | — | 3 | 0.00460726 ± 9.48e-06 | 0 ± 0 | 10.7827 ± 0.304 |
| 20 | B | — | 3 | 0.00460726 ± 9.48e-06 | 0 ± 0 | 17.3138 ± 1.38 |
| 20 | C | — | 3 | 0.00460062 ± 9.44e-06 | 17628.3 ± 102 | 11.8458 ± 0.217 |
| 20 | D | — | 3 | 0.00460062 ± 9.44e-06 | 17628.3 ± 102 | 18.398 ± 0.821 |
| 20 | E | 6.0 | 3 | 0.00459874 ± 9.48e-06 | 0 ± 0 | 9.71306 ± 0.79 |
| 20 | E | 10.0 | 3 | 0.00459921 ± 9.48e-06 | 0 ± 0 | 9.25058 ± 0.202 |
| 20 | F | 6.0 | 3 | 0.00460016 ± 9.48e-06 | 17569.7 ± 135 | 17.8272 ± 0.145 |
| 20 | F | 10.0 | 3 | 0.0046006 ± 9.44e-06 | 17602 ± 101 | 17.5722 ± 0.201 |
| 30 | A | — | 1 | unavailable | 0 | 16.5459 |
| 30 | B | — | 1 | unavailable | 0 | 24.2196 |
| 30 | C | — | 1 | unavailable | 17561 | 18.4652 |
| 30 | D | — | 1 | unavailable | 17561 | 27.9546 |
| 30 | E | 6.0 | 1 | unavailable | 0 | 14.5197 |
| 30 | E | 10.0 | 1 | unavailable | 0 | 15.2352 |
| 30 | F | 6.0 | 1 | unavailable | 17506 | 26.7588 |
| 30 | F | 10.0 | 1 | unavailable | 17538 | 27.5401 |

Spread is the sample standard deviation across three overlapping deterministic selections; it is not a confidence interval. Response is dimensionless (a normalized flat is near 1). Each held-out reference uses the complementary original frames with star masking and master suppression. It is disjoint, but small and from the same session, so it is not ground truth. Thirty-frame candidates have no independent reference. Compare configurations within each input count; the reference sample size changes across counts.

A: temporal rejection; B: A + star masking; C: A + master suppression; D: B + master suppression; E: spatial preprocessing + A; F: spatial preprocessing + D.

Timing excludes dataset auditing, held-out reference construction, residual measurement, and final master writing. Spatial preprocessing includes original reads, the filter, diagnostic mask writing, and temporary float FITS writing. The JSON records preprocessing time separately. Peak memory in run.json is the process-wide Linux VmHWM, not a separate measurement for each configuration.

**Decision: do not add per-frame spatial replacement to production on this evidence.** An independent calibrated reference, dark/bias calibrated light comparisons, and multiple capture sessions are missing. A lower flat-to-flat RMS alone cannot establish preservation of genuine pixel response.

## Flat-only light check

No dark or bias subtraction. These differences are relative to D, not errors against truth. The robust adjacent-difference statistic samples the scene, including any stars and galaxy structure.

| Master | Lights | RMS difference from D (ADU) | Sampled MAD noise σ (ADU) |
|---|---|---|---|
| A-sigma0-n30-seed0 | 3 | 11.1681 ± 0.000207 | 17.4555 ± 4.56 |
| B-sigma0-n30-seed0 | 3 | 11.1681 ± 0.000207 | 17.4555 ± 4.56 |
| C-sigma0-n30-seed0 | 3 | 0.00052719 ± 3.61e-05 | 17.4553 ± 4.56 |
| D-sigma0-n30-seed0 | 3 | 0 ± 0 | 17.4553 ± 4.56 |
| E-sigma10-n30-seed0 | 3 | 2.12333 ± 0.000414 | 17.4555 ± 4.56 |
| E-sigma6-n30-seed0 | 3 | 1.3884 ± 0.000513 | 17.4552 ± 4.56 |
| F-sigma10-n30-seed0 | 3 | 0.669328 ± 4.23e-06 | 17.4553 ± 4.56 |
| F-sigma6-n30-seed0 | 3 | 1.29903 ± 9.07e-05 | 17.455 ± 4.56 |
## Spatial correction persistence

Repeated corrections can be persistent artifacts or real sensor response; this is not a classification.

| Spatial σ | Available exposures | Unique coordinates | Corrected in every available exposure | Corrected once |
|---|---|---|---|---|
| 6 | 30 | 191286 | 27 | 190645 |
| 10 | 30 | 2197 | 17 | 2157 |