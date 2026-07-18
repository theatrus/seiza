# SIP distortion fitting on real fields

Validation of the SIP polynomial output (July 2026) on real, heavily
processed astrophotography JPEGs — narrowband and RGB composites exported
for AstroBin, solved with no position hints and no FITS headers.

## Method

Each image was blind-solved with the 25 MB Tycho-2 catalog to recover its
center and pixel scale, then solved twice hinted against the Gaia G≤15
catalog — once linear, once with `--sip-order 3`:

```
seiza solve-blind IMG.jpg --data stars-lite-tycho2.bin --min-scale 0.5 --max-scale 25
seiza solve IMG.jpg --data stars-gaia.bin --ra RA --dec DEC --scale S --radius 0.5 --sip-order 0
seiza solve IMG.jpg --data stars-gaia.bin --ra RA --dec DEC --scale S --radius 0.5 --sip-order 3
```

All seven images blind-solved. SIP was accepted by the
degrees-of-freedom-corrected guard on every field:

| Image | Linear RMS | SIP-3 RMS | Reduction | Matched stars |
| --- | ---: | ---: | ---: | --- |
| Sh2-119 (wide field) | 4.62" | 2.99" | 35% | 89 → 89 |
| NGC 7000 HOO (wide field) | 4.30" | 2.84" | 34% | 105 → 110 |
| M31 composite | 2.57" | 1.35" | 47% | 100 → 100 |
| WR 134 SHO | 2.72" | 2.28" | 16% | 108 → 108 |
| Elephant's Trunk | 1.61" | 1.29" | 20% | 79 → 79 |
| Triangulum | 0.82" | 0.79" | 4% | 97 → 97 |
| Crescent | 0.41" | 0.37" | 8% | 91 → 91 |

## Observations

- The improvement concentrates where distortion physically lives: the
  wide-field, lens-dominated images improve 34-47%, while well-corrected
  narrow fields already at sub-arcsecond linear residuals improve 4-8%.
  That is the signature of fitting real field curvature rather than noise.
- NGC 7000 matched five additional stars with SIP enabled: corner stars
  whose distortion exceeded the linear matching tolerance are recovered by
  the fit/re-match/re-fit loop.
- The acceptance guard operated at its intended boundary. Triangulum's 4%
  improvement corresponds to a residual sum-of-squares ratio of ~0.925
  against a degrees-of-freedom threshold of ~0.924 for its ~97 stars and
  order-3 parameter count — the weakest genuine improvement in the set
  barely cleared the bar, and anything weaker keeps the linear solution.

## Header validity

Independently of these solves, the emitted FITS keywords are
cross-validated against wcslib (via astropy, the same library Siril uses)
in `seiza-py/tests/test_astropy_crossval.py`: forward transforms rebuilt
purely from the emitted header agree with seiza's own to ~1e-10 arcsec,
and the emitted AP/BP inverse agrees with wcslib's exact iterative
inverse to ~0.001 px.

These are single-machine measurements on one image set; ratios depend on
the optics. The commands above reproduce them on any directory of images.
