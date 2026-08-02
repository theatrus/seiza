# seiza-background

`seiza-background` fits smooth, deterministic background models to linear
astronomical images. It is format-independent: callers provide interleaved
`f32` samples and receive a compact fitted model that can correct an image in
place or render the estimated background on demand.

The crate provides robust weighted polynomial surfaces and smoothed thin-plate
radial-basis surfaces. Automatic mode scores constant through a configured
polynomial degree on held-out background samples. It only accepts a more
flexible model when the normalized validation error improves by a set margin.
Callers can add the radial-basis candidate, but must do so explicitly because
held-out samples can share the same real extended emission.

Candidate windows are spread across the frame, moved toward nearby quiet
locations, and filtered with local-dispersion and iterative residual rejection.
Mono and RGB images share sample positions while fitting each channel
independently. An exclusion mask can protect pixel-level structures. Normalized
ellipse and polygon regions can protect solver-projected catalog bounds without
allocating a full image mask; projected OpenNGC contours can be passed as one
polygon per connected outline.

Both additive subtraction and multiplicative division preserve the robust
background reference level. Correction strength can range from zero to one.
Invalid input pixels remain invalid. Model fitting uses only a bounded number
of small windows, and correction needs only the caller's input/output buffer; a
full-size background image is allocated only when requested.

This is a conventional, non-ML baseline. The model enum and fit/correction
split leave room for other estimators without changing the surrounding
pipeline.
