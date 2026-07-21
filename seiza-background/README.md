# seiza-background

`seiza-background` fits smooth, deterministic background models to linear
astronomical images. It is format-independent: callers provide interleaved
`f32` samples and receive a compact fitted model that can correct an image in
place or render the estimated background on demand.

The first model is a robust weighted polynomial surface. Candidate windows are
distributed across the frame, moved toward nearby locations with the lowest
background level (a window's mean plus a quarter of its dispersion), and
filtered with local-dispersion and iterative residual rejection. Mono and RGB
images share sample positions while fitting each channel independently.

Both additive subtraction and multiplicative division preserve the robust
background reference level. Invalid input pixels remain invalid. Model fitting
uses only a bounded number of small windows, and correction needs only the
caller's input/output buffer; a full-size background image is allocated only
when explicitly requested.

This is a conventional, non-ML baseline. The model enum and fit/correction
split leave room for spline, radial-basis, multiscale, or learned estimators
without changing the surrounding pipeline.
