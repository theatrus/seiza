# seiza-imgproc

Pure-Rust image processing primitives with OpenCV-compatible semantics, for
star-detection and astrophotography pipelines that must not carry a native
OpenCV dependency:

- Separable Gaussian blur: the bit-exact `CV_8U` fixed-point path (OpenCV's
  cumulative-sum Q8 kernel quantization) and the bit-exact `CV_32F` path
  (per-kernel-size FMA accumulation orders with unfused scalar tails,
  runtime FMA dispatch on x86-64).
- 3x3 median blur.
- Canny edge detection (Sobel aperture 3, L1 gradient, TG22 fixed-point
  non-maximum suppression, stack hysteresis).
- Otsu thresholding.
- Binary erosion/dilation with rectangular, elliptical and cross
  structuring elements.
- External contour extraction (Suzuki-Abe border following, as in
  `findContours` with `RETR_EXTERNAL`) plus contour area, arc length,
  convex hull, moments and bounding rectangles computed the way OpenCV
  computes them.
- The Gastal-Oliveira domain transform filter (`DTF_NC`), including
  OpenCV's f32 prefix-sum box accumulation.
- À trous B3-spline wavelet structure removal.

Every operation was verified against OpenCV 4.13 — the Gaussian, Canny,
Otsu, morphology, contour and domain-transform paths bit-for-bit — and the
golden tests in `tests/golden.rs` lock that behavior with platform-stable
fixtures. All functions operate on plain row-major slices; there is no
image type to construct and no dependencies.
