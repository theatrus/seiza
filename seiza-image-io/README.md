# seiza-image-io

Open and save images by their file extension, the same way across Seiza.

- `open` reads PNG, JPEG and TIFF (upright by their EXIF orientation) and
  FITS and XISF, as floating-point samples. Integer samples are scaled to 0
  to 1 by their type's range, so an 8-bit PNG and a 16-bit FITS read alike;
  floating-point samples are kept as stored.
- `save` writes PNG and TIFF at 16 bits, and FITS and XISF as 32-bit
  floats. FITS files say their rows run top-down (`ROWORDER`), and XISF
  files declare the 0 to 1 range when the samples lie in it.
- `open_astronomy` reads a FITS or XISF file as Seiza's `FitsImage`, for
  callers that need its headers or stored sample type.
- `read_window` reads a rectangle of a FITS or XISF image's samples, in
  physical units, without decoding the rest of the file: the values under a
  viewer's cursor, at once on an image of any size. Its
  `normalized_full_scale` puts an XISF image declared as `0:1` on a camera's
  scale, as stacking does.
- `read_solve_prior` reads what a FITS or XISF file says about where it
  points (`seiza_core::prior::SolvePrior`), without decoding its pixels.
- `raster` opens JPEG, PNG and TIFF files, or uploaded bytes, upright by
  their EXIF orientation, so every reader shares one pixel frame.
  `PhotoMetadata` reads the capture time, GPS position and 35 mm-equivalent
  focal length, and `ScaleSearch` turns the focal length into pixel-scale
  ranges for a blind solve.

This crate does not depend on `seiza`, the solver, so `seiza` releases do
not force releases of it.

```rust
let image = seiza_image_io::open(std::path::Path::new("dust.xisf"))?;
seiza_image_io::save(std::path::Path::new("dust.tif"), &image)?;
# Ok::<(), seiza_image_io::Error>(())
```
