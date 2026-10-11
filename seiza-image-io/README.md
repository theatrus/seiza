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

```rust
let image = seiza_image_io::open(std::path::Path::new("dust.xisf"))?;
seiza_image_io::save(std::path::Path::new("dust.tif"), &image)?;
# Ok::<(), seiza_image_io::Error>(())
```
