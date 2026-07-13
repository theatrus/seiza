# seiza-fits

Fast, dependency-free FITS image reading for astrophotography, part of the
[seiza](https://crates.io/crates/seiza) family.

- Reads 8/16/32-bit integer and 32/64-bit float FITS images, applying
  BZERO/BSCALE (the common `BITPIX 16` + `BZERO 32768` unsigned camera
  layout is folded directly into the big-endian decode).
- Typed header access (logicals, integers, floats, strings, FORTRAN `D`
  exponents, quote escapes).
- Exact median and MAD statistics via a single histogram pass
  (O(n + 65536), no sort).
- Midtone-transfer-function autostretch matching N.I.N.A. / PixInsight STF
  behavior, rendered through a 65536-entry LUT straight to 8-bit.

A 26-megapixel camera sub loads in ~75 ms, statistics in ~8 ms, and
stretches to display range in ~25 ms on desktop hardware.

```rust
let fits = seiza_fits::FitsImage::open(Path::new("light.fits"))?;
let ra = fits.header_f64("RA");
let stats = fits.statistics();
let display = fits.stretch_to_u8(&Default::default());
```

## License

Apache-2.0
