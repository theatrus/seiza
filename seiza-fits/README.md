# seiza-fits

Fast, dependency-free FITS image reading for astrophotography, part of the
[seiza](https://crates.io/crates/seiza) family.

- Reads 8/16/32-bit integer and 32/64-bit float FITS images, applying
  BZERO/BSCALE (the common `BITPIX 16` + `BZERO 32768` unsigned camera
  layout is folded directly into the big-endian decode).
- Streams full-image opens directly into the final typed pixel vector using a
  fixed 1 MiB conversion buffer; it does not retain a second whole-file copy.
- Planar RGB (`NAXIS3`) support and OSC debayering from the `BAYERPAT`
  header.
- Typed header access (logicals, integers, floats, strings, FORTRAN `D`
  exponents, quote escapes).
- Exact median and MAD statistics via a single histogram pass
  (O(n + 65536), no sort).
- Midtone-transfer-function autostretch matching N.I.N.A. / PixInsight STF
  behavior, rendered through a 65536-entry LUT straight to 8-bit.

A 26-megapixel camera sub loads in ~75 ms, statistics in ~8 ms, and
stretches to display range in ~25 ms on desktop hardware.

`examples/io_profile.rs` compares buffered, mmap-plus-owned-decode, and
streamed complete decodes in separate release processes. On a local 122 MB,
61-megapixel `BITPIX=16` frame, streaming reduced median peak RSS from
247.6 MB to 125.2 MB and improved median load time from 25.8 ms to 16.5 ms.

```rust
let fits = seiza_fits::FitsImage::open(Path::new("light.fits"))?;
let ra = fits.header_f64("RA");
let stats = fits.statistics();
let display = fits.stretch_to_u8(&Default::default());
```

## License

Apache-2.0
