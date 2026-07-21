# seiza-fits

Fast FITS image reading and linear `f32` writing for astrophotography, part of the
[seiza](https://crates.io/crates/seiza) family.

The companion [`seiza-xisf`](../seiza-xisf/README.md) reader returns this
crate's decoded `FitsImage` representation, so downstream processing can share
one linear-image API without coupling operations to the source container.

- Reads 8/16/32-bit integer and 32/64-bit float FITS images, applying
  BZERO/BSCALE (the common `BITPIX 16` + `BZERO 32768` unsigned camera
  layout is folded directly into the big-endian decode).
- Streams full-image opens directly into the final typed pixel vector using a
  fixed 1 MiB conversion buffer; it does not retain a second whole-file copy.
- Planar RGB (`NAXIS3`) support and OSC debayering from the `BAYERPAT`
  header.
- Typed header access (logicals, integers, floats, strings, FORTRAN `D`
  exponents, quote escapes).
- Writes primary-HDU mono, interleaved RGB, or planar RGB linear `f32` images,
  with typed non-structural headers and FITS block padding.
- Publishes complete files atomically so a failed write cannot replace the
  previous output; stream-oriented callers can use `write_f32_image_to`.
- Exact median and MAD statistics via a single histogram pass
  (O(n + 65536), no sort).
- Midtone-transfer-function autostretch matching N.I.N.A. / PixInsight STF
  behavior, rendered through a 65536-entry LUT straight to 8-bit. Both now
  delegate to the reusable `seiza-stretch` model while this API stays stable.

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

let cards = [seiza_fits::WriteHeaderCard::new(
    "EXPTIME",
    seiza_fits::HeaderValue::Float(30.0),
)];
seiza_fits::write_f32_image(
    Path::new("linear-output.fits"),
    width,
    height,
    seiza_fits::F32ImageData::Mono(&samples),
    &cards,
)?;
```

## License

Apache-2.0
