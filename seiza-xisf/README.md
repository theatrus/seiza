# seiza-xisf

Practical XISF 1.0 image reading and writing for astrophotography, built on
Seiza's shared decoded astronomy-image representation.

The reader follows the XISF 1.0 specification, Revision 1 (September 2026).
It reads monolithic files and the header files (`.xish`) of distributed units,
whose blocks may live in local files or XISF data blocks files (`.xisb`). Pixel
blocks may be attached, inline, embedded, or external; stored planar or
normal; little- or big-endian UInt8, UInt16, UInt32, UInt64, Float32, and
Float64 samples; and compressed with zlib, LZ4, LZ4HC, or zstd, with or without
byte shuffling and subblocks. CIELab images convert to RGB through the image's
RGB working space. Complex32 and Complex64 images have their own reader,
`read_complex_image`, because `FitsImage` has no complex type. FITS
compatibility keywords and 2x2 Bayer color filter arrays are exposed through
the same APIs used by `seiza-fits`. SHA-1, SHA-256, SHA-512, SHA3-256, and
SHA3-512 data-block checksums are verified before decoding. Common
XISF object, pointing, acquisition-time, instrument, and observer properties
are also projected into non-destructive FITS-compatible headers for downstream
Seiza workflows.

```rust
let images = seiza_xisf::inspect(std::path::Path::new("integration.xisf"))?;
for image in &images.images {
    println!("{}: {}x{}", image.index, image.width, image.height);
}

let image = seiza_xisf::open(std::path::Path::new("integration.xisf"))?;
let display = image.stretch_to_u8(&Default::default());
# Ok::<(), seiza_xisf::XisfError>(())
```

`open` selects the first top-level image. Use `open_image` or
`open_image_by_id` for rejection maps, crop masks, and other auxiliary images.
This crate cannot decode images with remote data blocks, alpha channels, or
dimensions other than two. As the specification requires, such an image is
unavailable on its own: opening it fails with `Unsupported`, `inspect` lists it
under `unavailable`, and the other images in the file stay readable.

`read_image` also returns an `XisfMetadata` holding everything else the file
says about the image and its unit: every property, keyword and core element,
extension elements in other namespaces, and the unit's `Metadata` properties,
with attached and external blocks loaded as stored. Pass it back through
`WriteOptions::metadata` and the writer carries all of it into the new file.
The writer states the new pixel format and encoding itself, and drops what the
new pixels make false: FITS scaling keywords, the astrometric solution when the
width or height changed, and the color filter array when the channel count
changed. Other changes that make metadata false, such as a registration that
keeps the size, are for the caller to handle, for example with
`remove_properties("AstrometricSolution:")`.

`XisfMetadata::astrometric_solution` reads the Revision 1 `AstrometricSolution`
namespace into typed layers: projection, projective transformation, radial
basis function distortion models, and provenance. It follows the rules for
falling back from unusable layers, but does not evaluate the transformation.

```rust
let read = seiza_xisf::read_image(std::path::Path::new("light.xisf"))?;
if let Some(solution) = read.metadata.astrometric_solution()? {
    println!("{:?} at {:?}", solution.projection.system, solution.projection.reference_celestial);
}
# Ok::<(), seiza_xisf::XisfError>(())
```

The writer produces one Float32 image with the mandatory `XISF:CreationTime`
and `XISF:CreatorApplication` metadata. `write_f32_image` leaves the pixels
uncompressed, so readers built on the original 2017 document can open the
file. `write_f32_image_with_options` adds compression, where
`WriteCompression::recommended()` is the zstd with byte shuffling the
specification recommends, a checksum, and carried metadata. It cannot write
LZ4HC, since `lz4_flex` has no LZ4HC encoder.

Samples decode exactly as stored. That matters for floating-point images,
because PixInsight normalizes them to `bounds="0:1"` and nothing in the
samples says so — such a frame is not comparable with a camera frame's ADU.
`read_image` returns the declared range beside the pixels, and
`rescale_normalized_to` converts such a frame onto a chosen full scale:

```rust
let mut read = seiza_xisf::read_image(std::path::Path::new("integration.xisf"))?;
if read.rescale_normalized_to(65535.0) {
    println!("normalized frame placed on a 16-bit scale");
}
# Ok::<(), seiza_xisf::XisfError>(())
```

Treat `bounds` as a hint rather than a fact. Writers disagree about what the
range means — `write_f32_image` in this crate reports the observed sample
minimum and maximum, not a nominal `0:1` — so only an exact `0:1` carries a
settled meaning, and that is the only range `rescale_normalized_to` acts on.
Converting from anything else would as easily stretch an already-physical
frame as normalize a normalized one. When a caller knows what the samples mean
and the file does not say so usefully, `rescale_from` takes the source range
directly. Both decline integer samples, which already span their format's
range, and both map linearly without clamping, so unclipped highlights and
negative background residuals survive. `open` is unaffected either way: it
still hands back exactly what is stored.

An unusable `bounds` — a spelling this crate cannot read, or a range that does
not increase — reads as `None` rather than failing the file, because nothing
here needs the attribute to decode an image.

`write_f32_image` mirrors the `seiza-fits` writer: it atomically writes a
one-image monolithic XISF file with uncompressed little-endian `Float32`
planar samples and FITS-compatible keywords, sharing the `F32ImageData` and
`WriteHeaderCard` types so callers can pick the output format by extension.
Files written this way round-trip through this crate's reader and load in
PixInsight.

FITS keywords with missing or blank `value` attributes are preserved as
undefined values (`HeaderValue::Raw(String::new())`). Both the FITS and XISF
writers retain them, including blank `FILTER` keywords in calibration masters.
A quoted empty value (`value="''"`) remains an empty string, not an undefined
value.

## License

Apache-2.0
