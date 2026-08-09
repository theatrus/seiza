# seiza-xisf

Practical XISF 1.0 image reading and writing for astrophotography, built on
Seiza's shared decoded astronomy-image representation.

The reader supports monolithic files with attached two-dimensional grayscale
or RGB images, planar pixel storage, little- or big-endian UInt8, UInt16,
UInt32, Float32, and Float64 samples, and zlib, LZ4/LZ4HC, or zstd compression
with optional byte shuffling. FITS compatibility keywords and 2x2 Bayer color
filter arrays are exposed through the same APIs used by `seiza-fits`. SHA-1,
SHA-256, and SHA-512 data-block checksums are verified before decoding. Common
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
Distributed XISF units, inline or embedded image blocks, compression subblocks,
complex samples, CIELab, and dimensions other than two are rejected explicitly.

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

## License

Apache-2.0
