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
`read_image` returns the declared range beside the pixels, and `rescale_to`
converts onto a chosen full scale:

```rust
let mut read = seiza_xisf::read_image(std::path::Path::new("integration.xisf"))?;
if read.rescale_to(65535.0) {
    println!("normalized frame placed on a 16-bit scale");
}
# Ok::<(), seiza_xisf::XisfError>(())
```

`rescale_to` answers `false` and leaves the pixels alone when the file
declares no bounds, when they are degenerate, or when the samples are
integers — XISF integer formats already span their type's range. `open` is
unaffected either way: it still hands back exactly what is stored.

`write_f32_image` mirrors the `seiza-fits` writer: it atomically writes a
one-image monolithic XISF file with uncompressed little-endian `Float32`
planar samples and FITS-compatible keywords, sharing the `F32ImageData` and
`WriteHeaderCard` types so callers can pick the output format by extension.
Files written this way round-trip through this crate's reader and load in
PixInsight.

## License

Apache-2.0
