# seiza-xisf

Practical XISF 1.0 image reading for astrophotography, built on Seiza's shared
decoded astronomy-image representation.

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

## License

Apache-2.0
