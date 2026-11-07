# Raster upload allocation budget

A 65-byte PNG header describing 200,000 × 200,000 16-bit RGBA pixels reached
an attempted 320,000,000,000-byte pixel allocation through
`seiza_image_io::raster::decode_oriented`. The image crate's decoder limits
were installed, but constructing a DynamicImage allocated its output buffer
outside that check. Seiza already checked output size explicitly for local
files; the byte/upload reader passed no output-size budget.

The byte reader now supplies the image crate's default allocation budget
(currently 512 MiB) to the same explicit check before pixel construction.
This uses the dependency's default rather than duplicating its policy value.
The existing local-file budget remains 16 GiB.

Measured on Linux x86_64 with rustc 1.99.1 and image 0.25.10.
The pre-fix probe's rejecting allocator observed a 320 GB allocation attempt;
it did not allocate that memory. After the fix, the reader returns the typed
`RasterError::Image(ImageError::Limits(...))` error. The largest allocation
request observed before refusal is 131,072 bytes. This is an allocation-request
measurement, not a peak-RSS or throughput comparison. Raw logs are included.

## Reproduction and regression safety

```sh
cargo test -p seiza-image-io --test raster_decode_budget -- --nocapture
```

The test runs the public byte reader in a child process. Its allocator exits
with status 42 if any single request exceeds 512 MiB, so a regression cannot
allocate the claimed pixel buffer or crash the parent test runner. The
initial pre-fix diagnostic used a rejecting allocator and recorded Rust's
allocation-failure abort in that child; the stable regression uses an exit
to avoid producing core dumps.

A native boundary test also verifies a 16-bit RGBA pixel: a seven-byte pixel
budget refuses it, eight bytes succeeds, and its samples stay intact even
though the encoded PNG itself exceeds eight bytes.

## Output coverage

`uploaded_float_tiff_lights_keep_their_output` round-trips synthetic lights
through the public byte reader as float TIFFs before stacking. It reproduces
the existing MONO_DEFAULT summary on unchanged code. A temporary zero-byte
budget made the case fail, then the mutation was removed. All 76
stacking/stretch/pipeline output checks pass unchanged. No OUTPUT_VERSION or
recorded summary changes are needed.

## Final validation

On the branch based on PR #446's compile-memory fix:

- Full workspace with all features: 2,265 passed, 19 ignored.
- Fresh maturin build and Python runtime suite: 1,637 passed, 21 skipped.
- Image-I/O tests, including the bounded child regression: 32 pass in both
  debug and release builds.
- All 76 stacking/stretch/pipeline output checks pass unchanged.
- Workspace and Python formatting and Clippy, Rust 1.89 workspace
  all-targets/all-features locked checking, and binding parity pass.
