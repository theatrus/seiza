# seiza-image-ser

A standalone, seekable reader for planetary SER captures, including those written
by Siril. Registered as a workspace crate, but not connected to `seiza-image-io`,
the stacker, the CLI, C or Python. Existing image dispatch remains unchanged.

```rust
use seiza_image_ser::{Error, SerReader};

fn inspect(path: &std::path::Path) -> Result<(), Error> {
    let mut sequence = SerReader::open(path)?;
    let info = sequence.info().clone();
    for index in 0..info.frame_count {
        let frame = sequence.read_frame(index)?;
        // Pass frame.pixels and info to a future measurement/stacking stage.
        // Each frame is owned and can be retained or dropped independently.
        println!("{}: {:?}, {:?}", frame.index, info.color, frame.timestamp_utc);
    }
    Ok(())
}
```

`SerReader::new` also accepts any `Read + Seek` source, including an in-memory
`Cursor`. Opening checks the complete advertised file length without decoding
frames. Reading an index seeks directly to that frame and its optional timestamp;
memory usage does not grow with capture length. Independent readers can serve
parallel workers; a single reader uses a mutable seek cursor.

## Samples and metadata

- Mono, the four RGB Bayer mosaics, four complementary-color mosaics, RGB and BGR.
- Depths 1–8 use `Pixels::U8`; 9–16 use `Pixels::U16` in native byte order.
  Stored values are preserved, without shifting, scaling or demosaicing.
- Rows remain top-down. RGB/BGR samples are **interleaved**, not FITS planar data.
  `SequenceInfo::color` specifies their order. CFA patterns start at pixel (0, 0).
- Observer, instrument, telescope and header start times remain available through
  `metadata()`. Local time and UTC are separate. Per-frame timestamps preserve
  100 ns ticks; absent or zero dates are `None`, never estimated from a frame rate.
- Either no trailer or one timestamp per frame is accepted. Partial trailers,
  extra trailing bytes and truncated frames are rejected. A complete trailer is
  read even when the header's local date is zero. Empty sequences are allowed.
- Checked offset arithmetic supports captures above 4 GiB. `ReadOptions` limits
  sample storage to 512 MiB per frame by default; applications can adjust it.
  Decoding 16-bit samples uses an additional fixed 8 KiB scratch buffer.

The default endian convention follows
[Siril](https://siril.readthedocs.io/en/latest/file-formats/SER.html): pixel flag
0 means little endian and 1 means big endian. This reverses the
[published SER specification](https://grischa-hahn.hier-im-netz.de/astro/ser/SER%20Doc%20V3b.pdf).
For a writer that follows the specification literally, use
`ReadOptions { endian_convention: EndianConvention::Specification, ..Default::default() }`
with `SerReader::open_with_options` or `SerReader::with_options`. Header numbers
and timestamps are always little endian. Byte order is never guessed from pixel
brightness.

## Future sequence inputs

`ImageSequence` exposes fixed `SequenceInfo` and indexed `read_frame` access with
a reader-specific error type. A future stacker can select frames and make
multiple passes through that interface without extracting files or collecting
the whole sequence. It can borrow a trait object or be generic over the source.
The format reader preserves acquisition data; calibration, color conversion,
normalization and stacking belong to downstream stages.

**AVI is not decoded by this crate.** A raw AVI reader can implement the same
interface, but must explicitly handle supported codecs, row stride/orientation,
color layouts and timing. Relative AVI presentation times are not UTC dates and
must not be manufactured as SER timestamps. Additional timing metadata can be
introduced when that reader is added. The generic interface currently lives here
with its first implementation; it can move to a shared sequence crate then.

When sequences are integrated into Seiza, file routing should live in
`seiza-image-io`, with C and Python exposing the same native capabilities.

## Validation

`cargo test -p seiza-image-ser` exercises independently constructed binary fixtures
covering every supported color ID and bit depth, both endian conventions,
observation dates, complete and malformed trailers, indexed reads, bounded reads,
large offsets and allocation/overflow refusals. These are byte-level fixtures,
not a bundled real-world Siril capture.
