# Streaming FITS image I/O follow-up

Status: planned as a separate pull request

## Current behavior

`FitsImage::open` currently reads the complete FITS file into a raw byte vector
and then converts the data unit into the final typed pixel vector. Header-only
`read_header` already streams cards and stops at the end of the header. For a
full image open, however, the raw file buffer and typed pixels overlap during
conversion, increasing peak memory by roughly the image payload size.

This work is intentionally separate from the object-catalog mmap change. FITS
pixel conversion, scaling, planar color, and Bayer handling have different
correctness and benchmarking concerns from catalog indexing.

## Proposed implementation

Keep the public `FitsImage::open` result unchanged while replacing its internal
whole-file read with this pipeline:

1. Stream and parse the header, retaining the aligned data-unit offset and
   expected payload size.
2. Seek to the data unit and allocate only the final typed pixel buffer.
3. Read bounded row or multi-row chunks, convert FITS big-endian samples
   directly into the destination type, and apply the existing `BSCALE`/`BZERO`
   rules during conversion.
4. Skip or validate FITS block padding without retaining it.
5. Preserve the current planar RGB (`NAXIS3`), Bayer, statistics, preview, and
   solver behavior above the reader.

A raw mmap does not remove the final decoded buffer when endian conversion or
scaling is required. Direct chunked conversion is therefore the useful first
step; an optional raw mapped view can be considered later for callers that can
consume unscaled native data.

## Acceptance criteria

- Existing FITS unit and CLI tests remain byte-for-byte/metadata compatible.
- Add coverage for truncated headers/data, non-2880-byte payload endings,
  integer and floating `BITPIX`, `BSCALE`/`BZERO`, planar RGB, and Bayer input.
- Benchmark representative mono, OSC, and RGB files in release mode.
- Demonstrate that peak RSS is approximately the final decoded image plus a
  small fixed chunk, rather than raw file plus decoded image.
- Keep header-only reads streaming and ensure no open path performs an
  exhaustive image validation unless pixel decoding was requested.
