# Streaming FITS image I/O follow-up

Status: implemented on `codex/fits-streaming`

## Previous behavior

Before this change, `FitsImage::open` read the complete FITS file into a raw
byte vector and then converted the data unit into the final typed pixel vector.
Header-only `read_header` already streamed cards and stopped at the end of the
header. For a full image open, however, the raw file buffer and typed pixels
overlapped during conversion, increasing peak memory by roughly the image
payload size.

This work was intentionally separate from the object-catalog mmap change. FITS
pixel conversion, scaling, planar color, and Bayer handling have different
correctness and benchmarking concerns from catalog indexing.

## Implementation

Keep the public `FitsImage::open` result unchanged while replacing its internal
whole-file read with this pipeline:

1. Stream and parse the header, retaining the aligned data-unit offset and
   expected payload size.
2. Continue from the aligned data-unit offset and allocate only the final
   typed pixel buffer. Reservation is fallible, and elements are initialized
   only as their raw chunks arrive, so truncated files do not touch a huge
   header-declared allocation.
3. Read bounded 1 MiB chunks, convert FITS big-endian samples
   directly into the destination type, and apply the existing `BSCALE`/`BZERO`
   rules during conversion.
4. Stop after the declared payload without reading FITS block padding or
   trailing HDUs.
5. Preserve the current planar RGB (`NAXIS3`), Bayer, statistics, preview, and
   solver behavior above the reader.

A raw mmap does not remove the final decoded buffer when endian conversion or
scaling is required. Direct chunked conversion is therefore the useful first
step; an optional raw mapped view can be considered later for callers that can
consume unscaled native data.

The header, bounds checks, truncation behavior, and chunk reader are shared
across pixel types. Only the conversion kernel varies for `BITPIX=8`, `16`,
`32`, `-32`, and `-64`. The existing public pixel representations remain
`U8`, `U16`, `I32`, `F32`, and `F64`.

## Release profiling

Profiles used the optimized `io_profile` example in separate processes on
local disk files. Peak RSS and CPU counters came from macOS `/usr/bin/time -l`.
The first table reports medians of five warm-cache, single-decode processes.
Files under `/Volumes/astrobin` were excluded from performance comparisons and
used only for compatibility checks.

| Local file | Strategy | Load | Peak RSS |
| --- | --- | ---: | ---: |
| 122 MB, 9576x6388 mono | whole-file buffer | 25.8 ms | 247.6 MB |
|  | mmap + owned decode | 19.4 ms | 247.5 MB |
|  | 1 MiB stream + owned decode | 16.5 ms | 125.2 MB |
| 16 MB, 3840x2160 RGGB | whole-file buffer | 3.8 ms | 36.1 MB |
|  | mmap + owned decode | 2.8 ms | 36.0 MB |
|  | 1 MiB stream + owned decode | 2.4 ms | 19.5 MB |
| 4 MB, 1920x1080 | whole-file buffer | 1.0 ms | 11.2 MB |
|  | mmap + owned decode | 0.8 ms | 11.1 MB |
|  | 1 MiB stream + owned decode | 0.7 ms | 7.0 MB |

A separate 50-iteration, load-only run made process CPU time large enough to
measure without including statistics or checksum work:

| Strategy | Average wall load | Average CPU per load | Peak RSS |
| --- | ---: | ---: | ---: |
| whole-file buffer | 10.8 ms | 10.4 ms | 247.8 MB |
| mmap + owned decode | 9.3 ms | 9.2 ms | 247.8 MB |
| 1 MiB stream + owned decode | 7.5 ms | 7.4 ms | 125.4 MB |

The large local frame used about 49% less peak RSS when streamed. mmap did not
reduce peak RSS for a complete decode because the conversion touches the mapped
raw pages while the final owned vector is live. A lazy raw mmap view could make
open nearly allocation-free, but would change the public pixel API and defer
endian/scaling work to every downstream consumer.

Chunk-size tuning on the 122 MB local frame favored 1 MiB:

| Chunk size | Load | Peak RSS |
| ---: | ---: | ---: |
| 64 KiB | 16.8 ms | 124.5 MB |
| 1 MiB | 16.5 ms | 125.2 MB |
| 8 MiB | 17.7 ms | 132.8 MB |

## Validation

- Unit coverage exercises truncated headers/data, non-2880-byte payload
  endings, every supported integer and floating `BITPIX`, scaled 16-bit data,
  planar RGB, and Bayer input.
- Typed-pixel checksums match between buffered, mmap, and streaming decode for
  all three local test files, including the RGGB OSC frame.
- Buffered and streaming checksums also match for representative frames from
  Radian61, Ultracat, C925, Askar107PHQ, and SpaceCat61.
- Header-only reads stop at the end of the header; full-image reads stop at the
  end of the declared payload and do not touch padding or trailing HDUs.
