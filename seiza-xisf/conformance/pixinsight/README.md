# PixInsight conformance run

Checks `seiza-xisf` against PixInsight's XISF module in both directions:

1. `gen.js` makes PixInsight write a matrix of files with known pixel
   patterns: every sample format it writes, mono and RGB, every compression
   codec with and without byte shuffling, SHA-1, SHA-256 and SHA-512
   checksums, embedded and unaligned blocks, two images in one file, a CIELab
   image, and an image with properties of every scalar, vector and matrix
   type plus FITS keywords.
2. The `pixinsight_probe` example reads each file with seiza, compares the
   samples with the pattern, and writes two kinds of file for PixInsight: a
   round-trip copy of each source that carries its metadata, and files of
   seiza's own with every codec and checksum.
3. `verify.js` makes PixInsight read seiza's files. It compares the pixels
   with the pattern, both as PixInsight loads them by default and with
   `no-normalize`, and compares every property, keyword, thumbnail, color
   filter array and RGB working space of each round-trip copy with its source.

```sh
seiza-xisf/conformance/pixinsight/run.sh [work_dir]
```

It needs a PixInsight install (`PIXINSIGHT`, default `~/PixInsight`) and
`xvfb-run` when no display is available. A run takes about a minute.
