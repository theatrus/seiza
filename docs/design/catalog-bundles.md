# Hosted catalog bundles

Hosted paths are compatibility contracts. Once a released reader knows a URL,
the manifest and artifacts at that URL are never replaced with an incompatible
wire format.

## Compatibility layout

| Hosted path | Contents | Readers |
| --- | --- | --- |
| `/data/` | classic files with `SEIZAOB1` objects | v0.3 and classic-v1 clients |
| `/data/v3/` | reserved historical standalone object-v3 rollout URL; may be absent | v0.4.0 falls back to `/data/` |
| `/data/v2/` | frozen complete bundle with `SEIZAOB3` objects | v0.4.1 and v0.5 |
| `/data/v4/` | current complete bundle with sectioned `SEIZAOB\0` objects | v4-capable clients |

The complete bundle generation deliberately skips v3 because that path was
already released for the standalone object-v3 transition. Bundle generations
and individual wire-format versions remain separate concepts even when v4 is
currently shared by both names.

Downloads remain flat locally, so CLI arguments, server configuration, and
environment variables continue to use familiar names such as `objects.bin`.
New readers accept v1, v3, and v4 object files, but old readers never receive a
v4 file from a URL they already know.

## V4 manifest

`/data/v4/manifest.json` describes one complete, coherent catalog set. Every
artifact is addressed by its SHA-256 so an older cached manifest remains valid
after a newer manifest is published:

```json
{
  "version": "catalog-bundle-v4-2026-07-16",
  "files": [
    {
      "name": "objects.bin",
      "key": "artifacts/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef/objects.bin",
      "bytes": 123456789,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "transports": [
    {
      "name": "objects.bin",
      "encoding": "zstd",
      "key": "artifacts/fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210/objects.bin.zst",
      "bytes": 23456789,
      "sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    }
  ]
}
```

For v4, `key` is mandatory and must be exactly
`artifacts/<sha256>/<name>`. `name` is the safe flat filename used in the local
cache and materialized output directory. Legacy `catalog-bundle-v2-*`
manifests remain readable when a caller explicitly configures `/data/v2`; they
omit `key` and resolve files directly under that bundle URL. A
`catalog-bundle-v3-*` complete manifest is intentionally rejected so the
historical `/data/v3/` contract cannot be repurposed accidentally.

`transports` is optional additive metadata. The file-level `key`, `bytes`, and
`sha256` always describe the canonical uncompressed artifact and are retained
for old v4 readers. New readers select a supported alternate, verify its
encoded size and SHA-256, stream-decompress it into the normal cache path, and
then verify the uncompressed size and SHA-256 before installation. They never
mmap compressed bytes and do not retain a second compressed cache file.
Old readers ignore the entire top-level field, preserving both JSON and Rust
source compatibility with the existing public manifest structs. Unknown JSON
fields and unknown encoding names are ignored, so additional
transport encodings can be introduced without another bundle generation. V2
manifests remain frozen and cannot advertise alternate encodings.

See the [catalog zstd benchmark](../benchmarks/2026-07-catalog-zstd.md) for
measured complete-bundle and setup-preset savings.

The complete v4 manifest must contain the Tycho-2 and Gaia solver tiles, blind
index, stellar identifier sidecar, object and transient catalogs, and minor
bodies. The downloader rejects an incomplete manifest even when the caller
requested only one artifact.

## S3 publication contract

1. Build and exhaustively validate every changed catalog.
2. Upload changed uncompressed bytes, or server-side copy unchanged bytes, to
   `/data/v4/artifacts/<sha256>/<name>`.
3. Upload each alternate transport to its own content-addressed key, such as
   `/data/v4/artifacts/<encoded-sha256>/<name>.zst`.
4. Verify every hosted object's byte length and SHA-256, both encoded and
   uncompressed.
5. Publish an immutable copy at
   `/data/v4/manifests/<catalog-bundle-version>.json`.
6. Publish `/data/v4/manifest.json` last as the small current-bundle pointer.
7. Run hosted download, validation, and semantic lookup tests through that
   public manifest.

Artifact keys should use long-lived `immutable` cache headers. The current
manifest pointer should have a short lifetime or require revalidation. No
artifact referenced by a published manifest is overwritten; a changed nightly
transient catalog receives a new hash key. This ordering makes both old and new
manifests usable throughout a rollout, unlike replacing a flat artifact before
clients have refreshed their cached manifest.

The compatibility paths `/data/`, `/data/v3/`, and `/data/v2/` are not part of
the v4 publication transaction and remain untouched.

## Recurring transient and Solar-system bundles

`transients.bin` and `minor-bodies.bin` are independently rebuilt data products.
A single directory containing new copies of either or both can roll forward
both supported complete bundles while retaining the object database and all
unchanged solver catalogs from each base manifest:

```shell
seiza build-data manifest \
  --dir next-dynamic \
  --base-manifest current-v2.json \
  --version catalog-bundle-v2-2026-07-17 \
  --output next-v2.json

seiza build-data manifest \
  --dir next-dynamic \
  --base-manifest current-v4.json \
  --version catalog-bundle-v4-2026-07-17 \
  --output next-v4.json \
  --artifact-dir next-v4-artifacts
```

The first output serves the frozen v3-object compatibility bundle at
`/data/v2/`; it uses the flat keys required by v0.4.1/v0.5 readers. The second
serves v4 readers and automatically assigns every retained or replaced entry
its `artifacts/<sha256>/<name>` key. `--artifact-dir` stages both the canonical
uncompressed file and a maximum-compression zstd transport for every
replacement into one upload-ready content-addressed tree; one S3 sync therefore
publishes both old-reader and new-reader paths. `--zstd-level` can override the
default level 22. Existing transports for retained v4 entries roll forward
unchanged. Both generated manifests are rejected if the resulting catalog set
is incomplete. A v2 base can also be converted to a v4 manifest: retained
hashes become content-addressed without downloading the unchanged
multi-gigabyte files, which can then be copied server-side, but alternate
transports require access to their source bytes.

Sync the generated artifact tree to `/data/v4/` before uploading either
manifest. The tree contains both keys referenced by every replacement entry;
do not upload only the `.zst` objects. Archive `next-v4.json`, verify both
hosted representations, and update the mutable manifest pointer last.

For v4, upload the new content-addressed dynamic artifacts and archive the
manifest before changing the current pointer. For legacy v2, the old flat URL
contract cannot be made fully atomic: upload both dynamic files first and the
v2 manifest last. A client holding a stale v2 manifest may need to refresh it
if it did not already cache the old artifact. The v2 object catalog itself is
never changed during a dynamic-data publication.

## Client cache

`seiza-download` scopes cached manifests by endpoint and stores artifacts under
their SHA-256. The explicitly configured `/data/v2` endpoint retains the v0.5
manifest-cache filename; `/data/v4` uses its own manifest cache. Artifact bytes
with the same hash are shared regardless of which manifest selected them.

A normal cache hit checks file metadata and does not hash or page through a
multi-gigabyte mmap. Bytes are hashed during their initial streaming download;
exhaustive later verification is an explicit API operation. A per-hash
cross-process lock prevents concurrent applications from downloading the same
artifact twice.

When a zstd transport is available, transfer progress reports encoded bytes.
Decompression writes the same uncompressed temporary file that is atomically
installed into the content-addressed cache, so peak cache storage does not
include a retained `.zst` copy and all existing mmap readers remain unchanged.

The small manifest has configurable offline, prefer-cached, age-based refresh,
and force-refresh policies. The default refreshes it after 24 hours and falls
back to a valid stale copy when the network is unavailable. Network access only
occurs through an explicitly awaited `ensure` call, never through a normal
catalog `open`.

Raw source distributions are a separate concern owned by `seiza-sources`.
Gaia TAP, VizieR, MPC, and similar inputs are not installed into the runtime
bundle cache.
