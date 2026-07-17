# Hosted catalog bundles

Hosted paths are compatibility contracts. Once a released reader knows a URL,
the manifest and artifacts at that URL are never replaced with an incompatible
wire format.

## Compatibility layout

| Hosted path | Contents | Readers |
| --- | --- | --- |
| `/data/` | classic files with `SEIZAOB1` objects | v0.3 and classic-v1 clients |
| `/data/v3/` | historical standalone `SEIZAOB3` object/transient manifest | v0.4.0 |
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

The complete v4 manifest must contain the Tycho-2 and Gaia solver tiles, blind
index, stellar identifier sidecar, object and transient catalogs, and minor
bodies. The downloader rejects an incomplete manifest even when the caller
requested only one artifact.

## S3 publication contract

1. Build and exhaustively validate every changed catalog.
2. Upload changed bytes, or server-side copy unchanged bytes, to
   `/data/v4/artifacts/<sha256>/<name>`.
3. Verify every hosted object's byte length and SHA-256.
4. Publish an immutable copy at
   `/data/v4/manifests/<catalog-bundle-version>.json`.
5. Publish `/data/v4/manifest.json` last as the small current-bundle pointer.
6. Run hosted download, validation, and semantic lookup tests through that
   public manifest.

Artifact keys should use long-lived `immutable` cache headers. The current
manifest pointer should have a short lifetime or require revalidation. No
artifact referenced by a published manifest is overwritten; a changed nightly
transient catalog receives a new hash key. This ordering makes both old and new
manifests usable throughout a rollout, unlike replacing a flat artifact before
clients have refreshed their cached manifest.

The compatibility paths `/data/`, `/data/v3/`, and `/data/v2/` are not part of
the v4 publication transaction and remain untouched.

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

The small manifest has configurable offline, prefer-cached, age-based refresh,
and force-refresh policies. The default refreshes it after 24 hours and falls
back to a valid stale copy when the network is unavailable. Network access only
occurs through an explicitly awaited `ensure` call, never through a normal
catalog `open`.

Raw source distributions are a separate concern owned by `seiza-sources`.
Gaia TAP, VizieR, MPC, and similar inputs are not installed into the runtime
bundle cache.
