# Hosted catalog bundles

Hosted paths version complete, coherent catalog bundles rather than individual
wire formats.

## Layout

- `/data/manifest.json` and immediate `/data/*` files are the temporary
  classic-v1 compatibility surface.
- `/data/v2/manifest.json` describes every current prebuilt catalog, all stored
  immediately under `/data/v2/`.
- Downloads remain flat locally, so existing CLI arguments and environment
  variables use the familiar filenames.

The v2 bundle contains the Tycho-2 and Gaia solver tiles, the blind index, the
stellar identifier sidecar, object and transient catalogs, and minor bodies.
Individual files retain their own self-describing wire headers (`SEIZAST2`,
`SEIZASI1`, the sectioned `SEIZAOB\0` object container, and so on); the bundle version only selects a tested
combination of those formats and datasets.

## Publication contract

The publisher uploads or server-side copies every data object first, verifies
that the required filenames are present, builds the manifest from the hosted
bytes, and publishes `manifest.json` last. It refuses to publish a partial v2
bundle. The downloader likewise rejects a v2 manifest that omits any required
catalog, even when the user requested only one file.

Hosted integration tests download the stellar sidecar through the public
manifest, verify its SHA-256, validate the complete mapped file, and perform a
semantic name lookup. This turns a missing data upload into a release-gate
failure instead of a documentation-only feature.

## Client cache

`seiza-download` stores hosted artifacts under their manifest SHA-256 and
returns those immutable paths directly to library callers. A normal cache hit
checks file metadata and does not hash or page through a multi-gigabyte mmap.
The bytes are hashed during their initial streaming download; exhaustive later
verification is an explicit API operation. A per-hash cross-process lock
prevents concurrent applications from downloading the same artifact twice.

The small manifest has configurable offline, prefer-cached, age-based refresh,
and force-refresh policies. The default refreshes it after 24 hours and falls
back to a valid stale copy when the network is unavailable. Network access only
occurs through an explicitly awaited `ensure` call, never through a normal
catalog `open`.

Raw source distributions are a separate concern owned by `seiza-sources`.
Gaia TAP, VizieR, MPC, and similar inputs are not installed into the runtime
bundle cache.

When a future incompatible catalog set is needed, publish a complete
`/data/v3/` bundle. Unchanged large objects can be copied server-side; clients
must never assemble a bundle by mixing v2 and v3 manifests.
