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
`SEIZASI1`, `SEIZAOB3`, and so on); the bundle version only selects a tested
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

When a future incompatible catalog set is needed, publish a complete
`/data/v3/` bundle. Unchanged large objects can be copied server-side; clients
must never assemble a bundle by mixing v2 and v3 manifests.
