# Remote solving protocol

`seiza-server` keeps a star catalog and optional blind index open and exposes
the solving engine over HTTP. It is intended to sit behind the local
`seiza worker` process, preserving the JSON-RPC/FITS-path contract used by
acquisition applications.

```text
seiza-server --data data/stars-deep-gaia17.bin \
  --index data/blind-gaia16.idx --listen 0.0.0.0:7878

seiza worker --server http://solver-host:7878
```

Both commands read `SEIZA_SERVER_TOKEN`. When set on the server, requests must
use `Authorization: Bearer <token>`. Native TLS is intentionally outside this
small HTTP service; use an HTTPS reverse proxy or private network for traffic
that leaves a trusted host.

## Endpoints

### `GET /v1/status`

Returns the server version, catalog star count, blind-index state, and the
`gray8-zstd` image capability as JSON.

### `POST /v1/solve`

Accepts `Content-Type: application/vnd.seiza.solve-image+zstd` and returns the
same JSON solve result as the worker protocol. Unsuccessful requests return a
plain-text diagnostic with a non-2xx status.

The request body is a versioned binary envelope:

| Bytes | Meaning |
| --- | --- |
| 8 | Magic `SEIZA\\0I1` |
| 4 | Little-endian JSON metadata length |
| N | UTF-8 JSON metadata |
| remainder | Zstandard-compressed 8-bit grayscale pixels, row-major |

Metadata contains schema version 1, width, height, uncompressed byte count,
solve mode, hint or blind parameters, and star-detection parameters. The
server validates dimensions and byte counts and limits both request and
decompressed image sizes to 128 MiB.

## Why pixels instead of FITS

Local Seiza solving already converts FITS data to an MTF-stretched 8-bit
grayscale image before star detection. Sending that exact solver input:

- avoids transferring FITS headers and unused high-bit-depth samples;
- keeps local file paths and unrelated FITS metadata off the server;
- is lossless relative to the image the solver would use locally; and
- usually compresses substantially for astronomy images.

The worker reports `transfer.encoding`, `transfer.uncompressedBytes`, and
`transfer.encodedBytes` so clients can observe the actual bandwidth cost.

Protocol changes must increment the envelope schema version or introduce a
new content type. Unknown versions and encodings are rejected.
