# Persistent worker protocol

`seiza worker` keeps a star catalog and optional blind pattern index open while
it serves plate-solve requests over standard input and output. It is intended
for acquisition applications such as N.I.N.A. that perform several solves in
one session and want process isolation without paying process and catalog setup
costs for every image.

```text
seiza worker --data C:\seiza-data\stars-deep-gaia17.bin --index C:\seiza-data\blind-gaia16.idx
```

Remote-backed worker:

```text
seiza worker --server http://solver-host:8080
```

The JSON-RPC boundary accepts local image paths, including FITS, PNG, JPEG,
and TIFF. The caller remains responsible for the file's lifetime. A worker
may solve locally or use a remote `seiza-server`; remote operation does not
change the client-facing JSON protocol.

## Transport and lifetime

- The protocol is JSON-RPC 2.0 over newline-delimited UTF-8 JSON.
- Each stdin line is one request and each stdout line is one response.
- Stdout is reserved for protocol traffic. Diagnostics and warnings go to
  stderr.
- Requests are handled synchronously, one at a time.
- The maximum request line is 1 MiB.
- Notifications (requests without an `id`) execute without a response, per
  JSON-RPC 2.0. Request IDs may be strings, numbers, or null.
- EOF on stdin is a clean shutdown. A one-shot client sends `initialize` and
  one `solve`, then closes stdin. A persistent client keeps stdin open and
  sends additional `solve` requests.
- `shutdown` writes its response and exits without waiting for EOF.
- Protocol version 1 has no in-band cancellation because requests execute
  synchronously. To cancel an in-flight solve, terminate the child process and
  start a new worker for the next request.

## Initialize

Clients initialize the worker before the first solve and negotiate the Seiza
protocol version independently from JSON-RPC itself.

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientName":"N.I.N.A.","clientVersion":"4.x"}}
```

The response reports server identity, supported solve modes and image inputs,
and the loaded catalog/index metadata:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"server":{"name":"seiza","version":"0.4.1"},"client":{"name":"N.I.N.A.","version":"4.x"},"capabilities":{"solveModes":["hinted","blind"],"imageInputs":["image_path"],"maxConcurrentRequests":1},"catalog":{"starCount":154100000,"blindIndexLoaded":true,"blindIndexPatternCount":1234567,"blindIndexMagnitudeLimit":16.0}}}
```

Protocol version 1 is currently the only supported version.

## Hinted solve

```json
{"jsonrpc":"2.0","id":2,"method":"solve","params":{"imagePath":"C:\\NINA\\PlateSolver\\image.fits","mode":"hinted","hint":{"centerRaDeg":328.43,"centerDecDeg":47.27,"radiusDeg":2.0,"scaleArcsecPerPixel":0.33,"scaleTolerance":0.2},"detection":{"sigma":4.0,"ignoreBorder":0,"maxStars":500}}}
```

The `hint` object is required in hinted mode. Coordinates are ICRS/J2000
degrees. `scaleTolerance` is fractional, so `0.2` allows +/-20 percent.

## Blind solve

```json
{"jsonrpc":"2.0","id":3,"method":"solve","params":{"imagePath":"C:\\NINA\\PlateSolver\\image.fits","mode":"blind","blind":{"minScaleArcsecPerPixel":0.1,"maxScaleArcsecPerPixel":20.0,"indexMagnitudeLimit":12.7,"maxHypotheses":400,"maxCoarseHypotheses":20000},"detection":{"sigma":4.0,"ignoreBorder":0,"maxStars":600}}}
```

The `blind` and `detection` objects may be omitted to use their defaults. When
the worker starts without `--index`, the first blind request builds an index in
memory and later blind requests reuse it. Production clients should provide a
maintained prebuilt index for predictable startup time and fine-scale fields.
The CLI's global `--detection-backend`, `--detection-fallback`, and
`--detection-fallback-hypotheses` settings also apply to local worker solves.

## Solve result

```json
{"jsonrpc":"2.0","id":2,"result":{"schemaVersion":1,"mode":"hinted","image":{"width":6248,"height":4176},"center":{"raDeg":328.4308,"decDeg":47.2731},"pixelScaleArcsecPerPixel":0.3265,"rotationDeg":182.4,"parity":"mirrored","radiusDeg":0.341,"matchedStars":81,"rmsArcsec":0.58,"wcs":{"projection":"TAN","pixelOrigin":1,"crval":[328.4308,47.2731],"crpix":[3124.5,2088.5],"cd":[[-0.00009,0.0],[0.0,0.00009]]},"footprint":[{"raDeg":328.1,"decDeg":47.0},{"raDeg":328.8,"decDeg":47.0},{"raDeg":328.8,"decDeg":47.5},{"raDeg":328.1,"decDeg":47.5}],"timings":{"loadMs":72.0,"detectMs":48.0,"indexMs":0.0,"solveMs":90.0,"totalMs":214.0}}}
```

The WCS uses the FITS TAN/CD convention. `crpix` is one-based as indicated by
`pixelOrigin`; this differs from Seiza's internal zero-based pixel centers and
allows clients to pass the values directly to FITS/WCS consumers.

`indexMs` is normally zero for a worker started with `--index`. If the worker
builds an index lazily, the build time is reported there rather than in
`solveMs`.

Remote results also include `timings.encodeMs`, `timings.transportMs`, and a
`transfer` object containing the encoding and byte counts. `transportMs`
includes upload, queue wait, solve, polling, and result download. Local results
use zero for those timings and omit `transfer`.

## Remote backend

With `--server`, the worker still receives a local FITS path. It applies the
same MTF stretch as local solving and encodes the solver input as a lossless
8-bit grayscale PNG. It submits that PNG and the mapped solve options through
`seiza-server`'s native `POST /api/v1/solves` multipart API, then polls
`GET /api/v1/solves/{id}` until the queued solve completes. The server never
receives or resolves the client's path, and no server-side protocol extension
is required.

`--server-token` or `SEIZA_SERVER_TOKEN` supplies a bearer token for servers
running in API-key mode. Blind-index build controls that are local engine
details are not sent; the remote service uses its configured maintained index
and queue policy.

`--server-upload png` is the default. `--server-upload fits` uploads the
original file as a streamed multipart part instead, preserving FITS headers
such as `DATE-OBS` and the source bit depth without buffering the whole FITS
payload in memory. It requires a FITS input path and costs more bandwidth.

The upload, queue wait, solve, and result fetch share a five-minute deadline.
Use `--server-timeout SECONDS` to change it. Individual health checks are
capped at 30 seconds.

## Other methods

Ping:

```json
{"jsonrpc":"2.0","id":4,"method":"ping"}
```

Shutdown:

```json
{"jsonrpc":"2.0","id":5,"method":"shutdown"}
```

## Errors

Errors use the JSON-RPC error shape and echo the request ID when it was
available.

| Code | Meaning |
| ---: | --- |
| `-32700` | Invalid JSON |
| `-32600` | Invalid JSON-RPC request or unsupported JSON-RPC version |
| `-32601` | Unknown method |
| `-32602` | Invalid parameters or unsupported Seiza protocol version |
| `-32002` | `solve` sent before `initialize` |
| `-32010` | Image loading, detection, or plate solving failed |

Malformed requests do not stop the worker. A client may continue with the next
line or terminate and restart the child process.
