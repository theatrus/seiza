# Solving and annotations

[Documentation](README.md) · [Solve performance](performance.md#plate-solving)

## Setup and solve

Install catalogs with the `blind-deep` preset to solve without a position hint.
Set the scale bounds in arcseconds per pixel. For hinted solving, supply an
approximate position through image metadata or the CLI position options, plus
a pixel scale.

```sh
seiza setup --preset blind-deep
seiza solve-blind image.jpg --min-scale 0.5 --max-scale 15
seiza solve image.fits --scale 1.26
seiza solve integration.xisf --scale 1.26
```

FITS, XISF, and raster images share the image-reading path.
Raster EXIF orientation is applied before detection so solutions and overlays
use the same pixel coordinates.

For a portable catalog directory:

```sh
seiza download-data prebuilt --output data
seiza solve-blind image.fits --data data --index data --min-scale 0.5 --max-scale 15
```

`--data` accepts a file or directory. A directory selects the deepest available
star catalog; the index is also discovered there. After `seiza setup`, explicit
`--data` and `--index` are optional. See [catalog discovery](catalogs.md#locations).

Use `--sip-order 2` through `5` to request distortion fitting and `--time-limit`
to bound solving. Full position, scale, and output options are in the
[CLI reference](../seiza-cli/README.md#solving).

## Overlays and sky maps

```sh
seiza solve-blind phone.jpg --sip-order 2 --sky-map sky-map.png --sky-map-foreground
seiza solve image.fits --scale 1.26 --satellites-celestrak --annotate tracks.png
```

Sky maps draw constellation figures, named stars, and deep-sky objects.
`--sky-map-foreground` excludes marks on the ground in a photo with a horizon.
[Example sky map](images/phone/phone-sky-map.jpg).

Satellite tracks are opt-in predictions for one shutter-open exposure, not a
stack or a detected trail. FITS timing can come from `DATE-BEG`/`DATE-END`,
`DATE-AVG` plus `EXPTIME`, `DATE-OBS` plus `EXPTIME`, or `DATE-END` plus `EXPTIME`;
observer coordinates come from `OBSGEO-*`. Explicit `--time`,
`--exposure-seconds`, and `--observer-lat`/`--observer-lon` are also available.
For historical exposures, Seiza retrieves TLEs from the durable cache, rolling
mirror, or IAU SatChecker fallback. See [satellite tracks](design/satellite-tracks.md) and the
[mirror runbook](SATELLITE_MIRROR.md).

## Persistent and remote workers

```sh
seiza worker --data data --index data
```

The worker keeps catalogs and the index open, reads one JSON request per stdin
line, writes one response per stdout line, and exits at EOF. It accepts FITS
and raster paths. See the [request/response protocol](design/worker-protocol.md).

To forward the same requests to a [seiza-server](https://github.com/theatrus/seiza-server):

```sh
seiza worker --server http://solver-host:8080
```

The worker converts FITS to an 8-bit PNG for upload by default. Use
`--server-upload fits` to preserve the original file and headers. Authentication
uses `--server-token` or `SEIZA_SERVER_TOKEN`; `--server-timeout SECONDS` changes
the five-minute remote timeout. Remote results currently carry linear TAN WCS.
