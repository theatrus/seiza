# External image-processing tools

`seiza-stacking::external` drives external CLIs on stacked images. The first
supported family is RC-Astro's `rc-astro` multi-tool: BlurXTerminator
(`bxt`), StarXTerminator (`sxt`), and NoiseXTerminator (`nxt`).
`seiza-stacking::pixinsight` runs PixInsight's MLDenoise through an
installed PixInsight; see [PixInsight MLDenoise](#pixinsight-mldenoise).

## The contract

`rc-astro` publishes a machine-readable contract (their README-DEVS.md;
contract versions v3 through v6 are known):

- `rc-astro <tool> --json` prints a schema document: every parameter with
  its `flag`, `type`, range, and default, plus `schemaVersion`,
  `cliVersion`, `mlVersion`, and the product's `license` state.
- A processing run under `--json` emits NDJSON events on stdout:
  `progress` (percent done), `status` with `phase`/`output` (one
  saving/complete pair per file written), `warning`, `error`, and `info`
  (topics `device`, `license`, `update`). Stderr stays empty on success.
- Exit code 77 means the product is not licensed.
- v4 replaced the per-product `--engine` with a global `--device`; v6 added
  `--host <integrator>` for support attribution.

Parameter names and flags change between CLI builds (`--stars` became
`--difference` on some builds; bxt's `--nsr` became `--nsd`), so
[`RcAstroCli`] resolves every flag from the live schema instead of
hard-coding it, and a host UI should render its controls from
[`ExternalToolSchema`] the same way. Values coerce across the JSON number
divide — a whole number satisfies a float parameter and a fraction-free
float satisfies an int parameter — and a `false` switch emits the CLI's
documented `--no-<flag>` negation so it really overrides a true default.

## Sample scale

The tools clamp float samples to `[0, 1]` (the PixInsight convention). A
stacked image on a physical ADU scale would come back flattened to white —
verified against rc-astro 2.6.6. `process_image` therefore maps the image
onto that range for the exchange FITS and maps the results back, with the
same map PixInsight MLDenoise uses (`seiza-stacking`'s `exchange` module):

- An image within 0 to 1 keeps its scale. Any other is offset by its most
  negative sample, if it has one, and divided by 65535 or its span,
  whichever is larger, the scale PixInsight gives a 16-bit camera frame.
  Before `EXTERNAL_EXCHANGE_VERSION` 2 it was divided by its span alone,
  so a dim image or crop sat near the top of the range, where the tools see
  something other than what PixInsight would give them for the same data.
- Samples that are not finite, such as a stack's uncovered edges, go to the
  tool as their channel's median plus noise of the channel's robust sigma,
  and come back unchanged in the result and in the stars image. Before,
  they reached the tool as NaN: NoiseXTerminator 2.6.10 then returned the
  whole image as NaN, not only the uncovered pixels.
- The stars image, a difference of two images on 0 to 1, is restored with
  the scale alone, so `starless + stars` still reconstructs the original on
  its own scale.

A host caching `process_image` results adds `EXTERNAL_EXCHANGE_VERSION` to
its cache key, besides the tool's versions and the parameters.

## Star removal

`sxt` with the `stars` parameter (called `difference` on some builds)
writes a second image beside the requested output: the original minus the
starless result. Its path arrives in the event stream; a name-suffix probe
(`-stars` / `-difference`) is the fallback for builds that omit it.
`process_image` returns it as `ProcessedStackImage::stars` so a caller can
stretch starless and stars independently.

`split_stars` asks for unscreened stars, which restore exactly only on the
image's own 0 to 1. It is given a stretched image, so it clamps the image's
finite samples to 0 to 1 first; the map then leaves the image as it is, and
a sample just outside the range no longer moves the whole image onto the
camera scale.

## Cache keys

A host caching results must include `cliVersion` and `mlVersion` from the
schema in its cache key besides the parameter values: a CLI or model
upgrade changes the output for identical inputs. A host caching
`process_image` or `split_stars` results also includes
`EXTERNAL_EXCHANGE_VERSION` (C `seiza_external_exchange_version`, Python
`seiza.EXTERNAL_EXCHANGE_VERSION`): a change to the map gives different
results for the same tool and parameters.

## Bindings

The C ABI exposes the same surface as two JSON functions:
`seiza_rc_astro_tool_schema_json` (the live contract as schema-1 JSON) and
`seiza_rc_astro_process_file_json` (a file-level run taking the request as
JSON and an optional `SeizaCancelSignal`). Python mirrors it with
`seiza.rc_astro_locate`, `seiza.rc_astro_tool_schema`, and
`seiza.rc_astro_process_file`, which releases the GIL for the run and
reports progress through an optional callback. Both work at the file
level and pass files to the tool as they are. Python also runs a tool on an
array (`seiza.rc_astro_process_image`, `seiza.rc_astro_split_stars`) through
`process_image` and its map.

## Validation

Unit tests run against a fake `rc-astro` shell script replaying a captured
real event stream (Unix only; argv construction and event parsing are
tested purely everywhere). The example `rc_astro_starless` drives the real
tool; it was verified against rc-astro 2.6.6 on a 6248x4176 N.I.N.A. light
frame — starless background 476-890 ADU with the stars image 95.6% zero and
peaking near saturation, both restored to physical scale.

## PixInsight MLDenoise

MLDenoise is the neural-network noise reduction of PixInsight's
MachineLearning module, present in PixInsight 1.9.6. It runs ONNX models
that PixInsight distributes as `.xmlm` files. [`PixInsight`] runs it
headless.

### Launch

`PixInsight -n --automation-mode --no-focus-stealing
-r=<script>,<request>,<result>,<started> --force-exit` starts a new
instance that runs Seiza's script without stopping at most dialogs, keeps
its window behind the user's application, and exits when the script ends.
PixInsight needs a display; on Linux without one, `xvfb-run` supplies a
virtual one.

- **Process group.** The instance runs in its own process group, so closing
  it also stops PixInsight when Seiza started a launcher (`PixInsight.sh`
  or `xvfb-run`). A cancel or the time limit closes the group. A host that
  exits during a run does not: PixInsight finishes on its own and leaves its
  temporary folder behind.
- **Instance slots.** `-n` takes the first free instance slot, and two
  instances starting at the same moment can take the same one; the second
  then fails with "Unable to create application instance". Runs in one
  process therefore start one at a time: a lock is held until the script
  writes its `started` file. A run whose instance failed that way, for
  example against another process's instance, is started again, up to
  three times.
- **Time limit.** PixInsight shows license dialogs even in automation
  mode, and nobody can answer one on a virtual display. A run is closed
  after ten minutes plus 20 seconds per million samples, or after the
  caller's `timeout_seconds`.
- **Arguments.** PixInsight splits script arguments at commas, so a
  temporary folder with a comma is refused. The Linux launcher
  `PixInsight.sh` passes its arguments through the shell again, so with it a
  temporary folder whose path has characters other than letters, digits and
  `/ . _ - +` is refused too.

### Protocol

The script reads a JSON request (image paths, model and parameters),
applies MLDenoise, saves the result, and writes a JSON result with the
PixInsight version and build, the seconds taken and, on failure, the error
and the console log of the run. PixInsight's exit code does not show
whether the script succeeded, and its console output is kept only for a
run that ends without a result. A process's enumeration constants are
properties of its instances in PJSR (`P.Precision_High`), not of its
constructor.

### Exchange

The image goes to PixInsight as a float32 XISF declaring `bounds="0:1"`.
Seiza's own processed-image writer declares bounds 0:65535, and PixInsight
divides samples by the declared range when it opens a file, so the
exchange file is written with bounds 0:1 directly.

- **Scale.** An image whose samples lie within 0 to 1 keeps its scale. Any
  other image is offset by its most negative sample, if it has one, and
  divided by 65535 or its span, whichever is larger: the scale PixInsight
  gives a 16-bit camera frame. Dividing by the span alone made the result
  depend on the brightest sample; a dim crop then sat near the top of the
  range and came back with its background biased by about 10 ADU.
- **Samples that are not finite.** NaN marks a stack's uncovered pixels.
  These and infinite samples reach PixInsight as their channel's median
  plus noise of the channel's robust sigma, and come back unchanged, so a
  stack keeps its coverage. A noiseless fill is not enough: MLDenoise then
  barely denoises any of the image.
- **Result.** PixInsight 1.9.6 saves the result as Float32 with
  `bounds="0:1"`; a result declaring other bounds is refused.

`ml_denoise_file` writes its output with the input's headers. An XISF
output declares the bounds the input declared, so an image PixInsight
normalized stays on 0 to 1, or 0:65535 when the input declared none.

### Parameters

`modelPath`, `amount` (0 to 1, default 0.9), `overlapPercent` (12 to 50,
default 25), `precision` (`Precision_Fast` or `Precision_High`) and the
local-support mask: `mask`, `maskClipLow` (-20 to 20), `maskBackground` (0
to 1) and `maskSmoothness` (0 to 7). The ranges are the ones PixInsight
1.9.6 enforces, measured by setting values in a script;
[`MlDenoiseOptions::validate`] refuses values outside them before PixInsight
starts, as [`Error::ExternalToolRefused`]. `backend` stays `Backend_Auto`,
`useCache` off, and the previews off.

### Models

An `.xmlm` file starts with `XMLM0100`, a little-endian `u32` header
length and four reserved bytes, followed by an XML header: the process the
model is for, the format version (`V3`), and one `Model` element per
payload (`mono.onnx`, `rgb.onnx`, `weights.bin`) with its SHA-256.
[`MlDenoiseModel::read`] reads only that header, so a run can refuse a
model for another process, or one without a route for the image's channel
count, before PixInsight loads 600 MB of weights.

A host caching results keys on the header's SHA-256 (`header_sha256`), the
PixInsight version and build, and the options. The weights' SHA-256 alone
is not enough: `MLDenoise_v41.xmlm` and `MLDenoise_v42.xmlm` share their
weights, differ in their mono and RGB networks, and give results up to 20
ADU apart.

### Bindings and tests

The C ABI offers `seiza_pixinsight_locate`,
`seiza_pixinsight_ml_denoise_model_json` and
`seiza_pixinsight_ml_denoise_file_json`. Python offers the same, plus
`pixinsight_ml_denoise_image` for arrays.

Unit tests run a stand-in PixInsight, a Python script that follows the
`-r=` protocol (Unix only). A test that runs the installed PixInsight is
ignored by default; set `SEIZA_MLDENOISE_MODEL` to a model file to run it.
