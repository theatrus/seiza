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
verified against rc-astro 2.6.6. `process_image` therefore divides by the
image's peak before writing the exchange FITS and multiplies the results
back, starless and stars alike, so `starless + stars` still reconstructs
the original on its own scale.

## Star removal

`sxt` with the `stars` parameter (called `difference` on some builds)
writes a second image beside the requested output: the original minus the
starless result. Its path arrives in the event stream; a name-suffix probe
(`-stars` / `-difference`) is the fallback for builds that omit it.
`process_image` returns it as `ProcessedStackImage::stars` so a caller can
stretch starless and stars independently.

## Cache keys

A host caching results must include `cliVersion` and `mlVersion` from the
schema in its cache key besides the parameter values: a CLI or model
upgrade changes the output for identical inputs.

## Bindings

The C ABI exposes the same surface as two JSON functions:
`seiza_rc_astro_tool_schema_json` (the live contract as schema-1 JSON) and
`seiza_rc_astro_process_file_json` (a file-level run taking the request as
JSON and an optional `SeizaCancelSignal`). Python mirrors it with
`seiza.rc_astro_locate`, `seiza.rc_astro_tool_schema`, and
`seiza.rc_astro_process_file`, which releases the GIL for the run and
reports progress through an optional callback. Both bindings work at the
file level — array round trips stay a Rust-level (`process_image`)
concern.

## Validation

Unit tests run against a fake `rc-astro` shell script replaying a captured
real event stream (Unix only; argv construction and event parsing are
tested purely everywhere). The example `rc_astro_starless` drives the real
tool; it was verified against rc-astro 2.6.6 on a 6248x4176 N.I.N.A. light
frame — starless background 476-890 ADU with the stars image 95.6% zero and
peaking near saturation, both restored to physical scale.

## PixInsight MLDenoise

MLDenoise is the neural-network noise reduction of PixInsight's
MachineLearning module, present in PixInsight 1.9.6. It runs ONNX models that
PixInsight distributes as `.xmlm` files. [`PixInsight`] runs it headless:

- **Launch.** `PixInsight -n --automation-mode -r=<script>,<request>,<result>
  --force-exit`: a new instance runs Seiza's script without stopping at
  dialogs and exits when it ends. PixInsight needs a display; on Linux
  without one, `xvfb-run` supplies a virtual one. The instance runs in its
  own process group, so a cancel closes PixInsight itself and not only a
  launcher (`PixInsight.sh`, `xvfb-run`).
- **Protocol.** The script reads a JSON request (images, model and
  parameters), applies MLDenoise, saves the result, and writes a JSON result
  with the PixInsight version, the seconds taken and, on failure, the error
  and the console log of the run. PixInsight's exit code says nothing about
  the script, and its console output is kept only for a run that ends
  without a result. A process's enumeration constants are properties of its
  instances in PJSR (`P.Precision_High`), not of its constructor.
- **Exchange.** The image goes to PixInsight as a float32 XISF declaring
  `bounds="0:1"`, mapped onto that range the way RC-Astro images are (offset
  by the most negative sample, scaled by the span), and comes back the same
  way. Seiza's own processed-image writer declares 0 to 65535, which
  PixInsight would rescale by, so the exchange is written directly.
- **Parameters.** `modelPath`, `amount` (0 to 1, default 0.9),
  `overlapPercent` (12 to 50, default 25), `precision` (`Precision_Fast`
  or `Precision_High`) and the local-support mask: `mask`, `maskClipLow`
  (-20 to 20), `maskBackground` (0 to 1) and `maskSmoothness` (0 to 7).
  The ranges are the ones PixInsight 1.9.6 enforces, measured by setting
  values in a script; [`MlDenoiseOptions::validate`] refuses values outside
  them before PixInsight starts. `backend` stays `Backend_Auto`, `useCache`
  off, and the previews off.
- **Models.** An `.xmlm` file starts with `XMLM0100`, a little-endian
  `u32` header length and four reserved bytes, followed by an XML header:
  the process the model is for, the format version (`V3`), and one `Model`
  element per payload (`mono.onnx`, `rgb.onnx`, `weights.bin`) with its
  SHA-256. [`MlDenoiseModel::read`] reads only that header, so a run can
  refuse a model for another process, or one without a route for the
  image's channel count, before PixInsight loads 600 MB of weights.

A host caching results keys on the weights' SHA-256, the PixInsight version
and the options.

The C ABI offers `seiza_pixinsight_locate`,
`seiza_pixinsight_ml_denoise_model_json` and
`seiza_pixinsight_ml_denoise_file_json`. Python offers the same, plus
`pixinsight_ml_denoise_image` for arrays.

Unit tests run a stand-in PixInsight, a Python script that follows the
`-r=` protocol (Unix only). A test that runs the installed PixInsight is
ignored by default; with PixInsight 1.9.6 and `MLDenoise_v42.xmlm` it took
the background spread of a synthetic frame from 57.7 to 17.9 ADU in 8.2
seconds, model loading included, and kept its star's peak.
