# External image-processing tools

`seiza-stacking::external` drives external CLIs on stacked images. The first
supported family is RC-Astro's `rc-astro` multi-tool: BlurXTerminator
(`bxt`), StarXTerminator (`sxt`), and NoiseXTerminator (`nxt`).

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
