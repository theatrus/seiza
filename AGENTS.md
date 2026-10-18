# Notes for coding agents

## The core, the C ABI and Python move together

Seiza's core crates reach their hosts through two bindings that should
offer the same things in the same way: the C ABI (`seiza-cabi`, which the
Mac and Windows apps use) and Python (`seiza-py`). They drift apart easily,
because each change is usually made for one host. When you change a core
type the C ABI carries, or either binding:

1. **Put behaviour in the core crate** (`seiza`, `seiza-parallax`,
   `seiza-stacking`, ...), and keep each binding a thin translation of it.
   Logic written in one binding's glue is logic the other lacks: #277
   fixed cropped dust maps in the C ABI's glue, and Python's own copy of
   that glue kept the old behaviour until it was fixed too.
2. **Carry each change through the core, the C ABI and Python, in the same PR.**
   - A new public field on a core type the C ABI carries (each such pair is
     a `[rust.<C type>]` entry, such as `seiza-parallax::VideoOptions` for
     `VideoRequest`): the C field, then the Python keyword or attribute.
   - A new public function a core crate exports, from its root or from a
     public module (`pub mod dust` counts), or a new public method of an
     object hosts hold (`Parallax`, `LiveStacker` and
     the others under `[core_functions] objects`): a C export that calls it,
     then its Python twin.
   - A new C export: the Python function, method or property that does the
     same, declared in `seiza-py/seiza.pyi`, described in
     `seiza-py/README.md` and tested in `seiza-py/tests/`.
   - A new JSON request field: a Python keyword argument.
   - A new JSON response field: the same key or attribute on Python's result.
   - A behaviour change in a C function: the same change in its Python twin.
     `.github/binding-parity.toml` lists each C function's twin.
   - A Python change: the same in the C ABI, the other way round.
3. **Record only deliberate differences** in `.github/binding-parity.toml`:
   `none`, with the reason, where a binding does without by design (C handle
   lifetimes, the apps' display rendering, a cancel signal passed as an
   argument). There is no way to record a missing counterpart: the check
   fails until both bindings have it.
4. **Run the check:**
   `python3 .github/scripts/check_binding_parity.py --base origin/main`.
   CI runs it on every PR. A PR that changes one binding's source and not
   the other's fails, unless a commit message carries a line
   `Binding-parity: <why the other binding needs nothing>`.

Names: the C ABI's JSON is camelCase of the Rust field names; Python is
snake_case, with shorter units: `_pc` for `_parsecs`, `_deg` for `_degrees`,
`_arcsec_px` for `_arcsec_per_pixel`, `_mag` for `_magnitude`.

`seiza-py/tests/test_stub.py` holds `seiza.pyi` to the built module, so the
stub stays a true account of Python's surface.

## Output versions: revise and test, every time

`seiza_stacking::OUTPUT_VERSION` and `seiza_stretch::OUTPUT_VERSION` give the
release whose output a build reproduces. PSF Guard keys its cached stacks,
masters and previews on them. A wrong value is worse than none: if output
changes and the value stays put, users keep stale stacks and are never told.
So these steps are not optional, and a PR that skips them is not done.

1. **Before you change anything in `seiza-stacking` or `seiza-stretch`, or in
   a crate either one calls** (calibration, registration, resampling,
   rejection, normalization, weighting, demosaicing, drizzle, defect or
   impulse repair, masters, mosaic canvases, any stretch model), run:

   ```sh
   cargo test -p seiza-stacking --test output_version
   cargo test -p seiza-stretch --test output_version
   ```

2. **Check that a case covers what you are changing, in the mode you are
   changing**, not only the defaults. If none does, add one first and record
   it on the unchanged code. The data must be data your code acts on.
   Lanczos-3's clamp (#295) changed how faint skies stack, and the only
   Lanczos case, a bright sky, passed both before and after it: only a faint
   sky a few sigma above zero shows the clamp. A case counts only when a
   planted change in its mode makes it fail. Plant one, see it fail, and
   undo it.
3. **Make your change and run both tests again.**
   - **If a case fails and you meant to change output:** set that crate's
     `OUTPUT_VERSION` to the crate's next release and bump the crate to that
     version in the same PR, since the test requires `OUTPUT_VERSION` to be
     no newer than the crate. Then record the new summary the failure prints,
     and give the reason in the PR.
   - **If a case fails and you did not mean to change output:** your change
     is not as neutral as you thought. Find out why before going on.
   - **If nothing fails:** say in the PR that both tests pass unchanged, and
     which case covers your change.
4. **Never** re-record a summary without moving `OUTPUT_VERSION`, loosen
   `TOLERANCE` to make a case pass, or drop a case because it fails.
   `OUTPUT_VERSION` only moves forward: from "0.40.0" for stacking and
   "0.3.0" for stretching, each change sets the release that ships it.

## Other rules

- Which crate to depend on, and how releases work: `RELEASE.md`.
- Every image file is read and written through `seiza-image-io`, by its
  extension, the C ABI included.
- Pixels count from 0, with centres on whole numbers; an image's middle is
  `((width - 1) / 2, (height - 1) / 2)`. FITS headers count from 1, so a
  FITS `CRPIX` is the internal value plus 1.
- Building `seiza-cabi` regenerates `seiza-cabi/include/seiza_cabi.h`;
  commit it with the change.
- `seiza-py` is outside the workspace. Format, lint and check it on its own.

## Before opening a PR

```sh
cargo fmt --all --check && (cd seiza-py && cargo fmt --check)
cargo clippy --workspace --all-targets --all-features -- -D warnings
(cd seiza-py && cargo clippy --all-targets -- -D warnings)
cargo +1.89.0 check --workspace --all-targets --all-features --locked
cargo test --workspace --all-features
python3 .github/scripts/check_binding_parity.py --base origin/main
(cd seiza-py && maturin develop && python -m pytest tests)
```
