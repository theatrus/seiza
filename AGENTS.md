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
   - A new C export: the Python function, method or property that does the
     same, declared in `seiza-py/seiza.pyi`, described in
     `seiza-py/README.md` and tested in `seiza-py/tests/`.
   - A new JSON request field: a Python keyword argument.
   - A new JSON response field: the same key or attribute on Python's result.
   - A behaviour change in a C function: the same change in its Python twin.
     `.github/binding-parity.toml` lists each C function's twin.
   - A Python change: the same in the C ABI, the other way round.
3. **Record what a binding lacks** in `.github/binding-parity.toml`: `none`
   where it does without by design (C handle lifetimes, the apps' display
   rendering, a cancel signal passed as an argument), `gap` where it should
   have it but does not yet. The gaps are the work left; close them rather
   than adding to them.
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
