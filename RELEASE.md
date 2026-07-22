# Releasing Seiza

Seiza ships three artifacts from one version bump: **crates on crates.io**, **OS
packages on the GitHub Release** (deb / rpm / Windows), and a **Python wheel on
PyPI**. They are cut from a single "Release `<version>`" PR followed by two tags.

## Versioning scheme

- `seiza` and `seiza-cli` inherit the **workspace version**
  (`[workspace.package] version` in the root `Cargo.toml`) — they always move
  together, and the workspace version is *the* release version.
- The other crates version **independently** by what actually changed:
  `seiza-satellites`, `seiza-download`, `seiza-fits`, `seiza-imgproc`,
  `seiza-xisf`, `seiza-stacking`, and `seiza-sources` each have their own
  `version =` in
  their `Cargo.toml`, mirrored by their pin in the root
  `[workspace.dependencies]`.
- `seiza-py` is **`publish = false`** (it goes to PyPI, not crates.io) and its
  version **must always equal the workspace/`seiza` version** — the wheel and the
  crate are the same release.
- Pre-1.0 semver: **minor** bump for new features (e.g. 0.10 → 0.11), **patch**
  for fixes only (e.g. 0.8.0 → 0.8.1). Leaf crates bump by their own change size.

## 1. Decide the version

Diff `main` against the last `Release …` commit and see which crates changed:

```bash
git log --oneline "$(git log --grep='^Release' -1 --format=%H)"..HEAD
git diff --stat "$(git log --grep='^Release' -1 --format=%H)"..HEAD
```

Bump the workspace version if `seiza` or `seiza-cli` changed (they almost always
do). Bump each changed leaf crate by its own change. Fixes-only → patch;
new features → minor.

## 2. Bump versions (the Release PR)

Edit, on a `release/<version>` branch:

- root `Cargo.toml`: `[workspace.package] version`, and the
  `[workspace.dependencies]` pin of **every crate whose version changed**.
- each changed leaf crate's own `Cargo.toml` `version =`.
- `seiza-py/Cargo.toml` `version =` — **set equal to the new workspace version**.

Regenerate both lockfiles (path-crate version sync only — no registry churn):

```bash
cargo check --workspace
(cd seiza-py && cargo check)
git diff -- Cargo.lock seiza-py/Cargo.lock   # expect ONLY version = lines
```

Open the PR:

```bash
git switch -c release/<version>
git commit -am "Release <version>"
gh pr create --title "Release <version>" --body "…summary of what's in it…"
```

CI (`ci.yml`, `python-wheels.yml` test job) must be green. Merge to `main`.

## 3. Publish crates to crates.io (manual, after merge)

`release.yml` does **not** publish to crates.io — do it by hand from a clean
checkout of the merged `main`. Publish **only the crates whose version changed**,
in dependency order (a dependency must be indexed before its dependents; `cargo`
waits for the index):

```
seiza-stats  →  seiza-stretch  →  seiza-imgproc  →  seiza-fits  →  seiza-xisf
→  seiza-background  →  seiza-deconvolution  →  seiza-sources  →  seiza-download
→  seiza  →  seiza-satellites  →  seiza-stacking  →  seiza-cli
```

```bash
git switch main && git pull
cargo publish -p <crate> --locked      # for each changed crate, in the order above
```

(`seiza-py` is never published here — `publish = false`.)

## 4. Tag the OS-package release: `v<version>`

Pushing a `v*` tag triggers `release.yml`, which builds and attaches to the
GitHub Release: the Ubuntu `.deb` (`cargo-deb`), Fedora `.rpm` for fc43 + fc44
(`cargo-generate-rpm`), and the Windows `.zip` + `.msi`.

```bash
git tag -a v<version> -m "Seiza <version>"
git push origin v<version>
```

## 5. Tag the Python wheels: `py-v<version>`

Pushing a **separate** `py-v*` tag triggers the `publish` job in
`python-wheels.yml`, which builds abi3 wheels (linux x86_64 + aarch64, macOS
universal2, Windows x64) and an sdist, then publishes to **PyPI via OIDC trusted
publishing** (the `pypi` environment; no token). Because `seiza-py`'s version was
set to the workspace version in step 2, the wheel version matches the crate
release automatically.

```bash
git tag -a py-v<version> -m "seiza-py <version>"
git push origin py-v<version>
```

## Invariants / gotchas

- **`seiza-py` version == `seiza`/workspace version, always.** The Python package
  and the Rust crates are one release; the `v<version>` and `py-v<version>` tags
  carry the same `<version>`.
- The `v` tag and the `py-v` tag are **separate** — pushing one does not publish
  the other.
- crates.io publishing is **manual**; only `release.yml`'s OS packages and
  `python-wheels.yml`'s PyPI upload are automated by tags.
- If this is a repo's first PyPI publish, enable the trusted publisher / `pypi`
  environment on PyPI before pushing the `py-v` tag.

## Downstream (out of this repo)

The GitHub Release's `.rpm`/`.deb` are what downstream infra mirrors (e.g.
`pkg.stackworks.net`) and hosts consume via `dnf`/`apt`. Bumping a running host
to a new Seiza is a separate step in that infrastructure, after this release is
published.
