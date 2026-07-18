# Catalog path resolution

Status: implemented in the `seiza` library as `seiza::data_paths`; every CLI
command and compatibility mode resolves its data through it

## Goal

Users should not have to spell out one path per catalog file. A solve needs a
star catalog; annotation needs objects, minor bodies, and transients; blind
solving wants a pattern index. All of these usually sit together in one
directory, downloaded by `seiza setup` or `seiza download-data prebuilt`. One
configured path — or none at all — should be enough.

The same rules must work everywhere: CLI flags, the ASTAP and solve-field
compatibility modes, the worker, and applications that embed the library
(Tenrankai's gallery server is the first). That is why the module lives in
the library, not the CLI.

## How a path resolves

Each catalog kind has a resolver (`star_data`, `blind_index`, `objects`,
`star_identifiers`, `minor_bodies`, `transients`). Every resolver accepts an
optional path and follows the same steps:

1. **A file was given** — use that file. A missing file is an error, never a
   silent fallback.
2. **A directory was given** — pick the right file inside. Star catalogs are
   chosen deepest-first (`stars-deep-gaia17.bin`, then `stars-gaia.bin`,
   `stars-lite-tycho2.bin`, `stars.bin`). The other kinds match their
   standard names (`objects.bin`, `minor-bodies.bin`, `transients.bin`).
   Blind indexes and star-identifier sidecars also accept any file with the
   right extension (`.idx`, `.ids.bin`).
3. **Nothing was given** — check the standard places in order:
   - the kind's environment variable, when it has one (`SEIZA_STAR_DATA`,
     `SEIZA_BLIND_INDEX`); the variable may name a file or a directory. A
     set variable is a pinned choice: when it points at nothing usable,
     that is an error, never a silent fall-through to other catalogs;
   - a `seiza.toml` next to the executable naming the file, or a matching
     data file sitting next to the executable;
   - the shared catalog directories `seiza setup` installs into:
     `SEIZA_CATALOG_DIR` when set, otherwise the platform data directory.

The blind index is the one optional kind: omitted and not found means "build
the index in memory", so the resolver returns `Ok(None)` rather than an
error. An explicitly given index that does not exist is still an error — the
user asked for something specific and did not get it.

## Library and CLI split

The library owns the rules: the resolvers, the search order, the standard
file names, `default_catalog_dir`, and `CATALOG_DIR_ENV`. Errors are a typed
`DataPathError` so applications can match on them.

The CLI owns the flags. `--data` and `--index` feed the resolvers directly,
and `seiza setup` writes into `default_catalog_dir()` — the same function
resolution reads, so setup and lookup can never disagree about the shared
directory.

The transients resolver exists for embedding applications. No CLI command
reads `transients.bin` today, but annotation servers do, and they should not
have to hard-code the file name the rest of the pipeline treats as standard.

## What this replaces

Before 0.7.2, only the ASTAP and solve-field compatibility modes had this
search; every normal CLI command required explicit file paths, and the logic
was private to the CLI crate. 0.7.2 unified the commands behind one CLI
module; this design moves that module into the library with two
deliberate behavior changes — a set environment variable that resolves
to nothing is now an error instead of a silent fall-through, and the
macOS fallback search directory follows the platform convention
(`~/Library/Application Support/seiza`) instead of the Linux one — so
applications resolve catalogs the same way the CLI does.
