# Rank-robust star matching

Status: analysis and design recorded; first mitigation (source-image
redetection in solve-field mode) implemented; the catalog-seeded quad
fallback is the planned core change

## The root of the problem

Plate solving must find a correspondence between image detections and
catalog stars before the transform is known, and cannot test a
correspondence without the transform. Every solver breaks that loop the
same way: both sides independently nominate a small subset, and geometric
agreement is searched only within those subsets. The design question is
which shared observable nominates overlapping subsets.

Only two observables exist, and they differ sharply in robustness:

- **Relative geometry** (pairwise distances and angles) survives the whole
  imaging pipeline nearly untouched — stretching, clipping, narrowband
  filters, and saturation do not move centroids.
- **Brightness** passes through every lossy stage: sensor response,
  stretch, clipping, bandpass-versus-catalog mismatch, and finally the
  producer's choice of measurement (integrated flux, PSF amplitude, SNR).

Seiza's matcher uses brightness ranking as its only subset-nomination
mechanism: hinted solving forms triangles from the top-24 image stars
against the top-100 catalog stars per window; blind solving selects image
pattern stars by local brightness. That converts an intractable search
into a tiny one — the source of seiza's speed — but makes the fragile
channel load-bearing with no fallback. The implicit precondition is that
the caller's top-24 shares at least ~3 members with the catalog's bright
subset. The precondition is undocumented at the API boundary, unvalidated
at runtime (a bad ranking is indistinguishable from an empty field), and
fails hardest exactly where it matters: the brightest stars are the
saturated ones whose fitted amplitudes clip or diverge.

## Evidence

Siril's astrometry.net integration passes star lists whose `FLUX` column
is the fitted PSF amplitude (`psf_star->A`). Measured on a real stretched
M31 export solved through the solve-field drop-in:

- Star positions were excellent: 191/200 within 0.14 px of seiza's own
  detections on the same pixels.
- Only 2 of the 24 highest-amplitude stars were among the 100
  photometrically brightest; the catalog-bright stars sat at amplitude
  ranks 48–1875 (median 235).
- The identical star positions solved (85 matched, 1.88" RMS) when paired
  with photometric flux values, and failed with the amplitude ranking —
  under every orientation, subset-diversity, and pool-size variation
  tried. Doubling the triangle pool did not help; per-cell locally
  brightest selection did not help (amplitude noise is local too).

The same limitation applies to any external star-list producer: the
sirilpy scripting path (`get_image_stars()` exposes the same amplitudes),
worker-protocol clients, and seiza-py callers passing `(x, y, flux)`
tuples with untrusted flux.

## Mitigation 1 (implemented): re-measure the fragile channel

The solve-field mode receives `<image>.xyls`, so the source image path is
recoverable next to it. When a sibling image file exists and its
dimensions match the table's `IMAGEW`/`IMAGEH`, seiza re-detects stars
with its own detector — restoring trusted integrated flux — and
cross-matches the redetected positions against the table's to establish
the orientation (Siril's tables are bottom-up) and the sub-pixel offset
between the two frames, then solves in the exact frame the table (and
therefore the `.wcs` consumer) uses. If no image is found, dimensions
mismatch (cropped or downsampled selections), or the cross-match is weak,
the table's own stars are used unchanged.

This fixes the dominant real-world case at the cost of one extra
detection pass, but it is a workaround: it restores the precondition
rather than removing it, and it cannot help callers who genuinely only
have a star list.

## Mitigation 2 (planned): catalog-seeded quads

Remove the root by demoting brightness from load-bearing precondition to
optional search-ordering hint:

- Build 4-star quads from the **catalog** side, where brightness is
  trustworthy — the same anchor-plus-nearest-bright-neighbors recipe the
  blind index already uses, applied to the hinted window's cone results
  and projected to the tangent plane (~100–400 quads, built per solve).
- On the image side, use **no ranking at all**: a position hash over the
  full input star list. For each catalog quad, scan image pairs matching
  the quad's backbone length (annulus lookup; scale known to a few
  percent per sub-hypothesis). Each pair fixes rotation and parity, so
  the quad's remaining two stars become O(1) grid lookups. Two
  independent positional coincidences at match tolerance make false
  positives rare; survivors run the existing hinted verifier.
- Use claimed flux only as a **soft prior** ordering anchor candidates:
  half the catalog-bright stars sat within amplitude rank ~235 in the
  measured case, so typical solves stay fast and the worst case is
  bounded work instead of failure.

Estimated cost on the measured M31 case: tens of milliseconds typical,
sub-second bounded worst case.

Position-blind solving with an unranked list is harder: the whole-sky
pattern index is already catalog-seeded, but the image-side query
patterns are rank-selected today. The first step there is a
junk-tolerant, deeper image-pattern generation ladder with a larger
hypothesis budget (hash verification is cheap); index-format changes are
not required for that.

## Data-format implications

- **Hinted rank-robust fallback: none.** Catalog quads are derived per
  solve from what `cone_search` already returns; the image position hash
  is in-memory. Pure solver-code change, deployable in a point release.
- **Star tile catalogs: none.** Precomputing quads into a file would
  persist something cheaper to derive than to load.
- **Blind index: none required.** Rank sensitivity lives in image-side
  selection, which is in-memory. If search-strategy changes prove too
  slow, additional density tiers or per-pattern neighborhood metadata
  would be an additive index revision — the versioned headers and
  content-addressed v4 bundle manifests absorb that without disturbing
  released readers.
- **APIs: additive at most.** An optional flux-trust declaration
  (`photometric` / `ordinal` / `none`) on SolveHint, the worker request,
  and seiza-py lets callers skip the trial-and-error between the fast and
  robust paths; solve-field mode would declare untrusted automatically.
  Default behavior without the flag: fast path first, robust fallback.
- **Outputs: untouched.**

## Upstream

Siril writes `psf_star->A` into a column named `FLUX`
(`save_list_as_FITS_table`, src/algos/star_finder.c). Real astrometry.net
sorts by the same column (`-s FLUX`) and degrades too, just more
gracefully thanks to depth scanning. An upstream patch writing integrated
flux would benefit both solvers and is worth proposing independently of
seiza's mitigations.

## Acceptance criteria

- The real Siril invocation (siril-cli, `platesolve -localasnet`) solves
  the stretched M31 export that currently fails.
- A synthetic regression test that randomizes flux ranking on an
  otherwise solvable field passes with the robust path and fails without
  it.
- Well-ranked inputs keep their current solve times (fast path first).
