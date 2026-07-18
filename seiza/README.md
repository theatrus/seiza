# seiza (星座)

Star detection, WCS fitting, and plate solving — hinted and blind — for
astrophotography, in Rust.

- **Star detection** — tile-based background/noise estimation (median +
  MAD), sigma thresholding, connected components, flux-weighted sub-pixel
  centroids. `DetectConfig::backend` selects automatic, u8, or f32 sampling;
  the shared pipeline is statically dispatched over the sample type.
- **WCS** — TAN (gnomonic) projection with a CD matrix: pixel ↔ world
  transforms, scale/footprint helpers.
- **Hinted plate solving** — triangle matching over FOV-sized windows,
  affine candidate voting, iterative least-squares refinement, seeded by an
  approximate center and pixel scale. Solves real telescope images in tens
  of milliseconds with sub-arcsecond RMS. Optionally fits SIP distortion
  polynomials (orders 2-5, forward and inverse) when they improve the
  residual.
- **Blind plate solving** — no position needed, only a plausible
  pixel-scale range. A disc-anchored 4-star pattern index over the whole
  sky is matched against quads of the brightest detections; hypotheses are
  voted on, smoothed, and verified in parallel by the hinted solver.
  Under 2 seconds per wide-field image including building the whole-sky
  index from a 2.5M-star catalog; a 61 MP FITS frame goes from file open
  to hinted solution in 0.7 s.
- **Star catalogs** — compact memory-mappable tile formats with cone
  search; builders for Tycho-2, Gaia DR3 (via TAP), and ASTAP databases.
- **Object catalogs** — NGC/IC/Messier, Sharpless, Barnard, UGC, LDN, LBN,
  Cederblad, vdB, PGC, named/HD stars, and live transient (supernova/nova)
  lists built into an extensible, memory-mapped sectioned container with
  stable source IDs, aliases, hierarchy, and provenance; query known sky
  cones and convex footprints without plate solving, or project objects into
  solved images with full ellipse geometry. Cold detail sections keep every
  contributing upstream record, typed relations, preferred facet selections,
  and source-qualified geometry (ellipses and outline contours) behind
  `object_details`, `catalog_records`, `geometries`, `relations`, and
  `capabilities`, without touching normal query paths. Legacy `SEIZAOB1` and
  `SEIZAOB3` files remain readable.
- **Catalog path resolution** — `seiza::data_paths` finds catalog files the
  same way the CLI does. Give a resolver a file and it uses that file. Give
  it a directory and it picks the right file inside (the deepest star
  catalog wins). Give it nothing and it checks the standard places:
  environment variables (`SEIZA_STAR_DATA`, `SEIZA_BLIND_INDEX`), files next
  to the program, and the `seiza setup` directories (`SEIZA_CATALOG_DIR`).
  Resolvers exist for star, blind-index, object, star-identifier,
  minor-body, and transient catalogs.
- **Optional catalog downloads** — enable the non-default `downloads` feature
  for `seiza::downloads`, an async, verified shared cache of published catalog
  bundles. Normal catalog opens never access the network.

See the [`seiza-cli`](https://crates.io/crates/seiza-cli) crate for the
command-line tool, and [`seiza-fits`](https://crates.io/crates/seiza-fits)
for dependency-free FITS reading and autostretch. Raw catalog-building source
acquisition is separately available from
[`seiza-sources`](https://crates.io/crates/seiza-sources).

## License

Apache-2.0
