# seiza-pipeline

Distributed stacking for [Seiza](https://crates.io/crates/seiza): one target
stacked across many machines and many nights, kept current as frames arrive.
See the [design](../docs/design/distributed-stacking.md); this crate is its
first phase.

- **Partial stacks.** `PartialStack` holds one session's frames integrated on
  one grid: the weighted mean, each sample's sum of frame weights, coverage,
  rejection and variance. A weighted mean is a ratio of two sums, so stacks
  that keep their weights combine (`PartialStack::combine`) into exactly the
  mean one stack of all their frames would give, apart from rejection, which
  runs within each session. Combining pools the variances, sums the counts,
  and does not depend on the order the stacks arrive in.
- **Files that describe themselves.** A stack file is FITS or XISF, by its
  extension. Its first image is the mean, so any viewer opens it as an
  ordinary stack; `WEIGHT`, `COVER`, `REJECT` and `VARIANCE` follow. Header
  cards (`Identity`) say what it is (`SEIZAPST`), which versions made it
  (`SEIZASTK`, `SEIZAPIP`), the key of the step that made it (`STEPKEY`) and
  digests of what that step read. `PartialStack::read_identity` reads them
  without decoding pixels. The reference frame's WCS and instrument cards are
  carried; observer names, site coordinates and paths are not.
- **Keys.** A step's key is a SHA-256 digest of the files it reads, by their
  contents, its options and the output versions (`OUTPUT_VERSION`,
  `seiza_stacking::OUTPUT_VERSION`). Two machines with the same inputs
  compute the same key wherever their files are.
- **Sessions.** `SessionStep::plan` hashes a session's lights and masters and
  the channel reference's, and computes the key; `SessionStep::run` stacks
  the session against the reference with
  `LiveStacker::open_fits_sessions_with_reference_image`, each light
  calibrated by its own recipe, optionally reintegrating it with
  leave-one-out rejection.
- **Stores and resuming.** `Store` keeps each step's output under its key, written
  once and never replaced. `run_channel` stacks a channel on one machine,
  running only the steps whose outputs the store lacks: a night added is the
  only one stacked, and the channel is combined again from the stored
  partial stacks.

```rust
use seiza_pipeline::{ChannelPlan, Store, run_channel};

let plan: ChannelPlan = serde_json::from_reader(std::fs::File::open("ngc7000-ha.json")?)?;
let outcome = run_channel(&plan, &Store::new("store"), |progress| eprintln!("{progress:?}"))?;
outcome.stack.write(std::path::Path::new("ngc7000-ha.fits"))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The C ABI runs a channel, one session's step, an identity read and a combine
as one JSON call each (`seiza_pipeline_*_json`); Python binds `PartialStack`,
`SessionStep` and `run_channel`. The CLI's `seiza pipeline` runs a plan file.

`OUTPUT_VERSION` follows the rules of `seiza_stacking::OUTPUT_VERSION`:
`tests/output_version.rs` holds it to the output it promises.

## License

Apache-2.0
