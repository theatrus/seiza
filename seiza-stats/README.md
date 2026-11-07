# seiza-stats

Robust statistics shared across the seiza workspace: medians, median
absolute deviation, and the MAD-to-sigma scale for normal data.

One convention throughout: an even-length sample averages its two middle
elements. Values are ordered with `total_cmp`, including its ordering of
signed zeros and NaNs; callers that must exclude non-finite values filter
before calling.

The f64 median uses linear-time partial selection on a copy, or reads an
already ascending or descending sample without allocating. Robust sigma
selects directly in its absolute-deviation buffer, using one temporary
buffer instead of two. Both leave the caller's sample unchanged.

Run `cargo run --release -p seiza-stats --example robust_profile` to compare
the f64 statistics with their original sorting implementations. The example
checks matching results before timing random sky samples, ties, and ordered
inputs; its isolated modes also support peak-memory measurements.
