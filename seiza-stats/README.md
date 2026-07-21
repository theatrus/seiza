# seiza-stats

Robust statistics shared across the seiza workspace: medians, median
absolute deviation, and the MAD-to-sigma scale for normal data.

One convention throughout: an even-length sample averages its two middle
elements. Values are ordered with `total_cmp`, so NaN sorts high; callers
that must exclude non-finite values filter before calling.
