//! Robust statistics shared across the seiza workspace.
//!
//! One median convention throughout: an even-length sample averages its two
//! middle elements. Ordering uses `total_cmp`, including its ordering of
//! signed zeros and NaNs; callers that must exclude non-finite values filter
//! before calling.

/// Scale from the median absolute deviation to the standard deviation of
/// normally distributed data.
pub const NORMAL_MAD_SCALE: f64 = 1.4826;

/// [`NORMAL_MAD_SCALE`] for f32 pipelines.
pub const NORMAL_MAD_SCALE_F32: f32 = 1.4826;

/// Median by partial selection, reordering `values`. Returns `None` for an
/// empty sample.
///
/// The selection runs on each value's bits mapped to an integer that orders
/// as `total_cmp` does, so the median is the same value, and `values` is left
/// partitioned around it, as selecting by `total_cmp` would leave them. On a
/// 64x64 tile of sky that is 4.3 us against 6.2: comparing integers is
/// cheaper than `total_cmp`, which makes the same mapping at every
/// comparison.
pub fn median_in_place(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let keys: &mut [u32] = bytemuck::cast_slice_mut(values);
    for key in keys.iter_mut() {
        *key = order_key(*key);
    }
    let middle = keys.len() / 2;
    let even = keys.len().is_multiple_of(2);
    let (lower, median, _) = keys.select_nth_unstable(middle);
    let median = f32::from_bits(order_bits(*median));
    let lower_median = even.then(|| {
        let lower = lower
            .iter()
            .max()
            .expect("an even non-empty sample has a lower partition");
        f32::from_bits(order_bits(*lower))
    });
    for key in keys.iter_mut() {
        *key = order_bits(*key);
    }
    Some(match lower_median {
        Some(lower_median) => (lower_median + median) * 0.5,
        None => median,
    })
}

/// An f32's bits as an integer that orders as `f32::total_cmp` orders the
/// values: negative values have every bit flipped, the rest their sign bit.
fn order_key(bits: u32) -> u32 {
    bits ^ ((((bits as i32) >> 31) as u32) | 0x8000_0000)
}

/// The bits [`order_key`] took.
fn order_bits(key: u32) -> u32 {
    key ^ (((!key as i32 >> 31) as u32) | 0x8000_0000)
}

/// Median of an ascending-sorted sample. Returns `None` for an empty sample.
pub fn median_of_sorted(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

/// Median of an f64 sample in linear time, leaving the input unchanged.
/// Monotonic samples need no allocation; other samples are partially
/// selected on a copy. Returns `None` for an empty sample.
pub fn median_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(ordered_median_f64(values).unwrap_or_else(|| select_median_f64(&mut values.to_vec())))
}

/// The median when a non-empty sample is monotonic, without copying it.
fn ordered_median_f64(values: &[f64]) -> Option<f64> {
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    // Sorting already detects monotonic runs in linear time. Keep that fast
    // path: selection otherwise does extra partitioning on ordered samples.
    if values.is_sorted_by(|a, b| a.total_cmp(b).is_le()) {
        return Some(if even {
            (values[middle - 1] + values[middle]) * 0.5
        } else {
            values[middle]
        });
    }
    if values.is_sorted_by(|a, b| a.total_cmp(b).is_ge()) {
        return Some(if even {
            (values[middle] + values[middle - 1]) * 0.5
        } else {
            values[middle]
        });
    }
    None
}

/// Select both middle values of a non-empty sample with the same total
/// ordering and arithmetic as the sorted median.
fn select_median_f64(values: &mut [f64]) -> f64 {
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    let (lower, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if even {
        let lower = lower.iter().copied().max_by(f64::total_cmp).unwrap();
        (lower + upper) * 0.5
    } else {
        upper
    }
}

/// Robust sigma (MAD × [`NORMAL_MAD_SCALE`]) around `center`, overwriting
/// `values` with absolute deviations. Returns `None` for an empty sample.
pub fn robust_sigma_in_place(values: &mut [f32], center: f32) -> Option<f32> {
    for value in values.iter_mut() {
        *value = (*value - center).abs();
    }
    median_in_place(values).map(|mad| mad * NORMAL_MAD_SCALE_F32)
}

/// Robust sigma (MAD × [`NORMAL_MAD_SCALE`]) of an f64 sample around
/// `center`. Returns `None` for an empty sample.
pub fn robust_sigma_f64(values: &[f64], center: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut deviations: Vec<f64> = values.iter().map(|value| (*value - center).abs()).collect();
    let mad = ordered_median_f64(&deviations).unwrap_or_else(|| select_median_f64(&mut deviations));
    Some(mad * NORMAL_MAD_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_selection_matches_sorted_even_and_odd_samples() {
        let mut odd = vec![5.0_f32, 1.0, 3.0];
        assert_eq!(median_in_place(&mut odd), Some(3.0));
        let mut even = vec![4.0_f32, 1.0, 3.0, 2.0];
        assert_eq!(median_in_place(&mut even), Some(2.5));
        assert_eq!(median_in_place(&mut []), None);
    }

    #[test]
    fn sorted_and_f64_medians_share_the_averaging_convention() {
        assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0]), Some(2.0));
        assert_eq!(median_of_sorted(&[]), None);
        assert_eq!(median_f64(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median_f64(&[]), None);
    }

    #[test]
    fn robust_sigma_scales_the_mad() {
        let mut values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let sigma = robust_sigma_in_place(&mut values, 3.0).unwrap();
        assert!((sigma - 1.4826).abs() < 1.0e-6);
        let sigma = robust_sigma_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], 3.0).unwrap();
        assert!((sigma - 1.4826).abs() < 1.0e-12);
        assert_eq!(robust_sigma_f64(&[], 0.0), None);
    }

    /// Keep the old sorted implementation as an independent oracle, including
    /// signed zero, infinities and overflowing even medians. NaN payloads from
    /// arithmetic are unspecified; an odd median must preserve the selected bits.
    #[test]
    fn f64_statistics_preserve_sorted_results_and_input() {
        fn reference(values: &[f64]) -> Option<f64> {
            if values.is_empty() {
                return None;
            }
            let mut sorted = values.to_vec();
            sorted.sort_unstable_by(f64::total_cmp);
            let middle = sorted.len() / 2;
            Some(if sorted.len().is_multiple_of(2) {
                (sorted[middle - 1] + sorted[middle]) * 0.5
            } else {
                sorted[middle]
            })
        }

        fn check(values: &[f64]) {
            let original: Vec<_> = values.iter().map(|value| value.to_bits()).collect();
            let got = median_f64(values);
            let expected = reference(values);
            if values.len().is_multiple_of(2) && got.is_some_and(f64::is_nan) {
                assert!(expected.is_some_and(f64::is_nan));
            } else {
                assert_eq!(
                    got.map(f64::to_bits),
                    expected.map(f64::to_bits),
                    "median of {values:?}"
                );
            }
            for center in [0.0, -0.0, 1000.0, f64::INFINITY, f64::NAN] {
                let deviations: Vec<_> = values.iter().map(|v| (v - center).abs()).collect();
                let expected = reference(&deviations).map(|mad| mad * NORMAL_MAD_SCALE);
                let got = robust_sigma_f64(values, center);
                if got.is_some_and(f64::is_nan) {
                    assert!(expected.is_some_and(f64::is_nan));
                } else {
                    assert_eq!(
                        got.map(f64::to_bits),
                        expected.map(f64::to_bits),
                        "sigma around {center} of {values:?}"
                    );
                }
            }
            assert_eq!(
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original,
                "the borrowed input must stay unchanged"
            );
        }

        let special = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE / 4.0,
            -f64::MIN_POSITIVE / 4.0,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_0002),
            f64::from_bits(0x7ff0_0000_0000_0001),
        ];
        check(&[]);
        for &a in &special {
            check(&[a]);
            for &b in &special {
                check(&[a, b]);
                for &c in &special {
                    check(&[a, b, c]);
                }
            }
        }

        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for length in (1..40).chain([255, 256, 4096, 4097, 65536]) {
            let values: Vec<_> = (0..length).map(|_| f64::from_bits(next())).collect();
            check(&values);
            let mut sorted = values;
            sorted.sort_unstable_by(f64::total_cmp);
            check(&sorted);
            sorted.reverse();
            check(&sorted);
            let repeated: Vec<_> = (0..length).map(|_| (next() % 5) as f64).collect();
            check(&repeated);
        }
    }

    /// Every sample's median, and the partition it leaves, are those of
    /// selecting by `total_cmp`: signed zeros, infinities, NaN of either
    /// sign, subnormals and ties included.
    #[test]
    fn integer_selection_matches_total_cmp_selection() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let special = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::MIN_POSITIVE / 4.0,
            -f32::MIN_POSITIVE / 4.0,
            f32::MAX,
            f32::MIN,
        ];
        for length in (1..40).chain([255, 256, 4096, 4097]) {
            for _ in 0..20 {
                let values: Vec<f32> = (0..length)
                    .map(|_| match next() % 12 {
                        0 => special[(next() % special.len() as u64) as usize],
                        1..=3 => (next() % 5) as f32,
                        _ => f32::from_bits(next() as u32),
                    })
                    .collect();
                let mut expected = values.clone();
                let middle = length / 2;
                expected.select_nth_unstable_by(middle, f32::total_cmp);
                let reference = if length % 2 == 0 {
                    let lower = expected[..middle]
                        .iter()
                        .copied()
                        .max_by(f32::total_cmp)
                        .unwrap();
                    (lower + expected[middle]) * 0.5
                } else {
                    expected[middle]
                };
                let mut got = values.clone();
                let median = median_in_place(&mut got).unwrap();
                assert_eq!(median.to_bits(), reference.to_bits(), "{values:?}");
                let mut sorted_got: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
                let mut sorted_values: Vec<u32> =
                    values.iter().map(|value| value.to_bits()).collect();
                sorted_got.sort_unstable();
                sorted_values.sort_unstable();
                assert_eq!(sorted_got, sorted_values, "the same samples, reordered");
                let pivot = got[middle];
                assert!(
                    got[..middle]
                        .iter()
                        .all(|value| value.total_cmp(&pivot).is_le())
                );
                assert!(
                    got[middle..]
                        .iter()
                        .all(|value| value.total_cmp(&pivot).is_ge())
                );
            }
        }
        for bits in [
            0,
            1,
            0x7f80_0000,
            0x7fc0_0000,
            0x8000_0000,
            0xffff_ffff,
            0x3f80_0000,
        ] {
            assert_eq!(order_bits(order_key(bits)), bits);
        }
    }

    #[test]
    fn nan_sorts_high_and_does_not_poison_a_filtered_sample() {
        let mut values = vec![1.0_f32, f32::NAN, 2.0];
        assert_eq!(median_in_place(&mut values), Some(2.0));
    }
}
