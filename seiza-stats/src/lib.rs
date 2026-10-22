//! Robust statistics shared across the seiza workspace.
//!
//! One median convention throughout: an even-length sample averages its two
//! middle elements. Ordering uses `total_cmp`, so NaN sorts high; callers
//! that must exclude non-finite values filter before calling.

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

/// Median of an f64 sample, sorting a copy. Returns `None` for an empty
/// sample.
pub fn median_f64(values: &[f64]) -> Option<f64> {
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
    let deviations: Vec<f64> = values.iter().map(|value| (*value - center).abs()).collect();
    median_f64(&deviations).map(|mad| mad * NORMAL_MAD_SCALE)
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
