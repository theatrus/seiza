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
pub fn median_in_place(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    let even = values.len().is_multiple_of(2);
    let (lower, median, _) =
        values.select_nth_unstable_by(middle, |left, right| left.total_cmp(right));
    Some(if even {
        let lower_median = lower
            .iter()
            .max_by(|left, right| left.total_cmp(right))
            .expect("an even non-empty sample has a lower partition");
        (*lower_median + *median) * 0.5
    } else {
        *median
    })
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

    #[test]
    fn nan_sorts_high_and_does_not_poison_a_filtered_sample() {
        let mut values = vec![1.0_f32, f32::NAN, 2.0];
        assert_eq!(median_in_place(&mut values), Some(2.0));
    }
}
