//! Medians as the harness takes them: the mean of the two middle values of an
//! even count, in the values' own precision. `None` when there are no values.

macro_rules! median {
    ($name:ident, $type:ty) => {
        pub(crate) fn $name(values: &mut [$type]) -> Option<$type> {
            let count = values.len();
            if count == 0 {
                return None;
            }
            let (below, &mut upper, _) =
                values.select_nth_unstable_by(count / 2, |a, b| a.total_cmp(b));
            if count % 2 == 1 {
                return Some(upper);
            }
            let lower = below
                .iter()
                .copied()
                .fold(<$type>::NEG_INFINITY, <$type>::max);
            Some((lower + upper) / 2.0)
        }
    };
}

median!(median_f32, f32);
median!(median_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_median_is_the_middle_value_or_the_mean_of_the_two() {
        assert_eq!(median_f64(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median_f64(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median_f32(&mut [5.0]), Some(5.0));
        assert_eq!(median_f32(&mut []), None);
    }
}
