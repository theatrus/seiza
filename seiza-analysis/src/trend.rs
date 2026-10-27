use crate::{DepthPoint, NoiseTrend};

pub(crate) fn fit(curve: &[DepthPoint]) -> Option<NoiseTrend> {
    let last = curve.iter().rev().find(|p| p.noise.is_some())?.frame_count;
    let mut next = 4;
    let points: Vec<_> = curve
        .iter()
        .filter(|p| {
            if p.frame_count < next || p.noise.is_none() {
                return false;
            }
            next = ((p.frame_count as f64 * 1.25).ceil() as usize).next_multiple_of(2);
            p.frame_count * 4 >= last && p.noise.is_some_and(|v| v > 0.0 && v.is_finite())
        })
        .collect();
    let first = points.first()?.frame_count;
    let last = points.last()?.frame_count;
    if points.len() < 5 || last < first * 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points
        .iter()
        .map(|p| (p.frame_count as f64).ln())
        .sum::<f64>()
        / n;
    let mean_y = points.iter().map(|p| p.noise.unwrap().ln()).sum::<f64>() / n;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for p in &points {
        let x = (p.frame_count as f64).ln() - mean_x;
        let y = p.noise.unwrap().ln() - mean_y;
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }
    if xx <= 0.0 {
        return None;
    }
    let exponent = xy / xx;
    let r_squared = if yy > 1e-20 {
        (xy * xy / (xx * yy)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some(NoiseTrend {
        exponent,
        r_squared,
        checkpoints: points.len(),
        first_frame_count: first,
        last_frame_count: last,
        conditional_double_depth_noise_factor: ((-1.0..0.0).contains(&exponent)
            && r_squared >= 0.8)
            .then(|| 2f64.powf(exponent)),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn curve(exponent: f64, n: usize) -> Vec<DepthPoint> {
        (1..=n)
            .map(|n| DepthPoint {
                frame_index: n - 1,
                frame_count: n,
                integration_seconds: n as f64,
                unknown_exposures: 0,
                noise: (n % 2 == 0).then(|| 10.0 * (n as f64).powf(exponent)),
                ideal_noise: None,
            })
            .collect()
    }
    #[test]
    fn independent_noise_has_minus_half_exponent() {
        let trend = fit(&curve(-0.5, 128)).unwrap();
        assert!((trend.exponent + 0.5).abs() < 1e-12);
        assert!(
            (trend.conditional_double_depth_noise_factor.unwrap() - 0.5f64.sqrt()).abs() < 1e-12
        );
        assert!(trend.first_frame_count >= 32);
    }
    #[test]
    fn insufficient_or_flat_or_rising_data_cannot_promise_improvement() {
        assert!(fit(&curve(-0.5, 8)).is_none());
        assert!(
            fit(&curve(0.0, 128))
                .unwrap()
                .conditional_double_depth_noise_factor
                .is_none()
        );
        assert!(
            fit(&curve(0.2, 128))
                .unwrap()
                .conditional_double_depth_noise_factor
                .is_none()
        );
    }
}
