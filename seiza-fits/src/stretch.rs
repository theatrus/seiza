//! Midtone-transfer-function autostretch, matching N.I.N.A.'s algorithm
//! (and therefore PixInsight's STF family): shadows are clipped at
//! `shadows_clip` MADs below the median and the midtone balance is chosen
//! so the median lands at `target_median`.

use crate::stats::Statistics;

#[derive(Debug, Clone)]
pub struct StretchParams {
    /// Where the median should land in the output (N.I.N.A. default 0.2)
    pub target_median: f64,
    /// Shadow clipping point in MADs relative to the median (default -2.8)
    pub shadows_clip: f64,
}

impl Default for StretchParams {
    fn default() -> Self {
        Self {
            target_median: 0.2,
            shadows_clip: -2.8,
        }
    }
}

/// The PixInsight/N.I.N.A. midtones transfer function.
pub fn midtones_transfer_function(midtone: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    (midtone - 1.0) * x / ((2.0 * midtone - 1.0) * x - midtone)
}

/// Stretch 16-bit data directly to 8-bit output via a 65536-entry lookup
/// table computed once from the statistics.
pub fn stretch_u16_to_u8(data: &[u16], stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let map = stretch_map(stats, params);
    data.iter().map(|&v| map[v as usize]).collect()
}

fn stretch_map(stats: &Statistics, params: &StretchParams) -> Vec<u8> {
    let normalized_median = stats.median as f64 / 65535.0;
    // 1.4826 converts MAD to a normal-equivalent sigma
    let normalized_mad = stats.mad / 65535.0 * 1.4826;

    let (shadows, midtone, highlights) = if normalized_median > 0.5 {
        // Inverted or overexposed frame
        let highlights = normalized_median - params.shadows_clip * normalized_mad;
        let midtone = midtones_transfer_function(
            params.target_median,
            1.0 - (highlights - normalized_median),
        );
        (0.0, midtone, highlights)
    } else {
        let shadows = (normalized_median + params.shadows_clip * normalized_mad).max(0.0);
        let midtone = midtones_transfer_function(params.target_median, normalized_median - shadows);
        (shadows, midtone, 1.0)
    };

    (0..65536usize)
        .map(|v| {
            let value = v as f64 / 65535.0;
            let input = 1.0 - highlights + value - shadows;
            let stretched = midtones_transfer_function(midtone, input);
            (stretched.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtf_boundaries() {
        assert_eq!(midtones_transfer_function(0.5, 0.0), 0.0);
        assert_eq!(midtones_transfer_function(0.5, 1.0), 1.0);
        assert!((midtones_transfer_function(0.5, 0.5) - 0.5).abs() < 1e-12);
        assert!(midtones_transfer_function(0.25, 0.25) > 0.45);
    }

    #[test]
    fn stretch_pushes_median_toward_target() {
        // Synthetic sub: background ~600 with spread, a few bright pixels
        let mut data = vec![600u16; 100_000];
        for (i, v) in data.iter_mut().enumerate() {
            *v += ((i * 37) % 41) as u16;
        }
        data[0] = 60000;
        data[1] = 55000;

        let stats = crate::stats::statistics_u16(&data);
        let out = stretch_u16_to_u8(&data, &stats, &StretchParams::default());

        // Median output should land near 0.2 * 255 ≈ 51
        let mut sorted = out.clone();
        sorted.sort_unstable();
        let median_out = sorted[sorted.len() / 2] as f64;
        assert!(
            (median_out - 0.2 * 255.0).abs() < 16.0,
            "median landed at {median_out}"
        );
        // Bright pixels stay bright, background stays dark
        assert!(out[0] > 200);
        assert!(sorted[100] < 60);
    }
}
