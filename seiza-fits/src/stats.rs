//! Histogram-based image statistics: exact median and MAD in a single
//! O(n + 65536) pass instead of sorting the pixel array.

#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    pub min: u16,
    pub max: u16,
    pub mean: f64,
    pub median: u16,
    /// Median absolute deviation (exact, from the histogram)
    pub mad: f64,
    pub count: usize,
}

pub fn statistics_u16(data: &[u16]) -> Statistics {
    let mut histogram = vec![0u32; 65536];
    let mut sum = 0u64;
    for &v in data {
        histogram[v as usize] += 1;
        sum += v as u64;
    }
    let count = data.len();
    if count == 0 {
        return Statistics {
            min: 0,
            max: 0,
            mean: 0.0,
            median: 0,
            mad: 0.0,
            count: 0,
        };
    }

    let min = histogram.iter().position(|&c| c > 0).unwrap_or(0) as u16;
    let max = (65535 - histogram.iter().rev().position(|&c| c > 0).unwrap_or(0)) as u16;

    // Median: first bin where the cumulative count crosses half
    let half = count.div_ceil(2) as u64;
    let mut cumulative = 0u64;
    let mut median = 0u16;
    for (value, &bin_count) in histogram.iter().enumerate() {
        cumulative += bin_count as u64;
        if cumulative >= half {
            median = value as u16;
            break;
        }
    }

    // MAD: expand a window around the median until it holds half the pixels;
    // the deviation d at which that happens is the median absolute deviation
    let median_idx = median as usize;
    let mut inside = histogram[median_idx] as u64;
    let mut mad = 0u32;
    while inside < half && mad < 65535 {
        mad += 1;
        if let Some(&c) = histogram.get(median_idx + mad as usize) {
            inside += c as u64;
        }
        if mad as usize <= median_idx {
            inside += histogram[median_idx - mad as usize] as u64;
        }
    }

    Statistics {
        min,
        max,
        mean: sum as f64 / count as f64,
        median,
        mad: mad as f64,
        count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_sorted_reference_on_pseudo_random_data() {
        let mut state = 0xABCDEF12345u64;
        let data: Vec<u16> = (0..100_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Skewed like a real sub: mostly low background
                ((state >> 40) as u16) / 3 + 500
            })
            .collect();

        let stats = statistics_u16(&data);

        let mut sorted = data.clone();
        sorted.sort_unstable();
        let expected_median = sorted[sorted.len().div_ceil(2) - 1];
        assert_eq!(stats.median, expected_median);

        let mut deviations: Vec<u16> = data
            .iter()
            .map(|&v| (v as i32 - expected_median as i32).unsigned_abs() as u16)
            .collect();
        deviations.sort_unstable();
        let expected_mad = deviations[deviations.len().div_ceil(2) - 1];
        assert!(
            (stats.mad - expected_mad as f64).abs() <= 1.0,
            "{} vs {expected_mad}",
            stats.mad
        );

        assert_eq!(stats.min, sorted[0]);
        assert_eq!(stats.max, *sorted.last().unwrap());
        assert_eq!(stats.count, 100_000);
    }

    #[test]
    fn empty_input() {
        let stats = statistics_u16(&[]);
        assert_eq!(stats.count, 0);
    }
}
