use crate::{Error, LinearImage, ReferenceRegion, Result};
use rayon::prelude::*;
use std::str::FromStr;

/// How a composition is trimmed to the pixels every input channel covers.
///
/// Registering one filter stack onto another leaves `NaN` where the source
/// frame did not reach, so an uncropped composition keeps blank edges.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorCrop {
    /// Keep the full reference grid. Uncovered pixels stay `NaN`.
    #[default]
    None,
    /// Keep the bounding box of the covered pixels. A rotated overlap can
    /// still leave `NaN` corners inside that box.
    Bounds,
    /// Keep the largest rectangle in which every channel covers every pixel.
    ///
    /// This removes rotated corners, and is the mode to use when a later step
    /// cannot handle `NaN`. An uncovered pixel in the middle of the frame
    /// bounds the result just as an edge does.
    Inscribed,
}

impl ColorCrop {
    /// Lowercase name, for example `inscribed`.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bounds => "bounds",
            Self::Inscribed => "inscribed",
        }
    }
}

impl FromStr for ColorCrop {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "bounds" => Ok(Self::Bounds),
            "inscribed" => Ok(Self::Inscribed),
            _ => Err(Error::Color(format!("unknown crop mode {value}"))),
        }
    }
}

/// What one input channel covers of the shared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelCoverage {
    /// Caller's name for the channel, such as `green` or `OIII`.
    pub name: String,
    /// Bounding box of the pixels this channel covers.
    pub region: ReferenceRegion,
    /// Number of pixels this channel covers.
    pub covered_pixels: usize,
    /// Horizontal offset of this channel's coverage center from the other
    /// channels' consensus center, in pixels.
    pub center_offset_x: f64,
    /// Vertical offset of this channel's coverage center from the other
    /// channels' consensus center, in pixels.
    pub center_offset_y: f64,
    /// Whether that offset is large enough to look like a pointing error
    /// rather than ordinary dither and guiding drift.
    pub off_center: bool,
}

impl ChannelCoverage {
    /// Distance between this channel's coverage center and the consensus.
    pub fn center_offset_pixels(&self) -> f64 {
        self.center_offset_x.hypot(self.center_offset_y)
    }
}

/// The region a crop keeps, and what each channel contributed to it.
#[derive(Clone, Debug, PartialEq)]
pub struct CropReport {
    /// Width of the shared input grid.
    pub grid_width: usize,
    /// Height of the shared input grid.
    pub grid_height: usize,
    /// Region the requested mode keeps.
    pub region: ReferenceRegion,
    /// One entry per input channel, in the order they were supplied.
    pub channels: Vec<ChannelCoverage>,
}

impl CropReport {
    /// Pixel floor for flagging a channel as off center.
    pub const OFF_CENTER_PIXELS: f64 = 32.0;
    /// Fraction of the larger grid dimension for flagging a channel as off
    /// center.
    pub const OFF_CENTER_FRACTION: f64 = 0.02;

    /// Offset at which a channel is flagged, for a grid of the given size: the
    /// larger of the pixel floor and the fractional bound.
    pub fn off_center_limit_pixels(width: usize, height: usize) -> f64 {
        Self::OFF_CENTER_PIXELS.max(width.max(height) as f64 * Self::OFF_CENTER_FRACTION)
    }

    /// Share of the input grid the kept region covers.
    pub fn retained_fraction(&self) -> f64 {
        let grid = self.grid_width * self.grid_height;
        if grid == 0 {
            return 0.0;
        }
        (self.region.width * self.region.height) as f64 / grid as f64
    }

    /// Channels whose coverage sits far from where the others agree. Each one
    /// is a candidate for the frame that shrank the crop.
    pub fn off_center(&self) -> impl Iterator<Item = &ChannelCoverage> {
        self.channels.iter().filter(|channel| channel.off_center)
    }
}

/// The region of the shared pixel grid that `crop` keeps for these channels,
/// with per-channel coverage diagnostics.
///
/// A pixel is covered when every sample of every channel at that pixel is
/// finite, so the kept region is the inner area common to all of them. All
/// channels must already share one grid.
pub fn crop_report(channels: &[(&str, &LinearImage)], crop: ColorCrop) -> Result<CropReport> {
    let images = channels.iter().map(|(_, image)| *image).collect::<Vec<_>>();
    let reference = validate_grid(&images)?;
    let mut coverage = Vec::with_capacity(channels.len());
    for (name, image) in channels {
        let (region, covered_pixels) = channel_coverage(image)
            .ok_or_else(|| Error::Color(format!("{name} covers no pixel of the grid")))?;
        coverage.push(ChannelCoverage {
            name: (*name).to_owned(),
            region,
            covered_pixels,
            center_offset_x: 0.0,
            center_offset_y: 0.0,
            off_center: false,
        });
    }
    flag_off_center(&mut coverage, reference.width, reference.height);
    Ok(CropReport {
        grid_width: reference.width,
        grid_height: reference.height,
        region: region_of(&images, reference, crop)?,
        channels: coverage,
    })
}

/// The region of the shared pixel grid that `crop` keeps, without the
/// per-channel diagnostics of [`crop_report`].
pub fn covered_region(channels: &[&LinearImage], crop: ColorCrop) -> Result<ReferenceRegion> {
    let reference = validate_grid(channels)?;
    region_of(channels, reference, crop)
}

fn validate_grid(channels: &[&LinearImage]) -> Result<ReferenceRegion> {
    let reference = *channels
        .first()
        .ok_or_else(|| Error::Color("at least one channel is required".into()))?;
    for channel in channels {
        if channel.width != reference.width || channel.height != reference.height {
            return Err(Error::Color(format!(
                "channel dimensions {}x{} do not match {}x{}",
                channel.width, channel.height, reference.width, reference.height
            )));
        }
    }
    Ok(ReferenceRegion {
        x: 0,
        y: 0,
        width: reference.width,
        height: reference.height,
    })
}

fn region_of(
    channels: &[&LinearImage],
    grid: ReferenceRegion,
    crop: ColorCrop,
) -> Result<ReferenceRegion> {
    if crop == ColorCrop::None {
        return Ok(grid);
    }
    let covered = coverage(channels, grid.width * grid.height);
    let region = match crop {
        ColorCrop::None => Some(grid),
        ColorCrop::Bounds => bounding_rectangle(&covered, grid.width, grid.height),
        ColorCrop::Inscribed => inscribed_rectangle(&covered, grid.width, grid.height),
    };
    region.ok_or_else(|| Error::Color("no pixel is covered by every channel".into()))
}

fn coverage(channels: &[&LinearImage], pixel_count: usize) -> Vec<bool> {
    (0..pixel_count)
        .into_par_iter()
        .map(|pixel| channels.iter().all(|channel| pixel_covered(channel, pixel)))
        .collect()
}

fn pixel_covered(image: &LinearImage, pixel: usize) -> bool {
    let start = pixel * image.channels;
    image.data[start..start + image.channels]
        .iter()
        .all(|sample| sample.is_finite())
}

/// Bounding box and pixel count of one channel's own coverage.
fn channel_coverage(image: &LinearImage) -> Option<(ReferenceRegion, usize)> {
    let mut left = image.width;
    let mut right = 0;
    let mut top = image.height;
    let mut bottom = 0;
    let mut covered_pixels = 0;
    for row in 0..image.height {
        for column in 0..image.width {
            if pixel_covered(image, row * image.width + column) {
                covered_pixels += 1;
                left = left.min(column);
                right = right.max(column);
                top = top.min(row);
                bottom = bottom.max(row);
            }
        }
    }
    (covered_pixels > 0).then(|| {
        (
            ReferenceRegion {
                x: left,
                y: top,
                width: right - left + 1,
                height: bottom - top + 1,
            },
            covered_pixels,
        )
    })
}

/// Offset each channel's coverage center from the median of the others, and
/// flag the ones that sit far enough out to look like a pointing error.
///
/// The median holds the consensus even when one channel is badly placed, which
/// a mean would not.
fn flag_off_center(channels: &mut [ChannelCoverage], width: usize, height: usize) {
    if channels.len() < 3 {
        // Two channels disagree with each other symmetrically; there is no
        // majority to be the odd one out.
        return;
    }
    let limit = CropReport::off_center_limit_pixels(width, height);
    let centers = channels
        .iter()
        .map(|channel| center_of(channel.region))
        .collect::<Vec<_>>();
    for (index, channel) in channels.iter_mut().enumerate() {
        let others = centers
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, center)| *center)
            .collect::<Vec<_>>();
        let consensus = (
            median(others.iter().map(|center| center.0)),
            median(others.iter().map(|center| center.1)),
        );
        channel.center_offset_x = centers[index].0 - consensus.0;
        channel.center_offset_y = centers[index].1 - consensus.1;
        channel.off_center = channel.center_offset_pixels() > limit;
    }
}

fn center_of(region: ReferenceRegion) -> (f64, f64) {
    (
        region.x as f64 + (region.width as f64 - 1.0) / 2.0,
        region.y as f64 + (region.height as f64 - 1.0) / 2.0,
    )
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn bounding_rectangle(covered: &[bool], width: usize, height: usize) -> Option<ReferenceRegion> {
    let mut left = width;
    let mut right = 0;
    let mut top = height;
    let mut bottom = 0;
    for row in 0..height {
        for column in 0..width {
            if covered[row * width + column] {
                left = left.min(column);
                right = right.max(column);
                top = top.min(row);
                bottom = bottom.max(row);
            }
        }
    }
    (left <= right && top <= bottom).then(|| ReferenceRegion {
        x: left,
        y: top,
        width: right - left + 1,
        height: bottom - top + 1,
    })
}

/// Largest covered axis-aligned rectangle, by the usual per-row histogram
/// scan. Ties keep the topmost, then leftmost, rectangle.
fn inscribed_rectangle(covered: &[bool], width: usize, height: usize) -> Option<ReferenceRegion> {
    let mut columns = vec![0usize; width];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut best: Option<ReferenceRegion> = None;
    let mut best_area = 0;
    for row in 0..height {
        for (column, run) in columns.iter_mut().enumerate() {
            *run = if covered[row * width + column] {
                *run + 1
            } else {
                0
            };
        }
        stack.clear();
        for column in 0..=width {
            let run = columns.get(column).copied().unwrap_or(0);
            let mut start = column;
            while let Some(&(open_column, open_run)) = stack.last() {
                if open_run <= run {
                    break;
                }
                stack.pop();
                let area = open_run * (column - open_column);
                if area > best_area {
                    best_area = area;
                    best = Some(ReferenceRegion {
                        x: open_column,
                        y: row + 1 - open_run,
                        width: column - open_column,
                        height: open_run,
                    });
                }
                start = open_column;
            }
            if run > 0 && stack.last().is_none_or(|(_, open_run)| *open_run < run) {
                stack.push((start, run));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(width: usize, height: usize, rows: &[&str]) -> LinearImage {
        let data = rows
            .iter()
            .flat_map(|row| {
                row.chars()
                    .map(|cell| if cell == '#' { 1.0 } else { f32::NAN })
            })
            .collect::<Vec<_>>();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    /// A channel covering everything but a `shift` border on one side.
    fn shifted(size: usize, shift_x: isize, shift_y: isize) -> LinearImage {
        let data = (0..size * size)
            .map(|index| {
                let column = (index % size) as isize;
                let row = (index / size) as isize;
                let inside = column >= shift_x.max(0)
                    && column < size as isize + shift_x.min(0)
                    && row >= shift_y.max(0)
                    && row < size as isize + shift_y.min(0);
                if inside { 1.0 } else { f32::NAN }
            })
            .collect::<Vec<_>>();
        LinearImage::new(size, size, 1, data).unwrap()
    }

    #[test]
    fn none_keeps_the_whole_grid() {
        let image = mask(3, 2, &[".#.", "..."]);
        let region = covered_region(&[&image], ColorCrop::None).unwrap();
        assert_eq!(
            region,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 3,
                height: 2
            }
        );
    }

    #[test]
    fn bounds_trims_an_uncovered_border() {
        let image = mask(4, 4, &["....", ".##.", ".##.", "...."]);
        let region = covered_region(&[&image], ColorCrop::Bounds).unwrap();
        assert_eq!(
            region,
            ReferenceRegion {
                x: 1,
                y: 1,
                width: 2,
                height: 2
            }
        );
    }

    #[test]
    fn bounds_keeps_uncovered_corners_that_inscribed_removes() {
        let image = mask(4, 3, &[".###", "####", "####"]);
        assert_eq!(
            covered_region(&[&image], ColorCrop::Bounds).unwrap(),
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 4,
                height: 3
            }
        );
        assert_eq!(
            covered_region(&[&image], ColorCrop::Inscribed).unwrap(),
            ReferenceRegion {
                x: 1,
                y: 0,
                width: 3,
                height: 3
            }
        );
    }

    #[test]
    fn inscribed_prefers_area_over_the_widest_run() {
        // The full-width top row is only 5x1; the block on the right is 3x4.
        let image = mask(5, 4, &["#####", "..###", "..###", "..###"]);
        assert_eq!(
            covered_region(&[&image], ColorCrop::Inscribed).unwrap(),
            ReferenceRegion {
                x: 2,
                y: 0,
                width: 3,
                height: 4
            }
        );
    }

    #[test]
    fn inscribed_matches_a_brute_force_search() {
        let width = 11;
        let height = 9;
        for seed in 0..24u64 {
            let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let data = (0..width * height)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    if state >> 62 == 0 { f32::NAN } else { 1.0 }
                })
                .collect::<Vec<_>>();
            let covered = data
                .iter()
                .map(|value| value.is_finite())
                .collect::<Vec<_>>();
            let found = inscribed_rectangle(&covered, width, height);
            let mut best = 0;
            for top in 0..height {
                for left in 0..width {
                    for bottom in top..height {
                        for right in left..width {
                            let filled = (top..=bottom).all(|row| {
                                (left..=right).all(|column| covered[row * width + column])
                            });
                            if filled {
                                best = best.max((bottom - top + 1) * (right - left + 1));
                            }
                        }
                    }
                }
            }
            let area = found.map_or(0, |region| region.width * region.height);
            assert_eq!(area, best, "seed {seed}");
            if let Some(region) = found {
                for row in region.y..region.y + region.height {
                    for column in region.x..region.x + region.width {
                        assert!(covered[row * width + column], "seed {seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn coverage_intersects_every_channel() {
        let first = mask(3, 1, &["##."]);
        let second = mask(3, 1, &[".##"]);
        assert_eq!(
            covered_region(&[&first, &second], ColorCrop::Inscribed).unwrap(),
            ReferenceRegion {
                x: 1,
                y: 0,
                width: 1,
                height: 1
            }
        );
    }

    #[test]
    fn an_rgb_channel_needs_every_plane_finite() {
        let image = LinearImage::new(2, 1, 3, vec![1.0, 1.0, 1.0, 1.0, f32::NAN, 1.0]).unwrap();
        assert_eq!(
            covered_region(&[&image], ColorCrop::Bounds).unwrap(),
            ReferenceRegion {
                x: 0,
                y: 0,
                width: 1,
                height: 1
            }
        );
    }

    #[test]
    fn a_fully_uncovered_grid_is_rejected() {
        let image = mask(2, 2, &["..", ".."]);
        let error = covered_region(&[&image], ColorCrop::Inscribed).unwrap_err();
        assert!(error.to_string().contains("no pixel is covered"));
    }

    #[test]
    fn mismatched_channel_dimensions_are_rejected() {
        let first = mask(2, 1, &["##"]);
        let second = mask(3, 1, &["###"]);
        let error = covered_region(&[&first, &second], ColorCrop::Bounds).unwrap_err();
        assert!(error.to_string().contains("do not match"));
    }

    #[test]
    fn crop_modes_round_trip_through_their_names() {
        for mode in [ColorCrop::None, ColorCrop::Bounds, ColorCrop::Inscribed] {
            assert_eq!(mode.name().parse::<ColorCrop>().unwrap(), mode);
        }
        assert!("largest".parse::<ColorCrop>().is_err());
    }

    #[test]
    fn a_report_measures_each_channels_own_coverage() {
        let reference = shifted(64, 0, 0);
        let green = shifted(64, 3, -2);
        let blue = shifted(64, -4, 1);
        let report = crop_report(
            &[("red", &reference), ("green", &green), ("blue", &blue)],
            ColorCrop::Inscribed,
        )
        .unwrap();
        assert_eq!((report.grid_width, report.grid_height), (64, 64));
        assert_eq!(
            report.region,
            ReferenceRegion {
                x: 3,
                y: 1,
                width: 57,
                height: 61
            }
        );
        assert!(report.retained_fraction() > 0.8);
        assert_eq!(report.channels[1].name, "green");
        assert_eq!(report.channels[1].covered_pixels, 61 * 62);
        assert!(report.off_center().next().is_none());
    }

    #[test]
    fn a_channel_far_from_the_others_is_flagged() {
        let reference = shifted(256, 0, 0);
        let green = shifted(256, 2, 0);
        let stray = shifted(256, 0, 90);
        let report = crop_report(
            &[("red", &reference), ("green", &green), ("blue", &stray)],
            ColorCrop::Inscribed,
        )
        .unwrap();
        let flagged = report.off_center().collect::<Vec<_>>();
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].name, "blue");
        assert!(flagged[0].center_offset_y > 40.0);
        assert!(!report.channels[0].off_center);
        assert!(!report.channels[1].off_center);
    }

    #[test]
    fn ordinary_dither_is_not_flagged() {
        let channels = [
            shifted(256, 0, 0),
            shifted(256, 4, -3),
            shifted(256, -5, 2),
            shifted(256, 2, 6),
        ];
        let named = channels
            .iter()
            .zip(["luminance", "red", "green", "blue"])
            .map(|(image, name)| (name, image))
            .collect::<Vec<_>>();
        let report = crop_report(&named, ColorCrop::Inscribed).unwrap();
        assert!(report.off_center().next().is_none());
        assert!(report.retained_fraction() > 0.9);
    }

    #[test]
    fn two_channels_never_flag_each_other() {
        let reference = shifted(256, 0, 0);
        let stray = shifted(256, 0, 90);
        let report = crop_report(
            &[("H-alpha", &reference), ("OIII", &stray)],
            ColorCrop::Inscribed,
        )
        .unwrap();
        assert!(report.off_center().next().is_none());
    }

    #[test]
    fn a_channel_covering_nothing_is_named_in_the_error() {
        let reference = mask(2, 1, &["##"]);
        let empty = mask(2, 1, &[".."]);
        let error = crop_report(
            &[("H-alpha", &reference), ("OIII", &empty)],
            ColorCrop::Bounds,
        )
        .unwrap_err();
        assert!(error.to_string().contains("OIII covers no pixel"));
    }
}
