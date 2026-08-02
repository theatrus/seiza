use crate::{AffineTransform, Error, LinearImage, Result, resample_to_reference_affine};
use seiza::{FitsCardValue, Wcs};
use seiza_fits::{HeaderValue, WriteHeaderCard};

/// Version of the canonical display-orientation geometry and sampling rules.
pub const SKY_ORIENTATION_VERSION: u32 = 1;

/// The canonical celestial display convention used by Seiza stack products.
pub const SKY_ORIENTATION_NAME: &str = "north_up_east_left";

/// A bounded affine reprojection from one solved linear TAN grid to a
/// north-up, east-left grid with square pixels at the source's geometric-mean
/// scale.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyOrientationPlan {
    source_width: usize,
    source_height: usize,
    output_width: usize,
    output_height: usize,
    source_to_output: AffineTransform,
    output_wcs: Wcs,
}

impl SkyOrientationPlan {
    /// Build a full-footprint sky-up plan for one image grid.
    ///
    /// The current affine path accepts an undistorted TAN WCS. SIP must be
    /// removed by a true celestial reprojection before this operation.
    pub fn new(source_width: usize, source_height: usize, source_wcs: &Wcs) -> Result<Self> {
        if source_width == 0 || source_height == 0 {
            return Err(Error::Orientation(
                "sky orientation needs non-zero source dimensions".into(),
            ));
        }
        if source_wcs.sip.is_some() {
            return Err(Error::Orientation(
                "affine sky orientation does not accept SIP distortion".into(),
            ));
        }
        validate_wcs(source_wcs)?;

        let scale_degrees = source_wcs.scale_arcsec_per_px() / 3600.0;
        if !scale_degrees.is_finite() || scale_degrees <= 0.0 {
            return Err(Error::Orientation(
                "sky orientation needs a finite positive pixel scale".into(),
            ));
        }
        // Seiza's zero-rotation, unflipped WCS maps north toward decreasing Y
        // and east toward decreasing X.
        let canonical_cd = [[-scale_degrees, 0.0], [0.0, -scale_degrees]];
        let inverse_canonical = [[-1.0 / scale_degrees, 0.0], [0.0, -1.0 / scale_degrees]];
        let matrix = multiply_2x2(inverse_canonical, source_wcs.cd);

        let corners = [
            (0.0, 0.0),
            ((source_width - 1) as f64, 0.0),
            (0.0, (source_height - 1) as f64),
            ((source_width - 1) as f64, (source_height - 1) as f64),
        ];
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (x, y) in corners {
            let dx = x - source_wcs.crpix.0;
            let dy = y - source_wcs.crpix.1;
            let mapped_x = matrix[0][0].mul_add(dx, matrix[0][1] * dy);
            let mapped_y = matrix[1][0].mul_add(dx, matrix[1][1] * dy);
            min_x = min_x.min(mapped_x);
            max_x = max_x.max(mapped_x);
            min_y = min_y.min(mapped_y);
            max_y = max_y.max(mapped_y);
        }
        let output_width = bounded_extent(max_x - min_x, source_width, source_height)?;
        let output_height = bounded_extent(max_y - min_y, source_width, source_height)?;
        let output_crpix = (-min_x, -min_y);
        let source_to_output = AffineTransform {
            matrix,
            translation_x: output_crpix.0
                - matrix[0][0].mul_add(source_wcs.crpix.0, matrix[0][1] * source_wcs.crpix.1),
            translation_y: output_crpix.1
                - matrix[1][0].mul_add(source_wcs.crpix.0, matrix[1][1] * source_wcs.crpix.1),
        };
        source_to_output.validate()?;
        let output_pixels = output_width
            .checked_mul(output_height)
            .ok_or_else(|| Error::Orientation("sky-oriented dimensions overflow".into()))?;
        let source_pixels = source_width
            .checked_mul(source_height)
            .ok_or_else(|| Error::Orientation("source dimensions overflow".into()))?;
        if output_pixels > source_pixels.saturating_mul(4) {
            return Err(Error::Orientation(format!(
                "sky-oriented grid {output_width}x{output_height} is implausibly large for {source_width}x{source_height} input"
            )));
        }

        Ok(Self {
            source_width,
            source_height,
            output_width,
            output_height,
            source_to_output,
            output_wcs: Wcs {
                crval: source_wcs.crval,
                crpix: output_crpix,
                cd: canonical_cd,
                sip: None,
            },
        })
    }

    /// Resample one linear mono or RGB image onto this plan's sky-up grid.
    pub fn apply(&self, source: &LinearImage) -> Result<LinearImage> {
        if source.width != self.source_width || source.height != self.source_height {
            return Err(Error::Orientation(format!(
                "sky-orientation source is {}x{}; expected {}x{}",
                source.width, source.height, self.source_width, self.source_height
            )));
        }
        if self.output_width == self.source_width
            && self.output_height == self.source_height
            && self.source_to_output == AffineTransform::IDENTITY
        {
            return Ok(source.clone());
        }
        resample_to_reference_affine(
            source,
            self.output_width,
            self.output_height,
            self.source_to_output,
        )
    }

    /// Width of the full-footprint sky-up grid.
    pub const fn output_width(&self) -> usize {
        self.output_width
    }

    /// Height of the full-footprint sky-up grid.
    pub const fn output_height(&self) -> usize {
        self.output_height
    }

    /// Source-grid to sky-up-grid affine mapping.
    pub const fn source_to_output(&self) -> AffineTransform {
        self.source_to_output
    }

    /// Canonical north-up, east-left output WCS.
    pub const fn output_wcs(&self) -> &Wcs {
        &self.output_wcs
    }

    /// FITS cards that replace any source WCS on the reprojected image.
    pub fn fits_header_cards(&self) -> Vec<WriteHeaderCard> {
        let mut cards = self
            .output_wcs
            .fits_header_cards()
            .into_iter()
            .map(|(keyword, value)| {
                let value = match value {
                    FitsCardValue::Text(value) => HeaderValue::String(value.into()),
                    FitsCardValue::Number(value) => HeaderValue::Float(value),
                    FitsCardValue::Integer(value) => HeaderValue::Integer(i64::from(value)),
                };
                WriteHeaderCard::new(keyword, value).with_comment("canonical sky-up WCS")
            })
            .collect::<Vec<_>>();
        cards.push(
            WriteHeaderCard::new("RADESYS", HeaderValue::String("ICRS".into()))
                .with_comment("celestial reference frame"),
        );
        cards.push(
            WriteHeaderCard::new("SKYORIEN", HeaderValue::String("N-UP E-LEFT".into()))
                .with_comment("display orientation"),
        );
        cards
    }
}

fn validate_wcs(wcs: &Wcs) -> Result<()> {
    let determinant = wcs.cd[0][0].mul_add(wcs.cd[1][1], -wcs.cd[0][1] * wcs.cd[1][0]);
    if !wcs.crval.0.is_finite()
        || !wcs.crval.1.is_finite()
        || !(0.0..=360.0).contains(&wcs.crval.0)
        || !(-90.0..=90.0).contains(&wcs.crval.1)
        || !wcs.crpix.0.is_finite()
        || !wcs.crpix.1.is_finite()
        || wcs.cd.into_iter().flatten().any(|value| !value.is_finite())
        || !determinant.is_finite()
        || determinant.abs() <= 1.0e-15
    {
        return Err(Error::Orientation(
            "sky orientation needs a finite invertible WCS".into(),
        ));
    }
    Ok(())
}

fn bounded_extent(span: f64, source_width: usize, source_height: usize) -> Result<usize> {
    if !span.is_finite() || span < 0.0 {
        return Err(Error::Orientation(
            "sky-oriented footprint is not finite".into(),
        ));
    }
    let extent = (span - 1.0e-9).ceil().max(0.0) + 1.0;
    if extent > source_width.max(source_height).saturating_mul(4) as f64 {
        return Err(Error::Orientation(
            "sky-oriented footprint is implausibly large".into(),
        ));
    }
    Ok(extent as usize)
}

fn multiply_2x2(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0].mul_add(right[0][0], left[0][1] * right[1][0]),
            left[0][0].mul_add(right[0][1], left[0][1] * right[1][1]),
        ],
        [
            left[1][0].mul_add(right[0][0], left[1][1] * right[1][0]),
            left[1][0].mul_add(right[0][1], left[1][1] * right[1][1]),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_wcs_keeps_dimensions_and_samples() {
        let image = LinearImage::new(5, 3, 1, (0..15).map(|value| value as f32).collect()).unwrap();
        let wcs = Wcs::from_center_scale_rotation((120.0, 30.0), (2.0, 1.0), 1.5, 0.0, false);
        let plan = SkyOrientationPlan::new(image.width, image.height, &wcs).unwrap();
        let oriented = plan.apply(&image).unwrap();

        assert_eq!((plan.output_width(), plan.output_height()), (5, 3));
        assert_eq!(plan.source_to_output(), AffineTransform::IDENTITY);
        assert_eq!(oriented, image);
    }

    #[test]
    fn quarter_turn_retains_the_full_footprint_and_points_north_up() {
        let image = LinearImage::new(4, 3, 1, (0..12).map(|value| value as f32).collect()).unwrap();
        let wcs = Wcs::from_center_scale_rotation((120.0, 30.0), (1.5, 1.0), 1.5, 90.0, false);
        let plan = SkyOrientationPlan::new(image.width, image.height, &wcs).unwrap();
        let oriented = plan.apply(&image).unwrap();

        assert_eq!((oriented.width, oriented.height), (3, 4));
        let center = plan.output_wcs().crpix;
        let north = plan.output_wcs().pixel_to_world(center.0, center.1 - 1.0);
        let east = plan.output_wcs().pixel_to_world(center.0 - 1.0, center.1);
        assert!(north.1 > plan.output_wcs().crval.1);
        assert!(east.0 > plan.output_wcs().crval.0);
        assert_eq!(
            oriented
                .data
                .iter()
                .filter(|value| value.is_finite())
                .count(),
            12
        );
    }

    #[test]
    fn mirrored_wcs_is_reprojected_to_east_left() {
        let image = LinearImage::new(4, 2, 1, (0..8).map(|value| value as f32).collect()).unwrap();
        let wcs = Wcs::from_center_scale_rotation((120.0, 30.0), (1.5, 0.5), 1.5, 0.0, true);
        let plan = SkyOrientationPlan::new(image.width, image.height, &wcs).unwrap();
        let transform = plan.source_to_output();
        let determinant = transform.matrix[0][0].mul_add(
            transform.matrix[1][1],
            -transform.matrix[0][1] * transform.matrix[1][0],
        );
        let oriented = plan.apply(&image).unwrap();

        assert!(determinant < 0.0);
        assert_eq!(oriented.data, vec![3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0]);
    }
}
