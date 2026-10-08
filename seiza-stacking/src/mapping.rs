use crate::{
    AffineTransform, Error, LinearImage, NormalizationMap, ReferenceRegion, Result,
    SimilarityTransform, resample_region_to_reference_affine,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

const REGISTERED_FRAME_MAPPING_SCHEMA_VERSION: u32 = 1;

/// Versioned processing provenance that maps one prepared source frame onto a
/// stack reference grid.
/// How one admitted frame maps onto the reference grid: its registration
/// and normalization, enough to extract it again without registering it.
///
/// A polynomial warp, when registration fitted one, is serialized only by
/// self-describing formats such as JSON, as an optional `warp` field. Compact
/// binary formats keep the layout without it, so live-stack contexts written
/// before warps existed still read; a context stores its frames' warps in a
/// section of its own.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisteredFrameMapping {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: SimilarityTransform,
    normalization: NormalizationMap,
    warp: Option<crate::PolynomialWarp>,
}

#[derive(Serialize)]
struct RegisteredFrameMappingRef<'a> {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: &'a SimilarityTransform,
    normalization: &'a NormalizationMap,
}

#[derive(Serialize)]
struct RegisteredFrameMappingWarpedRef<'a> {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: &'a SimilarityTransform,
    normalization: &'a NormalizationMap,
    warp: &'a crate::PolynomialWarp,
}

impl Serialize for RegisteredFrameMapping {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self.warp {
            Some(warp) if serializer.is_human_readable() => RegisteredFrameMappingWarpedRef {
                schema_version: self.schema_version,
                reference_width: self.reference_width,
                reference_height: self.reference_height,
                transform: &self.transform,
                normalization: &self.normalization,
                warp,
            }
            .serialize(serializer),
            _ => RegisteredFrameMappingRef {
                schema_version: self.schema_version,
                reference_width: self.reference_width,
                reference_height: self.reference_height,
                transform: &self.transform,
                normalization: &self.normalization,
            }
            .serialize(serializer),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredFrameMappingWire {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: SimilarityTransform,
    normalization: NormalizationMap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredFrameMappingReadableWire {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: SimilarityTransform,
    normalization: NormalizationMap,
    #[serde(default)]
    warp: Option<crate::PolynomialWarp>,
}

impl<'de> Deserialize<'de> for RegisteredFrameMapping {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = if deserializer.is_human_readable() {
            RegisteredFrameMappingReadableWire::deserialize(deserializer)?
        } else {
            let wire = RegisteredFrameMappingWire::deserialize(deserializer)?;
            RegisteredFrameMappingReadableWire {
                schema_version: wire.schema_version,
                reference_width: wire.reference_width,
                reference_height: wire.reference_height,
                transform: wire.transform,
                normalization: wire.normalization,
                warp: None,
            }
        };
        let mut mapping = Self::from_parts(
            wire.schema_version,
            wire.reference_width,
            wire.reference_height,
            wire.transform,
            wire.normalization,
        )
        .map_err(D::Error::custom)?;
        mapping.set_warp(wire.warp).map_err(D::Error::custom)?;
        Ok(mapping)
    }
}

impl RegisteredFrameMapping {
    /// Build validated source-to-reference processing provenance.
    pub fn new(
        reference_width: usize,
        reference_height: usize,
        transform: SimilarityTransform,
        normalization: NormalizationMap,
    ) -> Result<Self> {
        Self::from_parts(
            REGISTERED_FRAME_MAPPING_SCHEMA_VERSION,
            reference_width,
            reference_height,
            transform,
            normalization,
        )
    }

    /// Build an identity mapping for a prepared reference frame.
    pub fn identity(reference: &LinearImage) -> Self {
        Self {
            schema_version: REGISTERED_FRAME_MAPPING_SCHEMA_VERSION,
            reference_width: reference.width,
            reference_height: reference.height,
            transform: SimilarityTransform::IDENTITY,
            normalization: NormalizationMap::identity(reference),
            warp: None,
        }
    }

    fn from_parts(
        schema_version: u32,
        reference_width: usize,
        reference_height: usize,
        transform: SimilarityTransform,
        normalization: NormalizationMap,
    ) -> Result<Self> {
        let mapping = Self {
            schema_version,
            reference_width,
            reference_height,
            transform,
            normalization,
            warp: None,
        };
        mapping.validate()?;
        Ok(mapping)
    }

    /// The normalization this mapping applies.
    pub fn normalization(&self) -> &NormalizationMap {
        &self.normalization
    }

    /// This mapping with another normalization, which must cover the same
    /// reference grid.
    pub fn with_normalization(&self, normalization: NormalizationMap) -> Result<Self> {
        let mut mapping = Self::from_parts(
            self.schema_version,
            self.reference_width,
            self.reference_height,
            self.transform,
            normalization,
        )?;
        mapping.set_warp(self.warp.clone())?;
        Ok(mapping)
    }

    /// The polynomial warp registration fitted, if any. Extraction resamples
    /// through it in place of [`Self::transform`].
    pub fn warp(&self) -> Option<&crate::PolynomialWarp> {
        self.warp.as_ref()
    }

    /// Attach or clear the polynomial warp, validating it.
    pub fn set_warp(&mut self, warp: Option<crate::PolynomialWarp>) -> Result<()> {
        if let Some(warp) = &warp {
            warp.validate()?;
        }
        self.warp = warp;
        Ok(())
    }

    fn geometry(&self) -> crate::registration::FrameGeometry<'_> {
        crate::registration::FrameGeometry::of(self.transform, self.warp.as_ref())
    }

    /// Check the serialized mapping before it is used for pixel work.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != REGISTERED_FRAME_MAPPING_SCHEMA_VERSION {
            return Err(Error::Registration(format!(
                "unsupported registered frame mapping schema version {}",
                self.schema_version
            )));
        }
        if self.reference_width == 0 || self.reference_height == 0 {
            return Err(Error::Registration(
                "registered frame reference dimensions must be non-zero".into(),
            ));
        }
        self.transform.validate()?;
        if let Some(warp) = &self.warp {
            warp.validate()?;
        }
        self.normalization.validate()?;
        if self.normalization.width() != self.reference_width
            || self.normalization.height() != self.reference_height
        {
            return Err(Error::Normalization(
                "registered frame normalization grid does not match its reference".into(),
            ));
        }
        Ok(())
    }

    /// Extract one normalized region on this mapping's reference grid.
    pub fn extract_region(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
    ) -> Result<LinearImage> {
        self.extract_region_with(source, region, crate::Interpolation::Bilinear)
    }

    /// [`Self::extract_region`] with the stack's interpolation, so a replay
    /// resamples each frame as the live pass did.
    pub fn extract_region_with(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        interpolation: crate::Interpolation,
    ) -> Result<LinearImage> {
        self.validate()?;
        let mut crop = crate::registration::resample_region_geometry(
            source,
            self.reference_width,
            self.reference_height,
            region,
            self.geometry(),
            interpolation.into(),
        )?;
        self.normalization
            .apply_region(&mut crop, region.x, region.y)?;
        Ok(crop)
    }

    /// [`Self::extract_region_with`]'s resampling alone, sampling as
    /// `sampling` says, with no normalization.
    pub(crate) fn resample_region(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        sampling: crate::registration::Sampling,
    ) -> Result<LinearImage> {
        self.validate()?;
        crate::registration::resample_region_geometry(
            source,
            self.reference_width,
            self.reference_height,
            region,
            self.geometry(),
            sampling,
        )
    }

    /// [`Self::extract_region`] for Bayer drizzle: the nearest source
    /// photosite per pixel, in its own channel, normalized the same way.
    /// `source` is the debayered frame and `layout` the Bayer layout it was
    /// debayered from.
    pub fn extract_region_photosites(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        layout: crate::BayerLayout,
    ) -> Result<LinearImage> {
        self.validate()?;
        let mut crop = crate::registration::resample_region_geometry(
            source,
            self.reference_width,
            self.reference_height,
            region,
            self.geometry(),
            crate::registration::Sampling::NearestPhotosite(layout),
        )?;
        self.normalization
            .apply_region(&mut crop, region.x, region.y)?;
        Ok(crop)
    }

    /// Extract a region after a second registration stage. Global
    /// normalization commutes with the second resampling and keeps this path
    /// bounded. Local normalization uses the exact two-stage order.
    pub fn extract_region_after(
        &self,
        source: &LinearImage,
        output_width: usize,
        output_height: usize,
        output_region: ReferenceRegion,
        reference_to_output: SimilarityTransform,
    ) -> Result<LinearImage> {
        self.extract_region_after_affine(
            source,
            output_width,
            output_height,
            output_region,
            reference_to_output.as_affine(),
        )
    }

    /// Extract a region after a general affine output reprojection. This keeps
    /// parity-changing sky orientation tied to the exact source registration
    /// and normalization provenance.
    pub fn extract_region_after_affine(
        &self,
        source: &LinearImage,
        output_width: usize,
        output_height: usize,
        output_region: ReferenceRegion,
        reference_to_output: AffineTransform,
    ) -> Result<LinearImage> {
        self.validate()?;
        reference_to_output.validate()?;
        if self.normalization.is_global() && self.warp.is_none() {
            let mut crop = resample_region_to_reference_affine(
                source,
                output_width,
                output_height,
                output_region,
                self.transform.as_affine().then(reference_to_output),
            )?;
            self.normalization.apply_global(&mut crop)?;
            return Ok(crop);
        }

        let mut intermediate = crate::registration::resample_region_geometry(
            source,
            self.reference_width,
            self.reference_height,
            ReferenceRegion {
                x: 0,
                y: 0,
                width: self.reference_width,
                height: self.reference_height,
            },
            self.geometry(),
            crate::registration::Sampling::Bilinear,
        )?;
        self.normalization.apply(&mut intermediate)?;
        resample_region_to_reference_affine(
            &intermediate,
            output_width,
            output_height,
            output_region,
            reference_to_output,
        )
    }

    /// Source-to-reference geometric transform.
    pub fn transform(&self) -> SimilarityTransform {
        self.transform
    }

    /// Reference-grid width.
    pub fn reference_width(&self) -> usize {
        self.reference_width
    }

    /// Reference-grid height.
    pub fn reference_height(&self) -> usize {
        self.reference_height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_carries_the_warp_and_compact_binary_leaves_it_out() {
        let reference = crate::registration::test_star_field(false);
        let source = crate::registration::test_star_field(true);
        let registration = crate::Registrar::new(
            &reference,
            crate::RegistrationOptions {
                model: crate::RegistrationModel::Quadratic,
                ..crate::RegistrationOptions::default()
            },
        )
        .unwrap()
        .register(&source)
        .unwrap();
        let mut mapping = RegisteredFrameMapping::new(
            reference.width,
            reference.height,
            registration.transform,
            NormalizationMap::identity(&reference),
        )
        .unwrap();
        mapping.set_warp(registration.warp.clone()).unwrap();
        assert!(mapping.warp().is_some());

        let json = serde_json::to_string(&mapping).unwrap();
        assert!(json.contains("\"warp\""));
        let from_json: RegisteredFrameMapping = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.warp(), mapping.warp());

        // Without a warp, JSON keeps the fields earlier releases wrote.
        let plain = RegisteredFrameMapping::new(
            reference.width,
            reference.height,
            registration.transform,
            NormalizationMap::identity(&reference),
        )
        .unwrap();
        assert!(!serde_json::to_string(&plain).unwrap().contains("warp"));

        let bytes = postcard::to_stdvec(&mapping).unwrap();
        assert_eq!(bytes, postcard::to_stdvec(&plain).unwrap());
        let from_bytes: RegisteredFrameMapping = postcard::from_bytes(&bytes).unwrap();
        assert!(from_bytes.warp().is_none());

        // The warped extraction follows the distortion: the registered source
        // matches the reference far better than through the similarity alone.
        let region = ReferenceRegion {
            x: 16,
            y: 16,
            width: reference.width - 32,
            height: reference.height - 32,
        };
        let difference = |image: &LinearImage| {
            let mut total = 0.0_f64;
            for y in 0..region.height {
                for x in 0..region.width {
                    let sample = image.data[y * region.width + x];
                    let expected = reference.data[(y + region.y) * reference.width + x + region.x];
                    total += f64::from((sample - expected).abs());
                }
            }
            total
        };
        let warped = mapping.extract_region(&source, region).unwrap();
        let similar = plain.extract_region(&source, region).unwrap();
        assert!(
            difference(&warped) * 3.0 < difference(&similar),
            "warped {} against similarity {}",
            difference(&warped),
            difference(&similar)
        );
    }
    use crate::NormalizationMode;

    #[test]
    fn mapping_round_trips_and_extracts_the_reference_region() {
        let source =
            LinearImage::new(6, 5, 1, (0..30).map(|value| value as f32).collect()).unwrap();
        let mapping = RegisteredFrameMapping::new(
            6,
            5,
            SimilarityTransform::IDENTITY,
            NormalizationMap::identity(&source),
        )
        .unwrap();
        let encoded = serde_json::to_vec(&mapping).unwrap();
        let decoded = serde_json::from_slice::<RegisteredFrameMapping>(&encoded).unwrap();
        let crop = decoded
            .extract_region(
                &source,
                ReferenceRegion {
                    x: 2,
                    y: 1,
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();

        assert_eq!(decoded, mapping);
        assert_eq!(crop.data, vec![8.0, 9.0, 14.0, 15.0]);
    }

    #[test]
    fn mapping_rejects_an_unknown_schema() {
        let source = LinearImage::new(2, 2, 1, vec![1.0; 4]).unwrap();
        let mapping = RegisteredFrameMapping::identity(&source);
        let mut value = serde_json::to_value(mapping).unwrap();
        value["schema_version"] = serde_json::json!(2);

        assert!(serde_json::from_value::<RegisteredFrameMapping>(value).is_err());
    }

    #[test]
    fn affine_output_stage_keeps_source_mapping_and_normalization() {
        let source = LinearImage::new(4, 2, 1, (0..8).map(|value| value as f32).collect()).unwrap();
        let mapping = RegisteredFrameMapping::new(
            4,
            2,
            SimilarityTransform::IDENTITY,
            NormalizationMap::identity(&source),
        )
        .unwrap();
        let crop = mapping
            .extract_region_after_affine(
                &source,
                4,
                2,
                ReferenceRegion {
                    x: 1,
                    y: 0,
                    width: 2,
                    height: 2,
                },
                AffineTransform {
                    matrix: [[-1.0, 0.0], [0.0, 1.0]],
                    translation_x: 3.0,
                    translation_y: 0.0,
                },
            )
            .unwrap();

        assert_eq!(crop.data, vec![2.0, 1.0, 6.0, 5.0]);
    }

    #[test]
    fn local_normalization_keeps_the_exact_two_stage_order() {
        let source = LinearImage::new(
            32,
            32,
            1,
            (0..32 * 32)
                .map(|index| ((index * 37) % 251) as f32)
                .collect(),
        )
        .unwrap();
        let reference = LinearImage::new(
            32,
            32,
            1,
            source
                .data
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let x = index % 32;
                    let y = index / 32;
                    let gain = if x < 16 { 1.5 } else { 2.0 };
                    let offset = if y < 16 { 3.0 } else { 9.0 };
                    value.mul_add(gain, offset)
                })
                .collect(),
        )
        .unwrap();
        let normalization = NormalizationMap::estimate(
            &reference,
            &source,
            NormalizationMode::Local { tile_size: 16 },
        )
        .unwrap();
        let mapping = RegisteredFrameMapping::new(
            32,
            32,
            SimilarityTransform::IDENTITY,
            normalization.clone(),
        )
        .unwrap();
        let output_transform = SimilarityTransform {
            scale: 1.0,
            rotation_radians: 0.0,
            translation_x: 1.0,
            translation_y: -1.0,
        };
        let region = ReferenceRegion {
            x: 3,
            y: 4,
            width: 20,
            height: 18,
        };

        let actual = mapping
            .extract_region_after(&source, 32, 32, region, output_transform)
            .unwrap();
        let mut intermediate =
            crate::resample_to_reference(&source, 32, 32, SimilarityTransform::IDENTITY).unwrap();
        normalization.apply(&mut intermediate).unwrap();
        let expected =
            crate::resample_region_to_reference(&intermediate, 32, 32, region, output_transform)
                .unwrap();

        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
}
