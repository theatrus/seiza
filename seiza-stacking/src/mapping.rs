use crate::{
    AffineTransform, Error, LinearImage, NormalizationMap, ReferenceRegion, Result,
    SimilarityTransform, resample_region_to_reference, resample_region_to_reference_affine,
    resample_to_reference,
};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

const REGISTERED_FRAME_MAPPING_SCHEMA_VERSION: u32 = 1;

/// Versioned processing provenance that maps one prepared source frame onto a
/// stack reference grid.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegisteredFrameMapping {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: SimilarityTransform,
    normalization: NormalizationMap,
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

impl<'de> Deserialize<'de> for RegisteredFrameMapping {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredFrameMappingWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.schema_version,
            wire.reference_width,
            wire.reference_height,
            wire.transform,
            wire.normalization,
        )
        .map_err(D::Error::custom)
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
        };
        mapping.validate()?;
        Ok(mapping)
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
        self.validate()?;
        let mut crop = resample_region_to_reference(
            source,
            self.reference_width,
            self.reference_height,
            region,
            self.transform,
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
        if self.normalization.is_global() {
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

        let mut intermediate = resample_to_reference(
            source,
            self.reference_width,
            self.reference_height,
            self.transform,
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
            resample_to_reference(&source, 32, 32, SimilarityTransform::IDENTITY).unwrap();
        normalization.apply(&mut intermediate).unwrap();
        let expected =
            resample_region_to_reference(&intermediate, 32, 32, region, output_transform).unwrap();

        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
}
