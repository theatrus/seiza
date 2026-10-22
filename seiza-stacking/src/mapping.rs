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
/// section of its own. So too the bin factor of a stack whose lights were
/// binned ([`crate::StackOptions::bin_factor`]): JSON carries it as an
/// optional `bin_factor`, left out at 1, and a context keeps it in its
/// options. Superpixel mappings also carry `channel_sites`, three `[x, y]`
/// offsets in source pixels in RGB order; contexts keep these in metadata.
/// Sources passed to extraction must use the same preparation as the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisteredFrameMapping {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: SimilarityTransform,
    normalization: NormalizationMap,
    warp: Option<crate::PolynomialWarp>,
    bin_factor: u32,
    channel_sites: Option<crate::registration::ChannelSites>,
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
struct RegisteredFrameMappingReadableRef<'a> {
    schema_version: u32,
    reference_width: usize,
    reference_height: usize,
    transform: &'a SimilarityTransform,
    normalization: &'a NormalizationMap,
    #[serde(skip_serializing_if = "Option::is_none")]
    warp: Option<&'a crate::PolynomialWarp>,
    #[serde(skip_serializing_if = "is_native_bin_factor")]
    bin_factor: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel_sites: Option<crate::registration::ChannelSites>,
}

fn native_bin_factor() -> u32 {
    1
}

fn is_native_bin_factor(bin_factor: &u32) -> bool {
    *bin_factor == 1
}

impl Serialize for RegisteredFrameMapping {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            RegisteredFrameMappingReadableRef {
                schema_version: self.schema_version,
                reference_width: self.reference_width,
                reference_height: self.reference_height,
                transform: &self.transform,
                normalization: &self.normalization,
                warp: self.warp.as_ref(),
                bin_factor: self.bin_factor,
                channel_sites: self.channel_sites,
            }
            .serialize(serializer)
        } else {
            RegisteredFrameMappingRef {
                schema_version: self.schema_version,
                reference_width: self.reference_width,
                reference_height: self.reference_height,
                transform: &self.transform,
                normalization: &self.normalization,
            }
            .serialize(serializer)
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
    #[serde(default = "native_bin_factor")]
    bin_factor: u32,
    #[serde(default)]
    channel_sites: Option<crate::registration::ChannelSites>,
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
                bin_factor: native_bin_factor(),
                channel_sites: None,
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
        mapping.bin_factor = wire.bin_factor;
        mapping.channel_sites = wire.channel_sites;
        mapping.validate().map_err(D::Error::custom)?;
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
            bin_factor: native_bin_factor(),
            channel_sites: None,
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
            bin_factor: native_bin_factor(),
            channel_sites: None,
        };
        mapping.validate()?;
        Ok(mapping)
    }

    /// The factor the frames this maps were binned by, the reference grid
    /// with them: 1 for the sensor's own sampling. See
    /// [`crate::StackOptions::bin_factor`].
    pub fn bin_factor(&self) -> u32 {
        self.bin_factor
    }

    pub(crate) fn channel_sites(&self) -> Option<crate::registration::ChannelSites> {
        self.channel_sites
    }
    pub(crate) fn set_channel_sites(&mut self, sites: Option<crate::registration::ChannelSites>) {
        self.channel_sites = sites;
    }

    pub(crate) fn set_bin_factor(&mut self, bin_factor: u32) {
        self.bin_factor = bin_factor;
    }

    /// The same mapping for the frames binned by `bin_factor` instead: 1
    /// for the sensor's full resolution. `bin_factor` must divide this
    /// mapping's own.
    ///
    /// Pixel `x` of a grid binned by `s` more is centred on pixel
    /// `X = s·x + c` of the finer one, `c = (s − 1) / 2`. Each part of the
    /// mapping is carried across by that map, `T' = C ∘ T ∘ C⁻¹`, which is
    /// exact: the similarity keeps its scale and rotation and takes the
    /// translation `s·t + (I − sR)·(c, c)`, which scaling the translation
    /// alone would miss by `s − 1` pixels after a meridian flip; the
    /// polynomial warp and its lens move their centres by `C` and stretch
    /// their scales by `s`; and the normalization's tiles grow by `s`, so
    /// each finer pixel reads the tile weights of the coarse pixel it lies
    /// in.
    pub fn rebinned(&self, bin_factor: u32) -> Result<Self> {
        if bin_factor == 0 || !self.bin_factor.is_multiple_of(bin_factor) {
            return Err(Error::Registration(format!(
                "a mapping binned by {} cannot be carried to a bin of {bin_factor}",
                self.bin_factor
            )));
        }
        let scale = (self.bin_factor / bin_factor) as usize;
        if scale == 1 {
            return Ok(self.clone());
        }
        let mapping = Self {
            schema_version: self.schema_version,
            reference_width: self.reference_width * scale,
            reference_height: self.reference_height * scale,
            transform: self.transform.rescaled(scale as f64),
            normalization: self.normalization.rescaled(scale),
            warp: self.warp.as_ref().map(|warp| warp.rescaled(scale as f64)),
            bin_factor,
            // Native preparation uses VNG, with co-sited channels.
            channel_sites: if bin_factor == 1 {
                None
            } else {
                self.channel_sites.map(|sites| sites.rescaled(scale as f64))
            },
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
        mapping.bin_factor = self.bin_factor;
        mapping.channel_sites = self.channel_sites;
        Ok(mapping)
    }

    /// This mapping read from its frames at the sensor's own sampling, onto
    /// the same binned reference grid, for a drizzle that drops a binned
    /// stack's sensor pixels: `T ∘ C⁻¹` for the map `C` that carries a
    /// binned pixel centre to the sensor's, and the warp `C ∘ W`, its lens
    /// on the sensor's pixels too. The mapping itself at a bin factor of 1.
    pub(crate) fn sensor_sourced(&self) -> Self {
        if self.bin_factor == 1 {
            return self.clone();
        }
        let scale = f64::from(self.bin_factor);
        let offset = (scale - 1.0) / 2.0;
        let unbin = SimilarityTransform {
            scale: 1.0 / scale,
            rotation_radians: 0.0,
            translation_x: -offset / scale,
            translation_y: -offset / scale,
        };
        Self {
            channel_sites: None,
            transform: unbin.then(self.transform),
            warp: self.warp.as_ref().map(|warp| warp.source_rescaled(scale)),
            ..self.clone()
        }
    }

    /// This mapping through another registration: `transform` and `warp` in
    /// place of its own, on the same grid with the same normalization.
    pub(crate) fn with_geometry(
        &self,
        transform: SimilarityTransform,
        warp: Option<crate::PolynomialWarp>,
    ) -> Result<Self> {
        let mut mapping = Self::from_parts(
            self.schema_version,
            self.reference_width,
            self.reference_height,
            transform,
            self.normalization.clone(),
        )?;
        mapping.set_warp(warp)?;
        mapping.bin_factor = self.bin_factor;
        mapping.channel_sites = self.channel_sites;
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

    /// The region of the reference grid a `source_width` by `source_height`
    /// frame reaches through this mapping; see
    /// [`crate::registration::footprint_region`].
    pub(crate) fn footprint(&self, source_width: usize, source_height: usize) -> ReferenceRegion {
        crate::registration::footprint_region(
            source_width,
            source_height,
            self.reference_width,
            self.reference_height,
            self.geometry(),
            self.transform,
        )
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
        if !(1..=4).contains(&self.bin_factor) {
            return Err(Error::Registration(format!(
                "registered frame bin factor must be 1 to 4, not {}",
                self.bin_factor
            )));
        }
        self.transform.validate()?;
        if let Some(sites) = self.channel_sites {
            sites.validate()?;
        }
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
        self.extract_region_sited(source, region, interpolation, self.channel_sites)
    }

    /// [`Self::extract_region_with`] for a source whose channels sample the
    /// sky at `sites`; see [`crate::registration::resample_region_sited`].
    pub(crate) fn extract_region_sited(
        &self,
        source: &LinearImage,
        region: ReferenceRegion,
        interpolation: crate::Interpolation,
        sites: Option<crate::registration::ChannelSites>,
    ) -> Result<LinearImage> {
        self.validate()?;
        let mut crop = crate::registration::resample_region_sited(
            source,
            self.reference_width,
            self.reference_height,
            region,
            self.geometry(),
            interpolation.into(),
            sites,
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
            if let Some(sites) = self.channel_sites.filter(|_| source.channels == 3) {
                let transform = self.transform.as_affine().then(reference_to_output);
                let mut data = Vec::new();
                for (channel, (dx, dy)) in sites.0.into_iter().enumerate() {
                    let shifted = AffineTransform {
                        translation_x: dx,
                        translation_y: dy,
                        ..AffineTransform::IDENTITY
                    }
                    .then(transform);
                    let plane = resample_region_to_reference_affine(
                        &source.channel(channel)?,
                        output_width,
                        output_height,
                        output_region,
                        shifted,
                    )?;
                    if data.is_empty() {
                        let samples = plane.data.len().checked_mul(3).ok_or_else(|| {
                            Error::Registration("resampled image dimensions overflow".into())
                        })?;
                        data.resize(samples, 0.0);
                    }
                    for (pixel, value) in data.as_chunks_mut::<3>().0.iter_mut().zip(plane.data) {
                        pixel[channel] = value;
                    }
                }
                let mut crop =
                    LinearImage::new(output_region.width, output_region.height, 3, data)?;
                self.normalization.apply_global(&mut crop)?;
                return Ok(crop);
            }
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

        let mut intermediate = crate::registration::resample_region_sited(
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
            self.channel_sites,
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
    fn superpixel_mapping_extracts_co_sited_planes_through_affine_and_warp() {
        use crate::registration::ChannelSites;
        let sites = ChannelSites([(-0.125, -0.125), (0.0, 0.0), (0.125, 0.125)]);
        let data = (0..32 * 24)
            .flat_map(|i| {
                sites.0.map(|(dx, dy)| {
                    (100.0 + 3.0 * (i % 32) as f64 + 2.0 * (i / 32) as f64 + 3.0 * dx + 2.0 * dy)
                        as f32
                })
            })
            .collect();
        let source = LinearImage::new(32, 24, 3, data).unwrap();
        let mut mapping = RegisteredFrameMapping::identity(&source);
        mapping.set_bin_factor(4);
        mapping.set_channel_sites(Some(sites));
        let identity = crate::NormalizationMap::identity(&source);
        mapping = mapping
            .with_normalization(identity)
            .unwrap()
            .with_geometry(SimilarityTransform::IDENTITY, None)
            .unwrap();
        let mapping: RegisteredFrameMapping =
            serde_json::from_str(&serde_json::to_string(&mapping).unwrap()).unwrap();
        assert_eq!(
            mapping.rebinned(2).unwrap().channel_sites(),
            Some(sites.rescaled(2.0))
        );
        assert_eq!(mapping.rebinned(1).unwrap().channel_sites(), None);
        assert_eq!(mapping.sensor_sourced().channel_sites(), None);
        let region = ReferenceRegion {
            x: 5,
            y: 5,
            width: 12,
            height: 10,
        };
        let pairs = [
            (0.0, 0.0),
            (31.0, 0.0),
            (0.0, 23.0),
            (31.0, 23.0),
            (15.0, 11.0),
        ]
        .map(|p| (p, p));
        let warp = crate::PolynomialWarp::fit(1, 32, 24, &pairs).unwrap();
        for mapping in [
            mapping.clone(),
            mapping
                .with_geometry(SimilarityTransform::IDENTITY, Some(warp))
                .unwrap(),
        ] {
            for crop in [
                mapping.extract_region(&source, region).unwrap(),
                mapping
                    .extract_region_after_affine(&source, 32, 24, region, AffineTransform::IDENTITY)
                    .unwrap(),
            ] {
                for (i, rgb) in crop.data.as_chunks::<3>().0.iter().enumerate() {
                    let expected = 100.0
                        + 3.0 * (region.x + i % region.width) as f32
                        + 2.0 * (region.y + i / region.width) as f32;
                    assert!(
                        rgb.iter().all(|v| (v - expected).abs() < 1e-3),
                        "{rgb:?} vs {expected}"
                    );
                }
            }
        }
        let mut json = serde_json::to_value(&mapping).unwrap();
        json["channel_sites"] = serde_json::json!([[99.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
        assert!(serde_json::from_value::<RegisteredFrameMapping>(json).is_err());
    }

    /// A binned mapping with a quadratic warp and a tiled normalization.
    fn binned_mapping(bin_factor: u32, rotation_radians: f64) -> RegisteredFrameMapping {
        let (width, height) = (96, 64);
        let transform = SimilarityTransform {
            scale: 1.002,
            rotation_radians,
            translation_x: 37.25,
            translation_y: -11.5,
        };
        // A warp that bends away from the similarity, as a lens would.
        let pairs = (0..9)
            .flat_map(|row| (0..9).map(move |column| (column, row)))
            .map(|(column, row)| {
                let (x, y) = (column as f64 * 11.0, row as f64 * 7.5);
                let (u, v) = transform.inverse_apply(x, y);
                ((x, y), (u + 1.0e-4 * x * y, v - 2.0e-4 * x * x))
            })
            .collect::<Vec<_>>();
        let warp = crate::PolynomialWarp::fit(2, width, height, &pairs).unwrap();
        let image = |seed: usize| {
            let data = (0..width * height)
                .map(|index| {
                    100.0 + ((index * 37 + seed * 11) % 53) as f32 + (index % width) as f32
                })
                .collect();
            LinearImage::new(width, height, 1, data).unwrap()
        };
        let normalization = NormalizationMap::estimate(
            &image(0),
            &image(1),
            crate::NormalizationMode::Local { tile_size: 16 },
        )
        .unwrap();
        let mut mapping =
            RegisteredFrameMapping::new(width, height, transform, normalization).unwrap();
        mapping.set_warp(Some(warp)).unwrap();
        mapping.set_bin_factor(bin_factor);
        mapping
    }

    /// Carried to the sensor, a binned mapping takes every native point
    /// where converting it to the binned grid, mapping it there and
    /// converting it back takes it: a half turn included, and through the
    /// warp as through the similarity.
    #[test]
    fn rebinned_mappings_agree_with_converting_each_point() {
        for bin_factor in [2_u32, 3, 4] {
            let scale = f64::from(bin_factor);
            let offset = (scale - 1.0) / 2.0;
            let to_native = |(x, y): (f64, f64)| (scale * x + offset, scale * y + offset);
            let to_binned = |(x, y): (f64, f64)| ((x - offset) / scale, (y - offset) / scale);
            for rotation in [0.0, 0.3, std::f64::consts::PI, std::f64::consts::PI + 0.01] {
                let binned = binned_mapping(bin_factor, rotation);
                let native = binned.rebinned(1).unwrap();
                assert_eq!(native.bin_factor(), 1);
                assert_eq!(
                    (native.reference_width(), native.reference_height()),
                    (96 * bin_factor as usize, 64 * bin_factor as usize)
                );
                for point in [(0.0, 0.0), (123.4, 56.7), (300.0, 250.0), (-20.0, 400.0)] {
                    let expected = to_native(
                        binned
                            .transform()
                            .apply(to_binned(point).0, to_binned(point).1),
                    );
                    let found = native.transform().apply(point.0, point.1);
                    assert!(
                        (found.0 - expected.0).hypot(found.1 - expected.1) < 1e-9,
                        "bin {bin_factor}, turn {rotation}: {found:?} against {expected:?}"
                    );
                    let (x, y) = to_binned(point);
                    let expected = to_native(binned.warp().unwrap().apply(x, y));
                    let found = native.warp().unwrap().apply(point.0, point.1);
                    assert!(
                        (found.0 - expected.0).hypot(found.1 - expected.1) < 1e-9,
                        "warp, bin {bin_factor}, turn {rotation}: {found:?} against {expected:?}"
                    );
                }
                // A half turn moves the translation `bin − 1` pixels from
                // scaling it alone.
                let naive_x = scale * binned.transform().translation_x;
                let naive_y = scale * binned.transform().translation_y;
                let shift = (native.transform().translation_x - naive_x)
                    .hypot(native.transform().translation_y - naive_y);
                if rotation == std::f64::consts::PI {
                    let expected = (scale - 1.0) * std::f64::consts::SQRT_2;
                    assert!(
                        (shift - 1.002 * expected).abs() < 0.01 * expected,
                        "{shift}"
                    );
                }
                // Each native pixel reads the tile weights at its centre on
                // the binned grid, which at an odd factor is a binned pixel's.
                if bin_factor == 3 {
                    let (coarse, fine) = (
                        binned.normalization().sampler(),
                        native.normalization().sampler(),
                    );
                    for (x, y) in [(0, 0), (5, 9), (40, 33), (95, 63)] {
                        let (gain, offset) = coarse.at(x, y, 0);
                        let (fine_gain, fine_offset) = fine.at(3 * x + 1, 3 * y + 1, 0);
                        assert!(
                            (gain - fine_gain).abs() < 1e-5 && (offset - fine_offset).abs() < 1e-3
                        );
                    }
                }
                // The same factor gives the mapping itself.
                assert!(binned.rebinned(bin_factor).unwrap() == binned);
            }
        }
        // Only a factor dividing the mapping's own is reachable.
        assert!(binned_mapping(4, 0.0).rebinned(2).is_ok());
        assert!(binned_mapping(4, 0.0).rebinned(3).is_err());
        assert!(binned_mapping(2, 0.0).rebinned(4).is_err());
    }

    /// JSON carries the bin factor, left out at 1 so earlier mappings read
    /// and write as they did; compact binary leaves it to the stack options.
    #[test]
    fn json_carries_the_bin_factor_and_compact_binary_leaves_it_out() {
        let binned = binned_mapping(2, 0.1);
        let json = serde_json::to_string(&binned).unwrap();
        assert!(json.ends_with(",\"bin_factor\":2}"), "{json}");
        let read: RegisteredFrameMapping = serde_json::from_str(&json).unwrap();
        assert!(read == binned);
        let mut native = binned.clone();
        native.set_bin_factor(1);
        assert!(
            !serde_json::to_string(&native)
                .unwrap()
                .contains("bin_factor")
        );
        assert_eq!(
            postcard::to_stdvec(&binned).unwrap(),
            postcard::to_stdvec(&native).unwrap()
        );
        let refused = json.replace("\"bin_factor\":2", "\"bin_factor\":5");
        assert!(serde_json::from_str::<RegisteredFrameMapping>(&refused).is_err());
    }

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
