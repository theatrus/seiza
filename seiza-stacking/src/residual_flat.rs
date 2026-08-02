//! Compatibility adapters for the host-neutral residual-response kernel.

use crate::{Error, LinearImage, Result};
use seiza_calibration::{LinearImageMut, LinearImageRef};
use std::sync::OnceLock;

pub use seiza_calibration::{
    RESIDUAL_FLAT_ALGORITHM_VERSION, ResidualFlatDiagnostics, ResidualFlatOptions,
};

/// A normalized response patch. Values at one are neutral; lower values
/// describe bounded attenuation that correction divides out.
#[derive(Clone, Debug)]
pub struct ResidualFlatPatch {
    inner: seiza_calibration::ResidualFlatPatch,
    compatibility_response: OnceLock<LinearImage>,
}

impl ResidualFlatPatch {
    /// Validate a cached or externally generated normalized response.
    pub fn from_response(response: LinearImage) -> Result<Self> {
        let inner = seiza_calibration::ResidualFlatPatch::from_response(
            response.data,
            response.width,
            response.height,
            response.channels,
        )
        .map_err(calibration_error)?;
        Ok(Self {
            inner,
            compatibility_response: OnceLock::new(),
        })
    }

    /// The normalized multiplicative response image.
    pub fn response(&self) -> &LinearImage {
        self.compatibility_response.get_or_init(|| {
            LinearImage::new(
                self.inner.width(),
                self.inner.height(),
                self.inner.channels(),
                self.inner.response().to_vec(),
            )
            .expect("residual response dimensions were validated at construction")
        })
    }

    /// Consume the patch and return its normalized response image.
    pub fn into_response(self) -> LinearImage {
        if let Some(response) = self.compatibility_response.into_inner() {
            return response;
        }
        let (width, height, channels, response) = self.inner.into_parts();
        LinearImage::new(width, height, channels, response)
            .expect("residual response dimensions were validated at construction")
    }

    /// Divide this response out of a same-sampling image at a detector origin.
    pub fn apply_at(&self, image: &mut LinearImage, x: usize, y: usize) -> Result<()> {
        let image = LinearImageMut::new(&mut image.data, image.width, image.height, image.channels)
            .map_err(calibration_error)?;
        self.inner.apply_at(image, x, y).map_err(calibration_error)
    }
}

impl PartialEq for ResidualFlatPatch {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A generated response patch and the evidence summary used to accept it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualFlatBuild {
    pub patch: ResidualFlatPatch,
    pub diagnostics: ResidualFlatDiagnostics,
}

/// Estimate a bounded multiplicative response from detector-aligned crops.
///
/// This keeps the `seiza-stacking` API stable for one release. New hosts
/// should call `seiza_calibration::build_residual_flat_patch` with borrowed
/// linear image views.
pub fn build_residual_flat_patch(
    samples: &[LinearImage],
    options: &ResidualFlatOptions,
) -> Result<ResidualFlatBuild> {
    let views = samples.iter().map(image_ref).collect::<Vec<_>>();
    let built =
        seiza_calibration::build_residual_flat_patch(&views, options).map_err(calibration_error)?;
    Ok(ResidualFlatBuild {
        patch: ResidualFlatPatch {
            inner: built.patch,
            compatibility_response: OnceLock::new(),
        },
        diagnostics: built.diagnostics,
    })
}

fn image_ref(image: &LinearImage) -> LinearImageRef<'_> {
    LinearImageRef::new(&image.data, image.width, image.height, image.channels)
        .expect("LinearImage dimensions were validated at construction")
}

fn calibration_error(error: seiza_calibration::Error) -> Error {
    Error::Calibration(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with_shadow(width: usize, height: usize, offset: f32) -> LinearImage {
        let data = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let background = 1_000.0 + 1.5 * x as f32 + 0.75 * y as f32 + offset;
                    let radius = (x as f32 - 15.5).hypot(y as f32 - 15.5);
                    let response = if (6.0..=9.0).contains(&radius) {
                        0.8
                    } else {
                        1.0
                    };
                    let moving_star = if (x + offset as usize) % width == y {
                        2_000.0
                    } else {
                        0.0
                    };
                    background * response + moving_star
                })
            })
            .collect();
        LinearImage::new(width, height, 1, data).unwrap()
    }

    #[test]
    fn compatibility_adapter_preserves_response_fixture() {
        let samples = (0..7)
            .map(|index| sample_with_shadow(32, 32, index as f32 * 3.0))
            .collect::<Vec<_>>();
        let options = ResidualFlatOptions {
            smoothing_sigma: 0.8,
            edge_feather_fraction: 0.0,
            ..ResidualFlatOptions::default()
        };

        let built = build_residual_flat_patch(&samples, &options).unwrap();
        let response_hash = built
            .patch
            .response()
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .fold(0xcbf29ce484222325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });

        assert_eq!(response_hash, 0x453e2fed4c1ad3cd);
    }

    #[test]
    fn compatibility_adapter_applies_patch() {
        let patch =
            ResidualFlatPatch::from_response(LinearImage::new(2, 2, 1, vec![0.5; 4]).unwrap())
                .unwrap();
        let mut image = LinearImage::new(4, 4, 1, vec![2.0; 16]).unwrap();

        assert!(patch.compatibility_response.get().is_none());
        patch.apply_at(&mut image, 1, 1).unwrap();

        assert_eq!(image.data[5], 4.0);
        assert_eq!(image.data[0], 2.0);
        assert!(patch.compatibility_response.get().is_none());
    }
}
