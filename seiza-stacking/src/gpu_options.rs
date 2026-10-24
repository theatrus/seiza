//! Which of stacking's passes a call lets the GPU run.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Which of stacking's heavy passes may run on the GPU, in a build with the
/// `gpu` feature on a machine that has one, and how. Every pass may by
/// default. A pass that may not, or that finds no GPU, runs on the CPU, as
/// in a build without the feature. The GPU works in single precision, so
/// its results differ from the CPU's by rounding: `docs/design/gpu-stacking.md`
/// gives each pass's accuracy.
///
/// Each request carries its own: [`crate::StackOptions::gpu`] governs a live
/// stack, from the frames it is pushed to its reintegration, which
/// [`crate::BatchStackOptions::gpu`] can only narrow;
/// [`crate::BatchStackOptions::gpu`] governs the integration of registered
/// frames, and [`crate::DefectDetectionOptions::gpu`] defect learning. A
/// call that takes no options, such as [`crate::resample_to_reference`] or
/// [`crate::FitsFrame::into_prepared`], uses the defaults. Its twin that
/// takes them runs where they allow:
/// [`crate::FitsFrame::into_prepared_with_gpu`],
/// [`crate::resample_region_to_reference_with_gpu`],
/// [`crate::RegisteredFrameMapping::extract_region_with_gpu`],
/// [`crate::RegisteredFrameMapping::extract_region_photosites_with_gpu`] and
/// [`crate::RegisteredFrameMapping::extract_region_after_affine_with_gpu`].
/// A saved stack reopens under the options it was saved with, or under
/// others by [`crate::LiveStacker::open_context_with_gpu`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuOptions {
    /// Whether to use the GPU at all: `false` keeps every pass on the CPU,
    /// and the call never opens the device.
    pub enabled: bool,
    /// Screen each light for the samples defect learning examines.
    pub defects: bool,
    /// Debayer a Bayer frame's interior by VNG.
    pub debayer: bool,
    /// Detect each frame's stars.
    pub detection: bool,
    /// Resample each frame, and its clip mask, onto the reference grid.
    pub resampling: bool,
    /// Hold a live stack's accumulator, and normalize and admit each frame
    /// there that was resampled there.
    pub accumulator: bool,
    /// Run the rejection passes of reintegration and of
    /// [`crate::integrate_registered_frames_banded`].
    pub integration: bool,
    /// Frame calls (debayering, detection, resampling, screening) the GPU
    /// takes at once, at least 1; the rest wait their turn without holding
    /// its memory. With a dozen frames prepared at once, uncapped calls
    /// took the M45 stack on an Iris Xe to 7.8-8.6 GB, against 7.3-7.5 GB at
    /// 3, the default. One at a time left that GPU idle while a call
    /// copied: 45-46 s, against 39.4-39.7 at 2 and 38.8-38.9 at 3.
    pub concurrency: usize,
    /// Bytes of device memory to plan the GPU's work in, or `None` for what
    /// the device has, as its backend reports it; a cap above that is held
    /// to it. A live stack's accumulator stays on the CPU when it, the
    /// frames prepared for it at once and the frame calls' buffers would
    /// need more than three quarters of this, and reintegration's bands
    /// keep to a sixteenth. Less than the device has leaves the rest to
    /// other programs, such as a desktop the GPU also drives.
    pub memory_bytes: Option<u64>,
    /// Print each GPU call's stages and their times to stderr, to see where
    /// the time goes on a machine.
    pub trace: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            defects: true,
            debayer: true,
            detection: true,
            resampling: true,
            accumulator: true,
            integration: true,
            concurrency: 3,
            memory_bytes: None,
            trace: false,
        }
    }
}

impl GpuOptions {
    /// Every pass on the CPU.
    pub fn cpu() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether these are the defaults, which options leave unserialized.
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.concurrency == 0 {
            return Err(Error::Stack(
                "the GPU's concurrency must be at least 1".into(),
            ));
        }
        if self.memory_bytes == Some(0) {
            return Err(Error::Stack(
                "the GPU's memory must be more than zero bytes".into(),
            ));
        }
        Ok(())
    }

    /// The passes both these and `other` allow, at the lower concurrency
    /// and memory, traced if either asks.
    pub(crate) fn within(self, other: Self) -> Self {
        Self {
            enabled: self.enabled && other.enabled,
            defects: self.defects && other.defects,
            debayer: self.debayer && other.debayer,
            detection: self.detection && other.detection,
            resampling: self.resampling && other.resampling,
            accumulator: self.accumulator && other.accumulator,
            integration: self.integration && other.integration,
            concurrency: self.concurrency.min(other.concurrency),
            memory_bytes: match (self.memory_bytes, other.memory_bytes) {
                (Some(own), Some(other)) => Some(own.min(other)),
                (own, other) => own.or(other),
            },
            trace: self.trace || other.trace,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_read_as_written_and_default_every_field() {
        let options: GpuOptions = serde_json::from_str(
            r#"{"debayer": false, "concurrency": 2, "memory_bytes": 8589934592}"#,
        )
        .unwrap();
        assert_eq!(
            options,
            GpuOptions {
                debayer: false,
                concurrency: 2,
                memory_bytes: Some(8 << 30),
                ..GpuOptions::default()
            }
        );
        assert!(serde_json::from_str::<GpuOptions>(r#"{"stages": 1}"#).is_err());
        assert!(GpuOptions::default().is_default());
        assert!(!GpuOptions::cpu().is_default());
    }

    #[test]
    fn a_narrower_request_narrows_and_never_widens() {
        let stack = GpuOptions {
            detection: false,
            concurrency: 2,
            memory_bytes: Some(8 << 30),
            ..GpuOptions::default()
        };
        let call = GpuOptions {
            integration: false,
            concurrency: 4,
            trace: true,
            ..GpuOptions::default()
        };
        assert_eq!(
            stack.within(call),
            GpuOptions {
                detection: false,
                integration: false,
                concurrency: 2,
                memory_bytes: Some(8 << 30),
                trace: true,
                ..GpuOptions::default()
            }
        );
        let smaller = GpuOptions {
            memory_bytes: Some(4 << 30),
            ..GpuOptions::default()
        };
        assert_eq!(stack.within(smaller).memory_bytes, Some(4 << 30));
        assert_eq!(smaller.within(stack).memory_bytes, Some(4 << 30));
        assert!(!GpuOptions::default().within(GpuOptions::cpu()).enabled);
        assert!(
            GpuOptions {
                concurrency: 0,
                ..GpuOptions::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            GpuOptions {
                memory_bytes: Some(0),
                ..GpuOptions::default()
            }
            .validate()
            .is_err()
        );
    }
}
