//! What Seiza's image steps share, so that a host will be able to run any
//! of them on part of an image and combine their results. This crate holds
//! the shared types; each step adopts them in its own crate.
//!
//! - [`ImageShape`]: the width, height and channel count of a caller's
//!   samples.
//! - [`Region`]: a rectangle of an image, with the copying and padding a
//!   step needs to work on it.
//! - [`Domain`]: whether samples are linear light or display-referred, and
//!   the refusal a step gives when handed the wrong kind.
//! - [`Mask`]: one plane of values in `[0, 1]` that limits where a step
//!   takes effect.
//! - [`RegionOperation`]: the contract for a step that is resolved once from
//!   a whole image and then applied to any region of it, and
//!   [`region_mismatch`], the check that an operation keeps the contract.
//!
//! This crate adds no image type. A step takes row-major, channel-interleaved
//! `f32` samples with their shape, as Seiza's steps already do, so a host
//! keeps its own storage.
//!
//! Pixels count from 0. A region's `x` and `y` are its first column and row.

mod domain;
mod mask;
mod region;
mod shape;
mod step;

pub use domain::{Accepts, Domain, DomainError, ParseAcceptsError, ParseDomainError};
pub use mask::{Mask, MaskError, MaskRange};
pub use region::{Region, RegionError};
pub use shape::ImageShape;
pub use step::{
    Block, Edges, RegionApplyError, RegionMismatch, RegionOperation, apply_to_region,
    region_mismatch,
};

/// Compiles the README's example.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
