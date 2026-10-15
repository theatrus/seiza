//! The types Seiza's crates share, kept apart from the solver so that a
//! crate needing only these does not move with every `seiza` release:
//!
//! - [`wcs`]: a TAN or TAN-SIP plate solution, [`Wcs`], read from and
//!   written to FITS cards.
//! - [`CancelSignal`]: cooperative cancellation for long work.
//! - [`prior`]: what an image's headers say about where it points,
//!   [`prior::SolvePrior`].
//!
//! `seiza` re-exports all of them at the same paths (`seiza::wcs::Wcs`,
//! `seiza::CancelSignal`, `seiza::prior::SolvePrior`). Changes here should
//! be rare and additive: a minor release of this crate moves every crate
//! that depends on it.

mod cancel;
pub mod prior;
pub mod wcs;

pub use cancel::CancelSignal;
pub use wcs::{FitsCardValue, Sip, Wcs, image_center};
