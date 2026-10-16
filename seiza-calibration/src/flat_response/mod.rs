//! Whether flat sessions describe the same light path, and which one a
//! light's own sky background follows.
//!
//! A flat set is reduced to one **response** ([`metric::build_response`]):
//! its frames binned to at most [`metric::MAX_SIDE`] pixels, each divided by
//! its own median, and combined. Two responses compare by their ratio
//! ([`metric::compare`]): a light shot under the later one and divided by the
//! earlier one carries `ratio − 1` of its sky background, so every number is
//! a fraction of the background. The comparison splits that into a smooth
//! part (`broad`) and compact dips (`dust`, with its spots), with stars a
//! tracked sky-flat set kept masked first.
//!
//! The same reduction answers two questions about lights:
//!
//! - [`evidence`]: given a change between two flat states (its
//!   [`metric::change_map`]), do a few lights' own background carry it?
//! - [`choice`]: given several candidate responses, such as one rig's flats
//!   at several rotator angles, which one does the lights' background follow?
//!
//! Both weigh the lights against **placebos**: the same fit repeated where
//! nothing changed, so compact structure of the sky itself, such as a
//! nebula, widens the spread and leaves the answer undecided rather than
//! wrong.
//!
//! Everything here is pure: planes and numbers in, plain values out. A host
//! reads the frames, keeps the responses and comparisons, and decides what
//! a verdict means for its library.

pub mod choice;
pub mod evidence;
pub mod metric;

pub use choice::{PairFit, PairOutcome, ResponseChoice, choose_response};
pub use evidence::{Fit, FitOutcome, Side, Verdict};
pub use metric::{Comparison, FlatResponse, Plane, Spot, SpotKind};
