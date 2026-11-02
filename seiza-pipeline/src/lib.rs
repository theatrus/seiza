//! Distributed stacking: one target or mosaic stacked across many machines
//! and many nights, kept current as frames arrive.
//!
//! The unit of work is the [`PartialStack`]: one session's frames
//! integrated on one grid, holding the weighted mean and the sums that let
//! it be combined with others ([`PartialStack::combine`]), and header cards
//! ([`Identity`]) that say what made it. Work runs where the frames are, and
//! only partial stacks move.
//!
//! Every step's output is named by a key, a digest of what the step reads,
//! its options and the output versions ([`OUTPUT_VERSION`],
//! `seiza_stacking::OUTPUT_VERSION`). A step is done when a file carrying its
//! key exists, so a run resumes by computing its keys and running only the
//! steps whose files are missing ([`Store`]).
//!
//! This is phase 1 of the design in `docs/design/distributed-stacking.md`:
//! partial stacks on one channel's reference grid, which every session
//! registers to ([`SessionStep`]), combined into one stack, and run on one
//! machine ([`run_channel`]).

mod channel;
mod digest;
mod identity;
mod partial;
mod session;
mod store;

pub use channel::{ChannelOutcome, ChannelPlan, ChannelProgress, SessionOutcome, run_channel};
pub use digest::{Digest, DigestParseError};
pub use identity::{Estimator, Identity, StackKind, WeightSource};
pub use partial::{PartialStack, carried_cards};
pub use session::{
    ChannelReference, Reintegration, Session, SessionOptions, SessionStack, SessionStep,
};
pub use store::Store;

use std::path::PathBuf;

/// The version of this crate whose output a build reproduces. A key holds
/// it, so a build that changes what a step writes makes new keys, and the
/// steps run again. It moves only forward, to the release that ships a
/// change, as `seiza_stacking::OUTPUT_VERSION` does; `tests/output_version.rs`
/// holds it to the output it promises.
pub const OUTPUT_VERSION: &str = "0.1.0";

/// The schema of the identity cards, `SEIZAPVR`. A reader refuses a file
/// with a newer one.
pub const SCHEMA_VERSION: u32 = 1;

/// What went wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An image file could not be read or written.
    #[error(transparent)]
    Image(#[from] seiza_image_io::Error),
    /// Stacking refused or failed.
    #[error(transparent)]
    Stack(#[from] seiza_stacking::Error),
    /// A file could not be read to hash it, or a store's directory made.
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A stack file's identity card is missing or unreadable.
    #[error("card {keyword} {problem}")]
    Card {
        keyword: &'static str,
        problem: String,
    },
    /// A stack file was written under a newer card schema than this build
    /// reads.
    #[error("the stack's cards are schema {0}, newer than this build reads")]
    Schema(u32),
    /// Stacks that cannot be combined: on other grids or of other shapes.
    #[error("{0}")]
    Incompatible(String),
    /// No light of a session was accepted, so it has no partial stack.
    #[error("no light of the session was accepted")]
    Empty,
    /// An input or a value is not one this crate can use.
    #[error("{0}")]
    Invalid(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;
