//! One channel's sessions stacked on one machine, each step skipped when
//! the store already holds its output.

use crate::partial::tile_identity;
use crate::session::PARTIAL;
use crate::{
    ChannelReference, Digest, Error, PartialStack, Result, Session, SessionOptions, SessionStep,
    Store,
};
use seiza_stacking::FrameRejectionReason;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The file name a tile stack is stored under.
const TILE: &str = "tile.fits";

/// One channel's sessions on its reference grid: what [`run_channel`]
/// stacks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelPlan {
    /// The image every session aligns to and the light they are normalized
    /// against.
    pub reference: ChannelReference,
    /// The sessions, each stacked into a partial stack of its own.
    pub sessions: Vec<Session>,
    /// How every session is stacked.
    #[serde(default)]
    pub options: SessionOptions,
}

/// What [`run_channel`] is doing, for a host's progress display.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelProgress {
    /// Hashing session `session`'s files to compute its key.
    Planning { session: usize },
    /// The store holds session `session`'s partial stack already.
    Reused { session: usize, key: Digest },
    /// Stacking session `session`.
    Stacking { session: usize, key: Digest },
    /// Combining this many partial stacks.
    Combining { stacks: usize },
}

/// One session's part in a channel.
#[derive(Clone, Debug)]
pub struct SessionOutcome {
    /// The session's step key.
    pub key: Digest,
    /// Its partial stack in the store, or `None` when it accepted no light.
    pub path: Option<PathBuf>,
    /// Whether the store held it already.
    pub reused: bool,
    /// The key of the step over the session's first lights it was extended
    /// from, when it was ([`crate::SessionOptions::keep_context`]).
    pub extended_from: Option<Digest>,
    /// The lights it turned away, and why: those it pushed, so none when it
    /// was reused and only the new ones when it was extended.
    pub rejected: Vec<(PathBuf, FrameRejectionReason)>,
}

/// A channel's stack, combined from its sessions' partial stacks.
#[derive(Clone, Debug)]
pub struct ChannelOutcome {
    /// The tile stack.
    pub stack: PartialStack,
    /// Where the store keeps it.
    pub path: PathBuf,
    /// Whether the store held it already.
    pub reused: bool,
    /// Each session's part, in the plan's order.
    pub sessions: Vec<SessionOutcome>,
}

/// Stack a channel on this machine: plan each session's step, stack each
/// one the store does not hold yet into a partial stack stored under its
/// key, then combine every partial stack into the channel's tile stack,
/// also stored under its key.
///
/// Running a plan again does only what is missing: a session added since
/// is the only one stacked, and the tile is combined again from the stored
/// partial stacks, one read at a time. Once nothing is missing, nothing is
/// stacked or combined. Each run still reads every light once, to hash it
/// for its session's key. A session that accepts no light adds nothing; a
/// channel none of whose sessions do is [`Error::Empty`].
pub fn run_channel(
    plan: &ChannelPlan,
    store: &Store,
    mut progress: impl FnMut(ChannelProgress),
) -> Result<ChannelOutcome> {
    if plan.sessions.is_empty() {
        return Err(Error::Invalid(
            "a channel needs at least one session".into(),
        ));
    }
    let mut sessions = Vec::with_capacity(plan.sessions.len());
    let mut identities = Vec::with_capacity(plan.sessions.len());
    for (index, session) in plan.sessions.iter().enumerate() {
        progress(ChannelProgress::Planning { session: index });
        let step = SessionStep::plan(&plan.reference, session, &plan.options)?;
        let key = step.key;
        let outcome = match store.find(&key, PARTIAL)? {
            Some(path) => {
                progress(ChannelProgress::Reused {
                    session: index,
                    key,
                });
                SessionOutcome {
                    key,
                    path: Some(path),
                    reused: true,
                    extended_from: None,
                    rejected: Vec::new(),
                }
            }
            None => {
                progress(ChannelProgress::Stacking {
                    session: index,
                    key,
                });
                step.run_into(store)?
            }
        };
        if let Some(path) = &outcome.path {
            identities.push((PartialStack::read_identity(path)?, path.clone()));
        }
        sessions.push(outcome);
    }
    if identities.is_empty() {
        return Err(Error::Empty);
    }
    // Two sessions with the same frames and recipe are one step, stored
    // once, and combined once.
    identities.sort_by_key(|(identity, _)| identity.step_key);
    identities.dedup_by_key(|(identity, _)| identity.step_key);
    let key = tile_identity(
        &identities
            .iter()
            .map(|(identity, _)| identity)
            .collect::<Vec<_>>(),
    )
    .step_key;
    if let Some(path) = store.find(&key, TILE)? {
        return Ok(ChannelOutcome {
            stack: PartialStack::open(&path)?,
            path,
            reused: true,
            sessions,
        });
    }
    progress(ChannelProgress::Combining {
        stacks: identities.len(),
    });
    let paths = identities.iter().map(|(_, path)| path).collect::<Vec<_>>();
    let stack = PartialStack::combine_files(&paths)?;
    let path = store.put(&stack, TILE)?;
    Ok(ChannelOutcome {
        stack,
        path,
        reused: false,
        sessions,
    })
}
