//! What a stack's frames record of how they were taken, and what the stack
//! can say of them all.

use crate::{FrameMetadata, fits};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use serde::{Deserialize, Serialize};

/// Cards that count the frames an image combines, each as some stacking
/// program writes it.
const COUNT_HEADER_KEYS: [&str; 3] = ["STACKCNT", "NUMFRAME", "NCOMBINE"];

/// How a frame was taken, or the frames of a stack together, as far as their
/// files say. [`crate::write_fits_f32`] writes a stack's as `LIVETIME` and
/// `FILTER`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Acquisition {
    /// The exposure in seconds: of a stack, the sum of its accepted frames'.
    /// `None` unless every frame's is known, so it is never short.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_exposure_seconds: Option<f64>,
    /// The filter. Of a stack, the one every accepted frame was taken
    /// through, spelled as the first frame spells it: `None` when the frames
    /// differ, letter case aside, or any of them does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
}

impl Acquisition {
    /// The acquisition `headers` and the `metadata` read from them record.
    ///
    /// `LIVETIME` is the exposure of a frame that is itself a stack. A stack
    /// without one has no known exposure: its `EXPTIME` is one frame's
    /// length in some programs and the total in others.
    pub(crate) fn of_frame(headers: &[(String, HeaderValue)], metadata: &FrameMetadata) -> Self {
        let number = |key: &str| {
            headers
                .iter()
                .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
                .and_then(|(_, value)| value.as_f64())
                .filter(|value| value.is_finite())
        };
        let combined = COUNT_HEADER_KEYS
            .iter()
            .any(|key| number(key).is_some_and(|count| count > 1.0));
        let single = metadata
            .signature
            .exposure_seconds
            .filter(|_| !combined)
            .filter(|seconds| seconds.is_finite());
        Self {
            total_exposure_seconds: number("LIVETIME")
                .or(single)
                .filter(|&seconds| seconds > 0.0),
            filter: metadata.signature.filter.clone(),
        }
    }

    /// Whether anything is recorded.
    pub(crate) fn is_known(&self) -> bool {
        self.total_exposure_seconds.is_some() || self.filter.is_some()
    }

    /// The total exposure and shared filter of `parts`, each with its own:
    /// a total only when each has one, a filter only when they agree.
    pub fn combined<'a>(parts: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut parts = parts.into_iter();
        let Some(first) = parts.next() else {
            return Self::default();
        };
        let mut total = first.total_exposure_seconds;
        let mut filter = first.filter.as_deref();
        for part in parts {
            total = total
                .zip(part.total_exposure_seconds)
                .map(|(total, seconds)| total + seconds);
            let other = part.filter.as_deref();
            filter = filter
                .filter(|filter| other.is_some_and(|other| other.eq_ignore_ascii_case(filter)));
        }
        Self {
            total_exposure_seconds: total,
            filter: filter.map(str::to_owned),
        }
    }

    /// What a stack file's `LIVETIME` and `FILTER` cards record.
    pub fn from_cards(cards: &[(String, HeaderValue)]) -> Self {
        let card = |key: &str| {
            cards
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| value)
        };
        Self {
            total_exposure_seconds: card("LIVETIME")
                .and_then(HeaderValue::as_f64)
                .filter(|seconds| seconds.is_finite() && *seconds > 0.0),
            filter: card("FILTER")
                .and_then(HeaderValue::as_str)
                .map(str::trim)
                .filter(|filter| !filter.is_empty())
                .map(str::to_owned),
        }
    }

    /// The `LIVETIME` and `FILTER` cards that record it. An unknown value
    /// writes no card, nor does a filter a FITS card cannot hold, which is
    /// printable ASCII only.
    pub fn cards(&self) -> Vec<WriteHeaderCard> {
        let mut cards = Vec::new();
        if let Some(seconds) = self.total_exposure_seconds {
            cards.push(fits::float_card(
                "LIVETIME",
                seconds,
                "[s] total exposure of accepted frames",
            ));
        }
        if let Some(filter) = &self.filter
            && filter.bytes().all(|byte| (b' '..=b'~').contains(&byte))
        {
            cards.push(fits::string_card(
                "FILTER",
                filter,
                "filter of every accepted frame",
            ));
        }
        cards
    }
}
