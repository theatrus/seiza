//! The header cards that say what a stack file is, what made it, and
//! whether it is current: enough to resume a run from the headers alone.

use crate::{Digest, Error, Result, SCHEMA_VERSION};
use seiza_fits::{HeaderValue, WriteHeaderCard};
use serde::{Deserialize, Serialize};

/// What a stack file is, as its `SEIZAPST` card names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StackKind {
    /// One session's frames integrated on one grid.
    Partial,
    /// Partial stacks on one grid, combined.
    Tile,
    /// Tile stacks assembled into one image.
    Mosaic,
}

/// How a stack's mean was estimated, as its `ESTIMATR` card names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Estimator {
    /// A running weighted mean, which more frames can extend.
    Additive,
    /// A mean after rejection that saw every frame at once, which more
    /// frames cannot extend: the stack can only be combined with others.
    Clipped,
}

/// Where a stack's weight came from, as its `WGHTSRC` card names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightSource {
    /// The stack's own sum of frame weights.
    Stack,
    /// Built from measured noise, for a stack made by other software.
    Measured,
}

macro_rules! word {
    ($type:ty, $what:literal, $($variant:ident => $word:literal),+ $(,)?) => {
        impl std::fmt::Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(match self {
                    $(Self::$variant => $word,)+
                })
            }
        }

        impl std::str::FromStr for $type {
            type Err = Error;

            /// The word, in either case.
            fn from_str(word: &str) -> Result<Self> {
                $(if word.eq_ignore_ascii_case($word) {
                    return Ok(Self::$variant);
                })+
                Err(Error::Invalid(format!(concat!("{:?} is not ", $what), word)))
            }
        }
    };
}

word!(StackKind, "a stack kind", Partial => "partial", Tile => "tile", Mosaic => "mosaic");
word!(Estimator, "an estimator", Additive => "additive", Clipped => "clipped");
word!(WeightSource, "a weight source", Stack => "stack", Measured => "measured");

/// The cards that identify a stack file: what it is, which versions made it,
/// the key of the step that made it and digests of what that step read.
///
/// A host reads them from the header alone ([`crate::PartialStack::read_identity`])
/// to learn whether a file is current, without decoding its pixels. A key
/// is a digest of a step's inputs, options and output versions, so a file
/// carrying the key a plan computes is that step's output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    /// `SEIZAPST`.
    pub kind: StackKind,
    /// `SEIZASTK`: `seiza_stacking::OUTPUT_VERSION` when the frames were
    /// integrated, or `mixed` for a tile of partial stacks that differ.
    pub stacking_version: String,
    /// `SEIZAPIP`: [`crate::OUTPUT_VERSION`] when the file was written.
    pub pipeline_version: String,
    /// `STEPKEY`: the key of the step that made the file.
    pub step_key: Digest,
    /// `INPUTSHA`: the ordered inputs, frame digests for a partial stack and
    /// partial stack keys for a tile.
    pub inputs: Digest,
    /// `NINPUTS`: how many inputs `inputs` covers, accepted or not.
    pub input_count: u32,
    /// `NCOMBINE`: the frames integrated.
    pub combined: u32,
    /// `CALIBSHA`: the calibration recipe, its masters by their digests.
    pub calibration: Option<Digest>,
    /// `OPTSSHA`: the options that change the pixels.
    pub options: Digest,
    /// `HOSTSHA`: a fingerprint the host supplies for decisions of its own,
    /// which Seiza only compares.
    pub host: Option<Digest>,
    /// `ESTIMATR`.
    pub estimator: Estimator,
    /// `CONTEXT`: the live-stack context that can extend an additive stack,
    /// when one was kept.
    pub context: Option<Digest>,
    /// `CONTRIB`: who contributed the frames, a handle they chose.
    pub contributor: Option<String>,
    /// `WGHTSRC`.
    pub weight_source: WeightSource,
    /// `GRIDSHA`: the grid the stack lies on.
    pub grid: Digest,
}

const KIND: &str = "SEIZAPST";
const SCHEMA: &str = "SEIZAPVR";
const STACKING: &str = "SEIZASTK";
const PIPELINE: &str = "SEIZAPIP";
const STEP_KEY: &str = "STEPKEY";
const INPUTS: &str = "INPUTSHA";
const INPUT_COUNT: &str = "NINPUTS";
const COMBINED: &str = "NCOMBINE";
const CALIBRATION: &str = "CALIBSHA";
const OPTIONS: &str = "OPTSSHA";
const HOST: &str = "HOSTSHA";
const ESTIMATOR: &str = "ESTIMATR";
const CONTEXT: &str = "CONTEXT";
const CONTRIBUTOR: &str = "CONTRIB";
const WEIGHT_SOURCE: &str = "WGHTSRC";
const GRID: &str = "GRIDSHA";

/// Every keyword the identity writes, which carried cards must not repeat.
pub(crate) const KEYWORDS: [&str; 16] = [
    KIND,
    SCHEMA,
    STACKING,
    PIPELINE,
    STEP_KEY,
    INPUTS,
    INPUT_COUNT,
    COMBINED,
    CALIBRATION,
    OPTIONS,
    HOST,
    ESTIMATOR,
    CONTEXT,
    CONTRIBUTOR,
    WEIGHT_SOURCE,
    GRID,
];

impl Identity {
    /// The cards, in the order a file holds them. Absent values write no
    /// card.
    pub fn cards(&self) -> Vec<WriteHeaderCard> {
        let text = |keyword: &str, value: String, comment: &str| {
            WriteHeaderCard::new(keyword, HeaderValue::String(value)).with_comment(comment)
        };
        let number = |keyword: &str, value: u32, comment: &str| {
            WriteHeaderCard::new(keyword, HeaderValue::Integer(i64::from(value)))
                .with_comment(comment)
        };
        let word = |keyword: &str, value: &dyn std::fmt::Display, comment: &str| {
            text(keyword, value.to_string().to_ascii_uppercase(), comment)
        };
        let mut cards = vec![
            word(KIND, &self.kind, "Seiza stack kind"),
            number(SCHEMA, SCHEMA_VERSION, "schema of the SEIZAP cards"),
            text(
                STACKING,
                self.stacking_version.clone(),
                "seiza-stacking output version",
            ),
            text(
                PIPELINE,
                self.pipeline_version.clone(),
                "seiza-pipeline output version",
            ),
            text(STEP_KEY, self.step_key.to_string(), ""),
            text(INPUTS, self.inputs.to_string(), ""),
            number(INPUT_COUNT, self.input_count, "inputs INPUTSHA covers"),
            number(COMBINED, self.combined, "frames integrated"),
        ];
        if let Some(calibration) = &self.calibration {
            cards.push(text(CALIBRATION, calibration.to_string(), ""));
        }
        cards.push(text(OPTIONS, self.options.to_string(), ""));
        if let Some(host) = &self.host {
            cards.push(text(HOST, host.to_string(), ""));
        }
        cards.push(word(
            ESTIMATOR,
            &self.estimator,
            "how the mean was estimated",
        ));
        if let Some(context) = &self.context {
            cards.push(text(CONTEXT, context.to_string(), ""));
        }
        if let Some(contributor) = &self.contributor {
            cards.push(text(CONTRIBUTOR, contributor.clone(), "contributor"));
        }
        cards.push(word(
            WEIGHT_SOURCE,
            &self.weight_source,
            "where WEIGHT came from",
        ));
        cards.push(text(GRID, self.grid.to_string(), ""));
        cards
    }

    /// Read the identity from a header's cards. A file written under a newer
    /// schema than this build reads is [`Error::Schema`]; a missing or
    /// unreadable card is [`Error::Card`].
    pub fn from_cards(cards: &[(String, HeaderValue)]) -> Result<Self> {
        let value = |keyword: &'static str| {
            cards
                .iter()
                .find(|(key, _)| key == keyword)
                .map(|(_, value)| value)
        };
        let text = |keyword: &'static str| -> Result<Option<String>> {
            match value(keyword) {
                None => Ok(None),
                Some(value) => value
                    .as_str()
                    .map(|text| Some(text.trim_end().to_string()))
                    .ok_or_else(|| card_error(keyword, "is not a string")),
            }
        };
        let required = |keyword: &'static str| -> Result<String> {
            text(keyword)?.ok_or_else(|| card_error(keyword, "is missing"))
        };
        let digest = |keyword: &'static str| -> Result<Option<Digest>> {
            text(keyword)?
                .map(|text| {
                    text.parse()
                        .map_err(|_| card_error(keyword, "is not a digest"))
                })
                .transpose()
        };
        let required_digest = |keyword: &'static str| -> Result<Digest> {
            digest(keyword)?.ok_or_else(|| card_error(keyword, "is missing"))
        };
        let number = |keyword: &'static str| -> Result<u32> {
            value(keyword)
                .ok_or_else(|| card_error(keyword, "is missing"))?
                .as_i64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| card_error(keyword, "is not a count"))
        };
        let word = |keyword: &'static str| -> Result<String> { required(keyword) };
        let schema = number(SCHEMA)?;
        if schema > SCHEMA_VERSION {
            return Err(Error::Schema(schema));
        }
        Ok(Self {
            kind: word(KIND)?
                .parse()
                .map_err(|_| card_error(KIND, "names no stack kind"))?,
            stacking_version: required(STACKING)?,
            pipeline_version: required(PIPELINE)?,
            step_key: required_digest(STEP_KEY)?,
            inputs: required_digest(INPUTS)?,
            input_count: number(INPUT_COUNT)?,
            combined: number(COMBINED)?,
            calibration: digest(CALIBRATION)?,
            options: required_digest(OPTIONS)?,
            host: digest(HOST)?,
            estimator: word(ESTIMATOR)?
                .parse()
                .map_err(|_| card_error(ESTIMATOR, "names no estimator"))?,
            context: digest(CONTEXT)?,
            contributor: text(CONTRIBUTOR)?,
            weight_source: word(WEIGHT_SOURCE)?
                .parse()
                .map_err(|_| card_error(WEIGHT_SOURCE, "names no weight source"))?,
            grid: required_digest(GRID)?,
        })
    }

    /// An error unless every value fits a FITS card: a contributor's
    /// handle is printable ASCII.
    pub(crate) fn check(&self) -> Result<()> {
        check_contributor(self.contributor.as_deref())
    }
}

/// An error unless a contributor's handle, when there is one, fits a FITS
/// card: printable ASCII, not empty.
pub(crate) fn check_contributor(contributor: Option<&str>) -> Result<()> {
    if let Some(contributor) = contributor
        && (contributor.is_empty()
            || !contributor
                .bytes()
                .all(|byte| (b' '..=b'~').contains(&byte)))
    {
        return Err(Error::Invalid(format!(
            "contributor {contributor:?} is not printable ASCII"
        )));
    }
    Ok(())
}

fn card_error(keyword: &'static str, problem: &str) -> Error {
    Error::Card {
        keyword,
        problem: problem.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An identity with every card a partial stack usually has.
    pub(crate) fn identity() -> Identity {
        Identity {
            kind: StackKind::Partial,
            stacking_version: "0.51.0".into(),
            pipeline_version: crate::OUTPUT_VERSION.into(),
            step_key: Digest::of_bytes(b"step"),
            inputs: Digest::of_bytes(b"inputs"),
            input_count: 12,
            combined: 11,
            calibration: Some(Digest::of_bytes(b"masters")),
            options: Digest::of_bytes(b"options"),
            host: None,
            estimator: Estimator::Clipped,
            context: None,
            contributor: Some("night owl".into()),
            weight_source: WeightSource::Stack,
            grid: Digest::of_bytes(b"grid"),
        }
    }

    fn read(cards: &[WriteHeaderCard]) -> Vec<(String, HeaderValue)> {
        cards
            .iter()
            .map(|card| (card.keyword().to_string(), card.value().clone()))
            .collect()
    }

    #[test]
    fn cards_round_trip() {
        let identity = identity();
        let cards = identity.cards();
        assert!(cards.iter().all(|card| card.keyword().len() <= 8));
        assert!(cards.iter().all(|card| KEYWORDS.contains(&card.keyword())));
        assert!(!cards.iter().any(|card| card.keyword() == HOST));
        assert_eq!(Identity::from_cards(&read(&cards)).unwrap(), identity);
        let full = Identity {
            host: Some(Digest::of_bytes(b"host")),
            context: Some(Digest::of_bytes(b"context")),
            calibration: None,
            contributor: None,
            ..identity
        };
        assert_eq!(Identity::from_cards(&read(&full.cards())).unwrap(), full);
    }

    #[test]
    fn a_newer_schema_or_a_bad_card_is_refused() {
        let mut cards = read(&identity().cards());
        let schema = cards.iter_mut().find(|(key, _)| key == SCHEMA).unwrap();
        schema.1 = HeaderValue::Integer(i64::from(SCHEMA_VERSION) + 1);
        assert!(matches!(
            Identity::from_cards(&cards),
            Err(Error::Schema(version)) if version == SCHEMA_VERSION + 1
        ));
        let mut cards = read(&identity().cards());
        cards.retain(|(key, _)| key != STEP_KEY);
        assert!(matches!(
            Identity::from_cards(&cards),
            Err(Error::Card {
                keyword: STEP_KEY,
                ..
            })
        ));
        let mut cards = read(&identity().cards());
        let kind = cards.iter_mut().find(|(key, _)| key == KIND).unwrap();
        kind.1 = HeaderValue::String("STACK".into());
        assert!(matches!(
            Identity::from_cards(&cards),
            Err(Error::Card { keyword: KIND, .. })
        ));
    }

    #[test]
    fn words_parse_in_either_case() {
        assert_eq!("PARTIAL".parse::<StackKind>().unwrap(), StackKind::Partial);
        assert_eq!("tile".parse::<StackKind>().unwrap(), StackKind::Tile);
        assert_eq!(Estimator::Additive.to_string(), "additive");
        assert_eq!(
            serde_json::to_string(&WeightSource::Measured).unwrap(),
            "\"measured\""
        );
        assert!("weighted".parse::<WeightSource>().is_err());
    }
}
