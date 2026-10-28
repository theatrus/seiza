//! The C ABI's flat group plan: a book built from one request's rows, and
//! where each of its lights stands, on [`groups::prepare`] and
//! [`groups::Book::place_light`].
use super::*;

/// A flat group plan: the book a request's rows make, and where each of its
/// lights stands. Indices refer to this plan's arrays, never to host
/// database rows. Keyed outputs are sorted lists of pairs, so their wire
/// format is stable.
#[cfg_attr(feature = "json", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct FlatGroupPlan {
    pub algorithm_version: u32,
    pub imaging: Vec<groups::ImagingSession>,
    pub sessions: Vec<groups::Session>,
    pub families: Vec<groups::Family>,
    pub boundaries: Vec<groups::Boundary>,
    pub events: Vec<groups::Event>,
    pub overrides: Vec<groups::OverrideState>,
    pub outliers: BTreeSet<usize>,
    pub spanning: BTreeSet<usize>,
    pub angle_terms: Vec<(String, groups::AngleTerm)>,
    pub epochs: Vec<(String, Vec<groups::Epoch>)>,
    pub pairs_to_compare: Vec<(usize, usize)>,
    pub sessions_in_windows: Vec<(usize, usize, String)>,
    pub sessions_needing_evidence: Vec<(usize, usize)>,
    pub lights: Vec<groups::LightResult>,
}

/// Build measured families and changes, then place the requested lights.
/// Missing measurements remain missing: no metadata-only verdict is invented.
pub fn flat_group_plan(request: groups::Request) -> crate::Result<FlatGroupPlan> {
    let lights = request.lights.clone();
    let book = groups::prepare(request)?;
    let lights = lights
        .iter()
        .map(|light| book.place_light(light))
        .collect::<crate::Result<Vec<_>>>()?;
    let pairs_to_compare = book.pairs_to_compare();
    let sessions_in_windows = book.sessions_in_windows();
    let sessions_needing_evidence = book.sessions_needing_evidence();
    let mut angle_terms: Vec<_> = book.angle_terms.into_iter().collect();
    angle_terms.sort_by(|a, b| a.0.cmp(&b.0));
    let mut epochs: Vec<_> = book.epochs.by_rig.into_iter().collect();
    epochs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(FlatGroupPlan {
        algorithm_version: groups::ALGORITHM_VERSION,
        imaging: book.imaging,
        sessions: book.sessions,
        families: book.families,
        boundaries: book.boundaries,
        events: book.events,
        overrides: book.overrides,
        outliers: book.outliers,
        spanning: book.spanning,
        angle_terms,
        epochs,
        pairs_to_compare,
        sessions_in_windows,
        sessions_needing_evidence,
        lights,
    })
}
