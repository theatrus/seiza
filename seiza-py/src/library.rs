//! The calibration library's native API (`seiza_calibration::library`), as
//! Rust hosts hold it: settings, stored quality, a library that selects for
//! each light, the sessions its selections build, the flat retry, and the
//! measuring stages of flat grouping. Each method calls one native function;
//! records are dicts converted directly to the core's types. The C ABI's
//! one-call JSON workflows are not offered here.
use crate::convert::{from_python, to_python};
use crate::flat_response::{as_plane, plane_array};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use seiza_calibration::flat_response::{Plane, groups, metric};
use seiza_calibration::library::{self, rules};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// `value`, or the default when it is absent or None.
fn optional<T: serde::de::DeserializeOwned + Default>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<T> {
    match value {
        Some(value) if !value.is_none() => from_python(value, name),
        _ => Ok(T::default()),
    }
}

fn word<T: std::str::FromStr>(text: &str) -> PyResult<T>
where
    T::Err: std::fmt::Display,
{
    text.parse().map_err(invalid)
}

fn setting(key: &str) -> PyResult<rules::SettingKey> {
    word(key)
}

fn associations(flat: &str, dark: &str) -> PyResult<library::Associations> {
    Ok(library::Associations {
        flat: word(flat)?,
        dark: word(dark)?,
    })
}

// --- settings --------------------------------------------------------------

/// One rig's settings layer: what the person saw it called, and its values.
#[derive(Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RigLayerRecord {
    name: Option<String>,
    values: rules::RuleValues,
}

/// The calibration settings in force: the server's layer and each rig's,
/// checked the way a settings panel saves them.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationRules",
    module = "seiza"
)]
pub(crate) struct PyCalibrationRules {
    book: rules::RuleBook,
}

#[pymethods]
impl PyCalibrationRules {
    #[new]
    #[pyo3(signature = (global_settings=None, rigs=None))]
    fn new(
        global_settings: Option<&Bound<'_, PyAny>>,
        rigs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let global: rules::RuleValues = optional(global_settings, "global settings")?;
        let rigs: BTreeMap<String, RigLayerRecord> = optional(rigs, "rig settings")?;
        let rigs = rigs
            .into_iter()
            .map(|(uuid, layer)| {
                (
                    uuid,
                    rules::RigLayer {
                        name: layer.name,
                        values: layer.values,
                    },
                )
            })
            .collect();
        Ok(Self {
            book: rules::RuleBook::validated(global, rigs).map_err(invalid)?,
        })
    }

    /// The rules a calibration frame of this rig is matched and graded by.
    fn for_rig(&self, rig_uuid: &str) -> PyCalibrationRigRules {
        PyCalibrationRigRules {
            rules: self.book.for_rig(rig_uuid),
        }
    }

    /// The rules a light is matched and checked with: its rig's.
    fn for_light(&self, light: &Bound<'_, PyAny>) -> PyResult<PyCalibrationRigRules> {
        let light: library::Light = from_python(light, "light")?;
        Ok(PyCalibrationRigRules {
            rules: self.book.for_light(&light),
        })
    }

    /// The rules every rig without settings of its own follows.
    fn global_rules(&self) -> PyCalibrationRigRules {
        PyCalibrationRigRules {
            rules: self.book.global(),
        }
    }

    /// Whether a rig sets `key` for itself rather than inheriting it.
    fn rig_sets(&self, rig_uuid: &str, key: &str) -> PyResult<bool> {
        Ok(self.book.rig_sets(rig_uuid, setting(key)?))
    }

    /// Whether any setting grading reads differs between the two.
    fn grades_differ(&self, other: PyRef<'_, PyCalibrationRules>) -> bool {
        self.book.grades_differ(&other.book)
    }

    /// A key of every setting, or None when nothing is set.
    #[getter]
    fn digest(&self) -> Option<String> {
        self.book.digest()
    }

    /// The server's layer, as given.
    #[getter]
    fn global_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, self.book.global_values())
    }

    /// Each rig's layer, as given: `{rig_uuid: {"name", "values"}}`.
    #[getter]
    fn rigs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rigs: BTreeMap<&String, RigLayerRecord> = self
            .book
            .rigs()
            .iter()
            .map(|(uuid, layer)| {
                (
                    uuid,
                    RigLayerRecord {
                        name: layer.name.clone(),
                        values: layer.values.clone(),
                    },
                )
            })
            .collect();
        to_python(py, &rigs)
    }
}

/// Every setting resolved for one rig: its own value, else the server's,
/// else the default.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationRigRules",
    module = "seiza"
)]
pub(crate) struct PyCalibrationRigRules {
    rules: Arc<rules::Rules>,
}

#[pymethods]
impl PyCalibrationRigRules {
    /// The rig these were resolved for, when it has settings of its own.
    #[getter]
    fn rig<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.rules.rig)
    }

    /// One setting's resolved value.
    fn value<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, self.rules.value(setting(key)?))
    }

    /// Where one setting's value came from: "default", "global" or "rig".
    fn source(&self, key: &str) -> PyResult<&'static str> {
        Ok(self.rules.source(setting(key)?).as_str())
    }

    /// Who set a setting to something other than its default: "global" or
    /// "rig", or None.
    fn set_by(&self, key: &str) -> PyResult<Option<&'static str>> {
        Ok(self
            .rules
            .set_by(setting(key)?)
            .map(|source| source.as_str()))
    }

    /// How a light check is set to act: "flag", "advisory" or "off".
    fn mode(&self, key: &str) -> PyResult<&'static str> {
        match setting(key)? {
            rules::SettingKey::Check(check) => Ok(self.rules.mode(check).as_str()),
            _ => Err(invalid(format!("{key} is not a light check"))),
        }
    }

    /// The settings that differ from their defaults, in the panel's order.
    fn notes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.rules.notes())
    }

    /// The tolerances a candidate is matched to a light with.
    fn tolerances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.rules.tolerances())
    }

    /// The tolerances the frames of one master are held to among themselves.
    fn coherence_tolerances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.rules.coherence_tolerances())
    }

    /// The limits flat grouping splits this rig's flats by.
    fn grouping_limits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.rules.grouping_limits())
    }

    /// The matching settings that differ from their defaults, as one key.
    #[getter]
    fn matching_digest(&self) -> Option<String> {
        self.rules.matching_digest()
    }
}

/// The calibration settings a library takes, in the order a settings panel
/// shows them, with their ranges and defaults.
#[pyfunction]
fn calibration_settings_schema(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    to_python(py, &rules::fields())
}

/// Judge one setting value as a settings panel saves it.
#[pyfunction]
fn calibration_setting_check<'py>(
    py: Python<'py>,
    key: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let value: serde_json::Value = from_python(value, "setting value")?;
    to_python(py, &rules::check_setting(key, &value))
}

// --- nights and quality ------------------------------------------------------

/// Where each frame's night begins, by rig and frame.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationNights",
    module = "seiza"
)]
pub(crate) struct PyCalibrationNights {
    nights: library::NightShifts,
}

#[pymethods]
impl PyCalibrationNights {
    /// The `boundaries` given, one per frame; with `frames` and `rules`,
    /// each rig's `night_zone` setting in force over its frames.
    #[new]
    #[pyo3(signature = (boundaries=None, *, frames=None, rules=None))]
    fn new(
        boundaries: Option<&Bound<'_, PyAny>>,
        frames: Option<&Bound<'_, PyAny>>,
        rules: Option<PyRef<'_, PyCalibrationRules>>,
    ) -> PyResult<Self> {
        let rows: Vec<groups::NightBoundary> = optional(boundaries, "night boundaries")?;
        let mut nights = library::NightShifts::checked(rows).map_err(invalid)?;
        if let Some(rules) = rules {
            let frames: Vec<library::CalibrationFrame> = optional(frames, "frames")?;
            nights = nights.with_zones(&frames, &rules.book);
        } else if frames.is_some_and(|frames| !frames.is_none()) {
            return Err(invalid("night zones need the rules their frames follow"));
        }
        Ok(Self { nights })
    }

    /// Seconds past midnight UTC at which this frame's night begins.
    fn boundary(&self, rig_uuid: &str, frame_uuid: &str) -> i64 {
        library::NightSplit::boundary(&self.nights, rig_uuid, frame_uuid)
    }

    /// The night, `YYYY-MM-DD`, a frame shot at `at` belongs to.
    fn night_of_frame(&self, rig_uuid: &str, frame_uuid: &str, at: i64) -> String {
        library::NightSplit::night_of_frame(&self.nights, rig_uuid, frame_uuid, at)
    }

    /// The boundaries held, one per frame.
    fn boundaries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.nights.boundaries())
    }
}

/// Fresh grades by frame UUID, as `CalibrationQuality.regrade` judged them.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationGrades",
    module = "seiza"
)]
pub(crate) struct PyCalibrationGrades {
    judged: HashMap<String, library::Judged>,
}

#[pymethods]
impl PyCalibrationGrades {
    fn __len__(&self) -> usize {
        self.judged.len()
    }

    /// One frame's grade, or None when it was not graded afresh.
    fn get<'py>(&self, py: Python<'py>, frame_uuid: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.judged.get(frame_uuid))
    }

    /// Every grade, by frame UUID.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.judged)
    }
}

/// The stored quality of a library's frames, by frame.
#[pyclass(skip_from_py_object, name = "CalibrationQuality", module = "seiza")]
pub(crate) struct PyCalibrationQuality {
    quality: library::QualityIndex,
}

#[pymethods]
impl PyCalibrationQuality {
    /// Index `records`, one per frame.
    #[new]
    #[pyo3(signature = (records=None))]
    fn new(records: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let rows: Vec<library::QualityRecord> = optional(records, "quality records")?;
        Ok(Self {
            quality: library::QualityIndex::new(rows).map_err(invalid)?,
        })
    }

    /// One frame's stored record, or None.
    fn get<'py>(&self, py: Python<'py>, frame_uuid: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.quality.get(frame_uuid))
    }

    /// Grade afresh every frame whose stored measurement is current, and
    /// count those frames judged from now on.
    fn regrade(
        &mut self,
        py: Python<'_>,
        frames: &Bound<'_, PyAny>,
        rules: PyRef<'_, PyCalibrationRules>,
        nights: PyRef<'_, PyCalibrationNights>,
    ) -> PyResult<PyCalibrationGrades> {
        let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
        let (book, nights) = (&rules.book, &nights.nights);
        let quality = &mut self.quality;
        let judged = py.detach(|| quality.regrade(&frames, book, nights));
        Ok(PyCalibrationGrades { judged })
    }

    /// Why `frame` stays out of masters and exports, or None when it serves.
    /// Its own `kept_out` is read as the stray-light mark an older build
    /// left: give frames as the catalog stores them.
    #[pyo3(signature = (frame, judged=None))]
    fn kept_out(
        &self,
        frame: &Bound<'_, PyAny>,
        judged: Option<PyRef<'_, PyCalibrationGrades>>,
    ) -> PyResult<Option<String>> {
        let frame: library::CalibrationFrame = from_python(frame, "frame")?;
        let grade = judged
            .as_ref()
            .and_then(|grades| grades.judged.get(&frame.frame_uuid));
        Ok(self.quality.kept_out(&frame, grade))
    }

    /// `frames`, as the catalog stores them, with each one's kept-out reason
    /// set. Apply it to the stored frames, not to frames it returned.
    #[pyo3(signature = (frames, judged=None))]
    fn keep_out<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        judged: Option<PyRef<'_, PyCalibrationGrades>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
        let empty = HashMap::new();
        let grades = judged.as_ref().map_or(&empty, |grades| &grades.judged);
        self.quality.keep_out(&mut frames, grades);
        to_python(py, &frames)
    }

    /// What is known of `frame`'s measurement: "external", "unmeasured",
    /// "stale", "current" or "failed".
    fn measurement_state(&self, frame: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        let frame: library::CalibrationFrame = from_python(frame, "frame")?;
        Ok(self.quality.measurement_state(&frame).as_str())
    }

    /// Whether a person chose for `frame`, or it carries a current grade.
    fn judged(&self, frame: &Bound<'_, PyAny>) -> PyResult<bool> {
        let frame: library::CalibrationFrame = from_python(frame, "frame")?;
        Ok(self.quality.judged(&frame))
    }
}

// --- flat grouping book --------------------------------------------------------

/// A measured flat grouping book, built from stored rows. Construction and
/// planning perform no I/O.
#[pyclass(skip_from_py_object, frozen, name = "FlatGroups", module = "seiza")]
pub(crate) struct PyFlatGroups {
    book: Arc<groups::Book>,
}

#[pymethods]
impl PyFlatGroups {
    #[new]
    fn new(request: &Bound<'_, PyAny>) -> PyResult<Self> {
        let request = from_python(request, "flat grouping request")?;
        let book = groups::prepare(request).map_err(invalid)?;
        Ok(Self {
            book: Arc::new(book),
        })
    }

    /// The imaging sessions, by index.
    #[getter]
    fn imaging<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.imaging)
    }

    /// The flat sessions, by index.
    #[getter]
    fn sessions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.sessions)
    }

    /// The response families, by index.
    #[getter]
    fn families<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.families)
    }

    /// The boundaries between sessions, by index.
    #[getter]
    fn boundaries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.boundaries)
    }

    /// The measured changes, by index.
    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.events)
    }

    /// The limits a rig's flats were grouped by, or None.
    fn limits<'py>(&self, py: Python<'py>, rig_uuid: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.limits(rig_uuid))
    }

    /// Imaging sessions inside a change's window, with their evidence keys:
    /// `(event, session, key)`, by index.
    fn sessions_in_windows(&self) -> Vec<(usize, usize, String)> {
        self.book.sessions_in_windows()
    }

    /// Imaging sessions inside a change's window with no evidence yet:
    /// `(event, session)`, by index.
    fn sessions_needing_evidence(&self) -> Vec<(usize, usize)> {
        self.book.sessions_needing_evidence()
    }

    /// A person's splits and joins, as applied, with whether each still
    /// finds what it was made for.
    #[getter]
    fn overrides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.overrides)
    }

    /// Boundaries on either side of one night unlike the nights around it,
    /// which measure alike with each other: a hazy dawn, not a change. They
    /// split groups but make no event. By index.
    #[getter]
    fn outliers(&self) -> Vec<usize> {
        self.book.outliers.iter().copied().collect()
    }

    /// Boundaries across two changes another family tells apart: they date
    /// neither, and make no event of their own. By index.
    #[getter]
    fn spanning(&self) -> Vec<usize> {
        self.book.spanning.iter().copied().collect()
    }

    /// Each rig's term for how its families' angles relate, by rig UUID.
    #[getter]
    fn angle_terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.angle_terms)
    }

    /// Each rig's sky-minus-mechanical angle epochs, by rig UUID.
    #[getter]
    fn epochs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.book.epochs.by_rig)
    }

    /// Flat session pairs whose comparison the book still needs, by index.
    fn pairs_to_compare(&self) -> Vec<(usize, usize)> {
        self.book.pairs_to_compare()
    }

    /// Where one light stands: its automatic family among the
    /// `compatible_families` it names, the pixel question that would settle
    /// a close call, a person's resolved choices, and its placement against
    /// each change of its rig and filter.
    fn place_light<'py>(
        &self,
        py: Python<'py>,
        light: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let light: groups::LightQuery = from_python(light, "light")?;
        let placed = self.book.place_light(&light).map_err(invalid)?;
        to_python(py, &placed)
    }
}

// --- library and selection -----------------------------------------------------

/// Every frame of a library, held once so many lights select against it.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationLibrary",
    module = "seiza"
)]
pub(crate) struct PyCalibrationLibrary {
    library: library::Library,
}

#[pymethods]
impl PyCalibrationLibrary {
    /// A library of `frames`, with kept-out reasons decided; `groups`, the
    /// measured flat grouping book, and the correction responses the host
    /// can read. Repeated or empty frame UUIDs, unverifiable sources and
    /// non-finite metadata raise ValueError. Given the `rules`, a book
    /// grouped by other limits than they give its rigs does too.
    #[new]
    #[pyo3(signature = (frames, *, groups=None, available_responses=Vec::new(), rules=None))]
    fn new(
        frames: &Bound<'_, PyAny>,
        groups: Option<PyRef<'_, PyFlatGroups>>,
        available_responses: Vec<String>,
        rules: Option<PyRef<'_, PyCalibrationRules>>,
    ) -> PyResult<Self> {
        let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
        let book = groups.map_or_else(Default::default, |groups| Arc::clone(&groups.book));
        let library =
            library::Library::checked(frames, book, available_responses).map_err(invalid)?;
        if let Some(rules) = rules {
            library.check_limits(&rules.book).map_err(invalid)?;
        }
        Ok(Self { library })
    }

    /// Raise ValueError when the flat grouping book was grouped by other
    /// limits than `rules` give its rigs.
    fn check_limits(&self, rules: PyRef<'_, PyCalibrationRules>) -> PyResult<()> {
        self.library.check_limits(&rules.book).map_err(invalid)
    }

    /// The frames, newest first.
    #[getter]
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.library.frames())
    }

    /// The flat grouping book the library selects with.
    #[getter]
    fn groups(&self) -> PyFlatGroups {
        PyFlatGroups {
            book: Arc::clone(self.library.flat_book()),
        }
    }

    /// What a stack would match to `light` by `rules`; kinds associated
    /// "one_set" are taken whole as one supplied set.
    #[pyo3(signature = (light, rules, *, flat_association="automatic", dark_association="automatic"))]
    fn select(
        &self,
        py: Python<'_>,
        light: &Bound<'_, PyAny>,
        rules: PyRef<'_, PyCalibrationRigRules>,
        flat_association: &str,
        dark_association: &str,
    ) -> PyResult<PyCalibrationSelection> {
        let light: library::Light = from_python(light, "light")?;
        light.check().map_err(invalid)?;
        let associations = associations(flat_association, dark_association)?;
        let rules = Arc::clone(&rules.rules);
        let selection = py.detach(|| self.library.select_associated(&light, rules, associations));
        Ok(PyCalibrationSelection { selection })
    }

    /// Whether the library's frames can stand as one supplied set for each
    /// kind associated "one_set"; raises ValueError when not.
    #[pyo3(signature = (rules, *, flat_association="automatic", dark_association="automatic"))]
    fn check_associations(
        &self,
        rules: PyRef<'_, PyCalibrationRules>,
        flat_association: &str,
        dark_association: &str,
    ) -> PyResult<()> {
        self.library
            .check_associations(
                &rules.book,
                associations(flat_association, dark_association)?,
            )
            .map_err(invalid)
    }

    /// The response families, by index in the book, whose flats a light's
    /// sensor and optics can take.
    fn flat_family_compatible(
        &self,
        light: &Bound<'_, PyAny>,
        rules: PyRef<'_, PyCalibrationRigRules>,
    ) -> PyResult<Vec<usize>> {
        let light: library::Light = from_python(light, "light")?;
        Ok(self.library.flat_family_compatible(&light, &rules.rules))
    }

    /// A person's choices for `light`'s flats, resolved.
    fn light_choices<'py>(
        &self,
        py: Python<'py>,
        light: &Bound<'py, PyAny>,
        rules: PyRef<'_, PyCalibrationRigRules>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let light: library::Light = from_python(light, "light")?;
        to_python(py, &self.library.light_choices(&light, &rules.rules))
    }
}

/// The frames one light matched, per kind, nearest first, and why.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "CalibrationSelection",
    module = "seiza"
)]
pub(crate) struct PyCalibrationSelection {
    selection: library::CalibrationSelection,
}

#[pymethods]
impl PyCalibrationSelection {
    /// The selection of exactly `frames`, each under its kind, in the order
    /// given, matched by `rules`: candidates a host chose or verified
    /// itself. `flat_basis`, `flat_family` and `flat_held_back` carry what a
    /// selection said about its flats.
    #[new]
    #[pyo3(signature = (frames, rules, *, flat_association="automatic", dark_association="automatic", flat_basis=None, flat_family=None, flat_held_back=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frames: &Bound<'_, PyAny>,
        rules: PyRef<'_, PyCalibrationRigRules>,
        flat_association: &str,
        dark_association: &str,
        flat_basis: Option<&Bound<'_, PyAny>>,
        flat_family: Option<&Bound<'_, PyAny>>,
        flat_held_back: Vec<String>,
    ) -> PyResult<Self> {
        let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
        let selection = library::CalibrationSelection {
            flat_basis: optional(flat_basis, "flat basis")?,
            flat_family: optional(flat_family, "flat family")?,
            flat_held_back,
            ..library::CalibrationSelection::of_frames(
                frames,
                Arc::clone(&rules.rules),
                associations(flat_association, dark_association)?,
            )
        };
        Ok(Self { selection })
    }

    #[getter]
    fn bias<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.bias)
    }
    #[getter]
    fn dark<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.dark)
    }
    #[getter]
    fn dark_flat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.dark_flat)
    }
    #[getter]
    fn flat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.flat)
    }
    /// Frames that matched but stay out.
    #[getter]
    fn kept_out<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.kept_out)
    }
    /// Which kinds the library holds for this light's sensor at all.
    #[getter]
    fn library<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.library)
    }
    /// The rules the light was matched with.
    #[getter]
    fn rules(&self) -> PyCalibrationRigRules {
        PyCalibrationRigRules {
            rules: Arc::clone(&self.selection.rules),
        }
    }
    #[getter]
    fn flat_basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.flat_basis)
    }
    #[getter]
    fn flat_held_back(&self) -> Vec<String> {
        self.selection.flat_held_back.clone()
    }
    #[getter]
    fn flat_family<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.flat_family)
    }
    #[getter]
    fn flat_choices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.flat_choices)
    }
    /// Kinds taken as one supplied set.
    #[getter]
    fn one_set(&self) -> Vec<&'static str> {
        self.selection
            .one_set
            .iter()
            .map(|kind| kind.as_str())
            .collect()
    }

    /// The tolerances a stack holds this light to against its masters.
    fn match_tolerances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.match_tolerances())
    }

    fn is_empty(&self) -> bool {
        self.selection.is_empty()
    }

    /// The candidates of one kind.
    fn of_kind<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.selection.of_kind(word(kind)?))
    }

    /// Every selected frame, bias, dark, dark-flat then flat.
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let frames: Vec<&library::CalibrationFrame> = self.selection.frames().collect();
        to_python(py, &frames)
    }
}

// --- sessions --------------------------------------------------------------------

/// The masters one light's selection builds, and how its flat applies.
#[pyclass(skip_from_py_object, name = "CalibrationSession", module = "seiza")]
pub(crate) struct PyCalibrationSession {
    session: library::Session,
}

#[pymethods]
impl PyCalibrationSession {
    #[getter]
    fn key(&self) -> String {
        self.session.key.clone()
    }
    #[getter]
    fn tolerances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session.tolerances)
    }
    /// Bias, dark-flat, dark and flat, in build order.
    #[getter]
    fn recipes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session.recipes)
    }
    #[getter]
    fn application<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session.application)
    }
    #[getter]
    fn flat_stability<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session.flat_stability)
    }
    #[getter]
    fn flat_held_back(&self) -> Vec<String> {
        self.session.flat_held_back.clone()
    }

    /// The recipe for one kind, or None.
    fn recipe<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session.recipe(word(kind)?))
    }

    /// Record that `kind`'s master failed to build; every master that
    /// depends on it is blocked.
    fn record_failure(&mut self, kind: &str, reason: String) -> PyResult<()> {
        self.session
            .record_failure(word(kind)?, reason)
            .map_err(invalid)
    }

    /// How the flat applies to `lights`, the session's own, once every
    /// master has built or failed. `without_masters` is "fit_pedestal" or
    /// "force".
    #[pyo3(signature = (lights, *, without_masters="fit_pedestal"))]
    fn plan_flat_use(&mut self, lights: &Bound<'_, PyAny>, without_masters: &str) -> PyResult<()> {
        let lights: Vec<library::Light> = from_python(lights, "lights")?;
        self.session.plan_flat_use(&lights, word(without_masters)?);
        Ok(())
    }

    /// Accept or refuse the pedestal fitted from the sample lights:
    /// `estimates`, at most three; a `pinned` value wins.
    #[pyo3(signature = (estimates, pinned=None))]
    fn accept_pedestal(&mut self, estimates: Vec<f32>, pinned: Option<f32>) -> PyResult<()> {
        self.session
            .accept_pedestal(&estimates, pinned)
            .map_err(invalid)
    }

    /// The whole session as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.session)
    }
}

/// The session a light's verified `selection` builds: each kind's recipe,
/// the complete gate, and an identity that covers what the masters are made
/// from and how.
#[pyfunction]
#[pyo3(signature = (selection, quality, *, flat_star_masking=false, flat_stability=None))]
fn calibration_resolve_session(
    selection: PyRef<'_, PyCalibrationSelection>,
    quality: PyRef<'_, PyCalibrationQuality>,
    flat_star_masking: bool,
    flat_stability: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyCalibrationSession> {
    let options = library::SessionOptions {
        flat_star_masking,
        flat_stability: optional(flat_stability, "flat stability policy")?,
    };
    Ok(PyCalibrationSession {
        session: library::resolve_session(&selection.selection, &quality.quality, &options),
    })
}

/// The frames one master of `kind` integrates from `frames`, given nearest
/// first, and those that clustered with them but would not integrate:
/// `{"kept": [...], "dropped": [...]}`.
#[pyfunction]
fn calibration_coherent_cohort<'py>(
    py: Python<'py>,
    kind: &str,
    frames: &Bound<'py, PyAny>,
    rules: PyRef<'_, PyCalibrationRigRules>,
) -> PyResult<Bound<'py, PyAny>> {
    let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
    let cohort = library::coherent_cohort(word(kind)?, &frames, &rules.rules);
    let result = PyDict::new(py);
    result.set_item("kept", to_python(py, &cohort.kept)?)?;
    result.set_item("dropped", to_python(py, &cohort.dropped)?)?;
    Ok(result.into_any())
}

/// The external master a light takes for `kind` from `frames`, its
/// candidates nearest first, used as it is and never integrated with raw
/// frames; None when the rules prefer the raw frames or ignore masters.
/// `held_back` flats, from the far side of a measured change, never stand in.
#[pyfunction]
#[pyo3(signature = (kind, frames, rules, *, held_back=Vec::new()))]
fn calibration_adopted_master<'py>(
    py: Python<'py>,
    kind: &str,
    frames: &Bound<'py, PyAny>,
    rules: PyRef<'_, PyCalibrationRigRules>,
    held_back: Vec<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
    let adopted = library::adopted_master(word(kind)?, &frames, &held_back, &rules.rules);
    to_python(py, &adopted)
}

/// Whether the integrator should screen a dark or dark-flat master's
/// `inputs` for stray light itself: when one of them has no current grade
/// or personal choice in `quality`.
#[pyfunction]
fn calibration_dark_level_screening(
    kind: &str,
    inputs: &Bound<'_, PyAny>,
    quality: PyRef<'_, PyCalibrationQuality>,
) -> PyResult<bool> {
    let inputs: Vec<library::CalibrationFrame> = from_python(inputs, "inputs")?;
    Ok(library::dark_level_screening(
        word(kind)?,
        &inputs,
        &quality.quality,
    ))
}

/// The pedestal under a session's lights, in ADU, from the fits of its
/// sample lights: their median, when they agree, lie in range and do not
/// contradict the camera's recorded offset `hint`. Raises ValueError saying
/// why they were refused.
#[pyfunction]
#[pyo3(signature = (estimates, hint=None))]
fn calibration_pedestal_from_fits(estimates: Vec<f32>, hint: Option<f64>) -> PyResult<f32> {
    library::pedestal_from_fits(&estimates, hint).map_err(invalid)
}

/// The pedestal a camera's recorded offset predicts, in ADU, where its
/// driver is known to record it; None otherwise.
#[pyfunction]
fn calibration_pedestal_hint(light: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let light: library::Light = from_python(light, "light")?;
    Ok(library::pedestal_hint(&light))
}

/// The lights a flat pedestal is fitted from: by sorted path, the first,
/// middle and last.
#[pyfunction]
fn calibration_pedestal_sample_lights(
    py: Python<'_>,
    paths: Vec<PathBuf>,
) -> PyResult<Bound<'_, PyAny>> {
    to_python(py, &library::pedestal_sample_lights(&paths))
}

/// "20 G flats from 2026-09-10", for a note about one flat set.
#[pyfunction]
fn calibration_flat_set_label(frames: &Bound<'_, PyAny>) -> PyResult<String> {
    let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
    Ok(library::flat_set_label(&frames))
}

/// A grade's finding in words, for a frame of `kind`.
#[pyfunction]
fn calibration_describe_finding(finding: &Bound<'_, PyAny>, kind: &str) -> PyResult<String> {
    let finding: seiza_calibration::Finding = from_python(finding, "finding")?;
    Ok(library::describe_finding(&finding, word(kind)?))
}

/// The family a flat belongs to, rotation aside, with its focal length
/// class: the key flat responses are kept by.
#[pyfunction]
#[pyo3(signature = (frame, focal=None))]
fn calibration_family_key(frame: &Bound<'_, PyAny>, focal: Option<f64>) -> PyResult<String> {
    let frame: library::CalibrationFrame = from_python(frame, "frame")?;
    Ok(library::family_key(&frame, focal))
}

/// Whether a kept response's pedestal moved enough to measure it again.
#[pyfunction]
#[pyo3(signature = (kept, now))]
fn calibration_pedestal_moved(kept: Option<f64>, now: Option<f64>) -> bool {
    library::pedestal_moved(kept, now)
}

// --- flat retry ------------------------------------------------------------------

/// A flat master's way to a set that agrees with itself, one step at a
/// time: leave a drifting minority out, else take the next session, else
/// keep the first set and say so.
#[pyclass(skip_from_py_object, name = "FlatRetry", module = "seiza")]
pub(crate) struct PyFlatRetry {
    first: library::FlatAttempt,
    candidates: Vec<library::CalibrationFrame>,
    /// The session the master is for, whose policy and held-back flats
    /// stand in for `held_back` and `policy`.
    session: Option<library::Session>,
    held_back: Vec<String>,
    rules: Arc<rules::Rules>,
    policy: library::FlatRetryPolicy,
    trimmed: Option<Option<library::FlatAttempt>>,
    alternate: Option<Option<library::FlatAttempt>>,
}

impl PyFlatRetry {
    /// The native retry with what it has been told: built each time, since
    /// it borrows the candidates this object owns.
    fn retry(&self) -> PyResult<library::FlatRetry<'_>> {
        let mut retry = match &self.session {
            Some(session) => library::FlatRetry::for_session(
                self.first.clone(),
                &self.candidates,
                session,
                &self.rules,
            ),
            None => library::FlatRetry::new(
                self.first.clone(),
                &self.candidates,
                &self.held_back,
                &self.rules,
                self.policy.clone(),
            ),
        }
        .map_err(invalid)?;
        if let Some(attempt) = &self.trimmed {
            retry.trimmed(attempt.clone()).map_err(invalid)?;
        }
        if let Some(attempt) = &self.alternate {
            retry.alternate(attempt.clone()).map_err(invalid)?;
        }
        Ok(retry)
    }
}

#[pymethods]
impl PyFlatRetry {
    /// The retry of a flat master first built as `first` from `candidates`,
    /// the light's flats nearest first, by `rules` (the selection's).
    /// `held_back` frames never stand in; `policy` defaults to all retries.
    /// Given the `session` the master is for, its policy and held-back flats
    /// are used instead, so a supplied set never takes another session.
    #[new]
    #[pyo3(signature = (first, candidates, rules, *, held_back=Vec::new(), policy=None, session=None))]
    fn new(
        first: &Bound<'_, PyAny>,
        candidates: &Bound<'_, PyAny>,
        rules: PyRef<'_, PyCalibrationRigRules>,
        held_back: Vec<String>,
        policy: Option<&Bound<'_, PyAny>>,
        session: Option<PyRef<'_, PyCalibrationSession>>,
    ) -> PyResult<Self> {
        if session.is_some() && (!held_back.is_empty() || policy.is_some_and(|p| !p.is_none())) {
            return Err(invalid(
                "give the session, or held_back and policy, not both",
            ));
        }
        let retry = Self {
            first: from_python(first, "first attempt")?,
            candidates: from_python(candidates, "candidates")?,
            session: session.map(|session| session.session.clone()),
            held_back,
            rules: Arc::clone(&rules.rules),
            policy: optional(policy, "flat retry policy")?,
            trimmed: None,
            alternate: None,
        };
        retry.retry()?;
        Ok(retry)
    }

    /// Record the trimmed build, or None when it made no master.
    #[pyo3(signature = (attempt=None))]
    fn trimmed(&mut self, attempt: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let attempt = attempt
            .filter(|value| !value.is_none())
            .map(|value| from_python(value, "trimmed attempt"))
            .transpose()?;
        let mut retry = self.retry()?;
        retry.trimmed(attempt.clone()).map_err(invalid)?;
        self.trimmed = Some(attempt);
        Ok(())
    }

    /// Record the next session's build, or None when it made no master.
    #[pyo3(signature = (attempt=None))]
    fn alternate(&mut self, attempt: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let attempt = attempt
            .filter(|value| !value.is_none())
            .map(|value| from_python(value, "alternate attempt"))
            .transpose()?;
        let mut retry = self.retry()?;
        retry.alternate(attempt.clone()).map_err(invalid)?;
        self.alternate = Some(attempt);
        Ok(())
    }

    /// The next step: `{"action", "inputs", "rebuild_first", "note"}`.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.retry()?.next())
    }
}

// --- flat sessions and pedestals ---------------------------------------------------

/// One flat session about to be measured: one night of one family.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "FlatSessionPlan",
    module = "seiza"
)]
pub(crate) struct PyFlatSessionPlan {
    plan: library::FlatSessionPlan,
}

/// What `flat_build_response` returns, as `FlatSessionPlan.measured_row`
/// reads it back.
struct ResponseArgument {
    plane: Plane,
    bin: usize,
    level: f32,
    noise: f32,
    frames: usize,
}

impl ResponseArgument {
    fn of(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let item = |name: &str| {
            value
                .get_item(name)
                .map_err(|_| invalid(format!("the response has no {name}")))
        };
        let plane: PyReadonlyArrayDyn<'_, f32> = item("plane")?.extract()?;
        Ok(Self {
            plane: as_plane(&plane, "response plane")?,
            bin: item("bin")?.extract()?,
            level: item("level")?.extract()?,
            noise: item("noise")?.extract()?,
            frames: item("frames")?.extract()?,
        })
    }
}

#[pymethods]
impl PyFlatSessionPlan {
    #[getter]
    fn session_key(&self) -> String {
        self.plan.session_key.clone()
    }
    #[getter]
    fn family_key(&self) -> String {
        self.plan.family_key.clone()
    }
    #[getter]
    fn rig_uuid(&self) -> String {
        self.plan.rig_uuid.clone()
    }
    #[getter]
    fn night(&self) -> String {
        self.plan.night.clone()
    }
    #[getter]
    fn filter(&self) -> Option<String> {
        self.plan.filter.clone()
    }
    #[getter]
    fn rotation(&self) -> Option<f64> {
        self.plan.rotation
    }
    /// In capture order.
    #[getter]
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.plan.frames)
    }

    /// The frames the response combines, evenly spaced.
    fn combined<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.plan.combined())
    }

    /// The response's identity.
    fn response_key(&self) -> String {
        self.plan.response_key()
    }

    /// Whether the session holds enough flats to make a response.
    fn measurable(&self) -> bool {
        self.plan.measurable()
    }

    /// The row a planner shows before any flat is read.
    #[pyo3(signature = (pedestal=None))]
    fn pending_row<'py>(
        &self,
        py: Python<'py>,
        pedestal: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.plan.pending_row(pedestal))
    }

    /// The row a host keeps once it has read the combined flats: `read`
    /// planes binned by `bin`, the `response` `flat_build_response` built
    /// from them (None when they would not combine), and the flats that
    /// would not read.
    #[pyo3(signature = (pedestal, read, bin, response=None, unread=None))]
    fn measured_row<'py>(
        &self,
        py: Python<'py>,
        pedestal: Option<f64>,
        read: usize,
        bin: usize,
        response: Option<&Bound<'py, PyAny>>,
        unread: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let response = response
            .filter(|value| !value.is_none())
            .map(ResponseArgument::of)
            .transpose()?
            .map(|r| metric::FlatResponse {
                plane: r.plane,
                bin: r.bin,
                level: r.level,
                noise: r.noise,
                frames: r.frames,
            });
        let read = library::ResponseRead {
            read,
            bin,
            response: response.as_ref(),
            unread: optional(unread, "unread flats")?,
        };
        to_python(py, &self.plan.measured_row(pedestal, read))
    }
}

/// The flat sessions in use: raw flats that serve and record a capture
/// time, one night of one family each.
#[pyfunction]
fn calibration_flat_sessions(
    frames: &Bound<'_, PyAny>,
    nights: PyRef<'_, PyCalibrationNights>,
    rules: PyRef<'_, PyCalibrationRules>,
) -> PyResult<Vec<PyFlatSessionPlan>> {
    let frames: Vec<library::CalibrationFrame> = from_python(frames, "frames")?;
    Ok(
        library::plan_flat_sessions(&frames, &nights.nights, &rules.book)
            .into_iter()
            .map(|plan| PyFlatSessionPlan { plan })
            .collect(),
    )
}

/// One measured bias or dark-flat level.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PedestalLevelRecord {
    rig_uuid: String,
    #[serde(default)]
    gain: Option<i64>,
    #[serde(default)]
    offset: Option<i64>,
    kind: library::CalibrationKind,
    #[serde(default)]
    exposure_s: Option<f64>,
    level: f64,
}

/// Bias and dark-flat levels by rig, gain and offset: what lies under a
/// session's flats, and under lights.
#[pyclass(skip_from_py_object, name = "PedestalTable", module = "seiza")]
pub(crate) struct PyPedestalTable {
    table: library::PedestalTable,
}

#[pymethods]
impl PyPedestalTable {
    /// The measured level of every raw bias and dark-flat of `frames`.
    #[new]
    #[pyo3(signature = (frames=None, quality=None))]
    fn new(
        frames: Option<&Bound<'_, PyAny>>,
        quality: Option<PyRef<'_, PyCalibrationQuality>>,
    ) -> PyResult<Self> {
        let frames: Vec<library::CalibrationFrame> = optional(frames, "frames")?;
        let empty = library::QualityIndex::default();
        let quality = quality.as_ref().map_or(&empty, |quality| &quality.quality);
        Ok(Self {
            table: library::PedestalTable::new(&frames, quality),
        })
    }

    /// Add one measured level: `{"rig_uuid", "gain", "offset", "kind",
    /// "exposure_s", "level"}`.
    fn insert(&mut self, level: &Bound<'_, PyAny>) -> PyResult<()> {
        let level: PedestalLevelRecord = from_python(level, "pedestal level")?;
        self.table.insert(library::PedestalLevel {
            rig_uuid: &level.rig_uuid,
            gain: level.gain,
            offset: level.offset,
            kind: level.kind,
            exposure_s: level.exposure_s,
            level: level.level,
        });
        Ok(())
    }

    /// The pedestal under lights of a rig at this gain and offset.
    #[pyo3(signature = (rig_uuid, gain=None, offset=None))]
    fn light_pedestal(
        &self,
        rig_uuid: &str,
        gain: Option<i64>,
        offset: Option<i64>,
    ) -> Option<f64> {
        self.table.light_pedestal(rig_uuid, gain, offset)
    }

    /// The pedestal under flats of a rig at this gain, offset and exposure.
    #[pyo3(signature = (rig_uuid, gain=None, offset=None, exposure_s=None))]
    fn flat_pedestal(
        &self,
        rig_uuid: &str,
        gain: Option<i64>,
        offset: Option<i64>,
        exposure_s: Option<f64>,
    ) -> Option<f64> {
        self.table.flat_pedestal(rig_uuid, gain, offset, exposure_s)
    }

    /// The pedestal under one session's flats.
    fn for_session(&self, plan: PyRef<'_, PyFlatSessionPlan>) -> Option<f64> {
        self.table.for_session(&plan.plan)
    }
}

// --- light evidence ----------------------------------------------------------------

/// One light of an imaging session, as the evidence stages ask it.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceLightRecord {
    path: PathBuf,
    at: i64,
    #[serde(default)]
    angle: Option<f64>,
    #[serde(default)]
    filter: String,
    #[serde(default)]
    target: String,
}

impl From<EvidenceLightRecord> for library::EvidenceLight {
    fn from(light: EvidenceLightRecord) -> Self {
        Self {
            path: light.path,
            at: light.at,
            angle: light.angle,
            filter: light.filter,
            target: light.target,
        }
    }
}

impl From<&library::EvidenceLight> for EvidenceLightRecord {
    fn from(light: &library::EvidenceLight) -> Self {
        Self {
            path: light.path.clone(),
            at: light.at,
            angle: light.angle,
            filter: light.filter.clone(),
            target: light.target.clone(),
        }
    }
}

fn evidence_lights(value: &Bound<'_, PyAny>) -> PyResult<Vec<library::EvidenceLight>> {
    let lights: Vec<EvidenceLightRecord> =
        from_python(&crate::convert::with_fspaths(value)?, "evidence lights")?;
    Ok(lights.into_iter().map(Into::into).collect())
}

fn lights_python<'py, 'a>(
    py: Python<'py>,
    lights: impl IntoIterator<Item = &'a library::EvidenceLight>,
) -> PyResult<Bound<'py, PyAny>> {
    let records: Vec<EvidenceLightRecord> = lights.into_iter().map(Into::into).collect();
    to_python(py, &records)
}

fn planes(arrays: Vec<PyReadonlyArrayDyn<'_, f32>>, name: &str) -> PyResult<Vec<Plane>> {
    arrays.iter().map(|array| as_plane(array, name)).collect()
}

/// The flat session standing for each candidate of a family question.
#[pyclass(
    skip_from_py_object,
    frozen,
    name = "FamilyRepresentatives",
    module = "seiza"
)]
pub(crate) struct PyFamilyRepresentatives {
    representatives: library::Representatives,
}

#[pymethods]
impl PyFamilyRepresentatives {
    /// The flat sessions, by index in the book, in the candidates' order.
    #[getter]
    fn sessions(&self) -> Vec<usize> {
        self.representatives.sessions().to_vec()
    }
}

/// One group of an imaging session's lights the family stage asks: one
/// filter, target and angle, with two or more candidate families. It keeps
/// the book it was asked of.
#[pyclass(skip_from_py_object, frozen, name = "FamilyQuestion", module = "seiza")]
pub(crate) struct PyFamilyQuestion {
    question: library::FamilyQuestion,
    book: Arc<groups::Book>,
}

#[pymethods]
impl PyFamilyQuestion {
    /// Its imaging session, by index in the book.
    #[getter]
    fn session(&self) -> usize {
        self.question.session
    }
    #[getter]
    fn rig(&self) -> String {
        self.question.rig.clone()
    }
    #[getter]
    fn filter(&self) -> String {
        self.question.filter.clone()
    }
    #[getter]
    fn target(&self) -> String {
        self.question.target.clone()
    }
    /// In angle order; empty when answered.
    #[getter]
    fn lights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lights_python(py, &self.question.lights)
    }
    /// Candidate families, by index in the book; empty when answered.
    #[getter]
    fn candidates(&self) -> Vec<usize> {
        self.question.candidates.clone()
    }
    #[getter]
    fn angle(&self) -> f64 {
        self.question.angle
    }
    #[getter]
    fn low(&self) -> f64 {
        self.question.low
    }
    #[getter]
    fn high(&self) -> f64 {
        self.question.high
    }
    #[getter]
    fn start(&self) -> i64 {
        self.question.start
    }
    #[getter]
    fn end(&self) -> i64 {
        self.question.end
    }
    #[getter]
    fn key(&self) -> String {
        self.question.key.clone()
    }
    /// Its answer is kept under its current key: nothing to read.
    #[getter]
    fn answered(&self) -> bool {
        self.question.answered
    }

    /// Where its row is kept: `{"session_key", "filter", "target", "angle",
    /// "key"}`.
    fn retention_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let key = self.question.retention_key(&self.book);
        let result = PyDict::new(py);
        result.set_item("session_key", key.session_key)?;
        result.set_item("filter", key.filter)?;
        result.set_item("target", key.target)?;
        result.set_item("angle", key.angle)?;
        result.set_item("key", key.key)?;
        Ok(result)
    }

    /// The flat session standing for each candidate, or None when one has
    /// none and the question cannot be weighed.
    fn representatives(&self) -> Option<PyFamilyRepresentatives> {
        self.question
            .representatives(&self.book)
            .map(|representatives| PyFamilyRepresentatives { representatives })
    }

    /// The lights to read, evenly spaced.
    fn sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lights_python(py, self.question.sample())
    }

    /// Answer from the sampled lights' binned planes, their pedestal, and
    /// each representative's response plane, pedestal out (None when not on
    /// disk). Returns the family evidence row to keep.
    #[pyo3(signature = (representatives, binned, planes, pedestal=None))]
    fn decide<'py>(
        &self,
        py: Python<'py>,
        representatives: PyRef<'_, PyFamilyRepresentatives>,
        binned: Vec<PyReadonlyArrayDyn<'py, f32>>,
        planes: Vec<Option<PyReadonlyArrayDyn<'py, f32>>>,
        pedestal: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let binned = self::planes(binned, "binned lights")?;
        let planes = planes
            .iter()
            .map(|plane| {
                plane
                    .as_ref()
                    .map(|p| as_plane(p, "response plane"))
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let representatives = &representatives.representatives;
        let row = py.detach(|| {
            self.question
                .decide(&self.book, representatives, binned, planes, pedestal)
        });
        to_python(py, &row)
    }

    /// The row a choice among the candidates makes (`flat_choose_response`'s
    /// answer), from `lights` lights.
    fn row<'py>(
        &self,
        py: Python<'py>,
        choice: &Bound<'py, PyAny>,
        lights: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let choice: seiza_calibration::flat_response::choice::ResponseChoice =
            from_python(choice, "response choice")?;
        let row = self
            .question
            .row(&self.book, &choice, lights)
            .map_err(invalid)?;
        to_python(py, &row)
    }
}

/// The family questions of one rig's imaging sessions. `lights_of(session)`
/// gives a session's lights as `{"path", "at", "angle", "filter",
/// "target"}`; `header(path)` reads one light's header as a light record,
/// or None when it cannot.
#[pyfunction]
fn calibration_family_questions(
    py: Python<'_>,
    library: PyRef<'_, PyCalibrationLibrary>,
    rules: PyRef<'_, PyCalibrationRules>,
    rig: &str,
    lights_of: Bound<'_, PyAny>,
    header: Bound<'_, PyAny>,
) -> PyResult<Vec<PyFamilyQuestion>> {
    // The callbacks cannot fail the native walk; the first error is raised
    // once it returns.
    let failure: RefCell<Option<PyErr>> = RefCell::new(None);
    let fail = |error: PyErr| {
        failure.borrow_mut().get_or_insert(error);
    };
    let questions = library::family_questions(
        &library.library,
        &rules.book,
        rig,
        |session| {
            let result = to_python(py, session)
                .and_then(|session| lights_of.call1((session,)))
                .and_then(|lights| evidence_lights(&lights));
            result.unwrap_or_else(|error| {
                fail(error);
                Vec::new()
            })
        },
        |path: &Path| {
            // As the host gave it in the light records: a string.
            let result = header.call1((path.to_string_lossy(),)).and_then(|light| {
                if light.is_none() {
                    Ok(None)
                } else {
                    from_python::<library::Light>(&crate::convert::with_fspaths(&light)?, "header")
                        .map(Some)
                }
            });
            result.unwrap_or_else(|error| {
                fail(error);
                None
            })
        },
    );
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }
    let book = Arc::clone(library.library.flat_book());
    Ok(questions
        .into_iter()
        .map(|question| PyFamilyQuestion {
            question,
            book: Arc::clone(&book),
        })
        .collect())
}

/// One filter's lights of a change question, with the flat session of a
/// known side their background is fitted against.
#[pyclass(skip_from_py_object, frozen, name = "ChangeFilter", module = "seiza")]
pub(crate) struct PyChangeFilter {
    filter: library::ChangeFilter,
    book: Arc<groups::Book>,
}

#[pymethods]
impl PyChangeFilter {
    /// As captured.
    #[getter]
    fn filter(&self) -> String {
        self.filter.filter.clone()
    }
    /// The lights to read.
    #[getter]
    fn lights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lights_python(py, &self.filter.lights)
    }
    /// The reference flat session, by index in the book.
    #[getter]
    fn reference(&self) -> usize {
        self.filter.reference
    }
    /// The side of the change the reference stands on.
    #[getter]
    fn reference_side<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_python(py, &self.filter.reference_side)
    }
    /// The boundary whose change map the fit reads, by index in the book.
    #[getter]
    fn map(&self) -> usize {
        self.filter.map
    }

    /// The responses the change map is measured between: before, after.
    fn map_responses<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (before, after) = self.filter.map_responses(&self.book);
        Ok((to_python(py, before)?, to_python(py, after)?))
    }

    /// The change map from the two response planes, pedestal out.
    fn change_map<'py>(
        &self,
        py: Python<'py>,
        before: PyReadonlyArrayDyn<'py, f32>,
        after: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Option<Bound<'py, PyArrayDyn<f32>>>> {
        let (before, after) = (as_plane(&before, "before")?, as_plane(&after, "after")?);
        py.detach(|| self.filter.change_map(&self.book, &before, &after))
            .map(|plane| plane_array(py, plane))
            .transpose()
    }

    /// Fit the lights' binned planes, with their pedestal, against the
    /// reference session's plane and the change map.
    #[pyo3(signature = (binned, reference, map, pedestal=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        binned: Vec<PyReadonlyArrayDyn<'py, f32>>,
        reference: PyReadonlyArrayDyn<'py, f32>,
        map: PyReadonlyArrayDyn<'py, f32>,
        pedestal: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let binned = planes(binned, "binned lights")?;
        let (reference, map) = (as_plane(&reference, "reference")?, as_plane(&map, "map")?);
        let fit = py.detach(|| {
            self.filter
                .fit(&self.book, binned, &reference, &map, pedestal)
        });
        to_python(py, &fit)
    }
}

/// The lights of one imaging session shot during a change, by filter.
#[pyclass(skip_from_py_object, frozen, name = "ChangeQuestion", module = "seiza")]
pub(crate) struct PyChangeQuestion {
    question: library::ChangeQuestion,
    book: Arc<groups::Book>,
}

#[pymethods]
impl PyChangeQuestion {
    /// The change, by index in the book.
    #[getter]
    fn event(&self) -> usize {
        self.question.event
    }
    /// The imaging session, by index in the book.
    #[getter]
    fn session(&self) -> usize {
        self.question.session
    }
    #[getter]
    fn filters(&self) -> Vec<PyChangeFilter> {
        self.question
            .filters
            .iter()
            .map(|filter| PyChangeFilter {
                filter: filter.clone(),
                book: Arc::clone(&self.book),
            })
            .collect()
    }

    /// The evidence row its filters' fits make.
    fn row<'py>(&self, py: Python<'py>, fits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let fits: Vec<groups::EvidenceFit> = from_python(fits, "evidence fits")?;
        let row = self.question.row(&self.book, fits).map_err(invalid)?;
        to_python(py, &row)
    }
}

/// What an imaging session's `lights` can say about change `event` shot
/// during it, or None when the session has no lights to read.
#[pyfunction]
fn calibration_change_question(
    groups: PyRef<'_, PyFlatGroups>,
    event: usize,
    session: usize,
    lights: &Bound<'_, PyAny>,
) -> PyResult<Option<PyChangeQuestion>> {
    if event >= groups.book.events.len() || session >= groups.book.imaging.len() {
        return Err(invalid("no such change or imaging session in the book"));
    }
    let lights = evidence_lights(lights)?;
    Ok(
        library::change_question(&groups.book, event, session, lights).map(|question| {
            PyChangeQuestion {
                question,
                book: Arc::clone(&groups.book),
            }
        }),
    )
}

// --- flat stability ------------------------------------------------------------------

/// Measure sources from files. For cached pixels use master_flat_stability_images.
#[pyfunction]
#[pyo3(signature = (*, master, inputs, bias=None, dark=None, dark_exposure_seconds=None, normalized_full_scale=None))]
fn master_flat_stability<'py>(
    py: Python<'py>,
    master: PathBuf,
    inputs: &Bound<'py, PyAny>,
    bias: Option<PathBuf>,
    dark: Option<PathBuf>,
    dark_exposure_seconds: Option<f64>,
    normalized_full_scale: Option<f32>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = seiza_stacking::FlatStabilityFileRequest {
        master,
        inputs: from_python(&crate::convert::with_fspaths(inputs)?, "flat inputs")?,
        bias,
        dark,
        dark_exposure_seconds,
        normalized_full_scale,
    };
    let result = py
        .detach(|| seiza_stacking::measure_flat_stability_files(request))
        .map_err(invalid)?;
    to_python(py, &result)
}

/// Measure cached NumPy images without file I/O. All pixels use the same units.
#[pyfunction]
#[pyo3(signature = (master, inputs, *, bias=None, dark=None, input_metadata=None, bayer=None, dark_exposure_seconds=None, dark_bias_subtracted=false))]
#[allow(clippy::too_many_arguments)]
fn master_flat_stability_images<'py>(
    py: Python<'py>,
    master: PyReadonlyArrayDyn<'py, f32>,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
    bias: Option<PyReadonlyArrayDyn<'py, f32>>,
    dark: Option<PyReadonlyArrayDyn<'py, f32>>,
    input_metadata: Option<&Bound<'py, PyAny>>,
    bayer: Option<&Bound<'py, PyAny>>,
    dark_exposure_seconds: Option<f64>,
    dark_bias_subtracted: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let master = crate::arrays::linear_image(master)?;
    let inputs = inputs
        .into_iter()
        .map(crate::arrays::linear_image)
        .collect::<PyResult<Vec<_>>>()?;
    let bias = bias.map(crate::arrays::linear_image).transpose()?;
    let dark = dark.map(crate::arrays::linear_image).transpose()?;
    let options = seiza_stacking::FlatStabilityImageOptions {
        inputs: optional(input_metadata, "input metadata")?,
        bayer: match bayer {
            Some(value) if !value.is_none() => Some(from_python(value, "Bayer layout")?),
            _ => None,
        },
        dark_exposure_seconds,
        dark_bias_subtracted,
    };
    let result = py
        .detach(|| {
            seiza_stacking::measure_flat_stability_images(
                &master,
                &inputs,
                bias.as_ref(),
                dark.as_ref(),
                options,
            )
        })
        .map_err(invalid)?;
    to_python(py, &result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let code = std::ffi::CString::new(include_str!("library_types.py"))
        .expect("type definitions contain no NUL");
    module.py().run(&code, Some(&module.dict()), None)?;
    for name in module
        .getattr("_calibration_exports")?
        .extract::<Vec<String>>()?
    {
        module.add(name.as_str(), module.getattr(name.as_str())?)?;
    }
    module.add_class::<PyCalibrationRules>()?;
    module.add_class::<PyCalibrationRigRules>()?;
    module.add_class::<PyCalibrationNights>()?;
    module.add_class::<PyCalibrationGrades>()?;
    module.add_class::<PyCalibrationQuality>()?;
    module.add_class::<PyFlatGroups>()?;
    module.add_class::<PyCalibrationLibrary>()?;
    module.add_class::<PyCalibrationSelection>()?;
    module.add_class::<PyCalibrationSession>()?;
    module.add_class::<PyFlatRetry>()?;
    module.add_class::<PyFlatSessionPlan>()?;
    module.add_class::<PyPedestalTable>()?;
    module.add_class::<PyFamilyRepresentatives>()?;
    module.add_class::<PyFamilyQuestion>()?;
    module.add_class::<PyChangeFilter>()?;
    module.add_class::<PyChangeQuestion>()?;
    module.add_function(wrap_pyfunction!(calibration_settings_schema, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_setting_check, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_resolve_session, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_coherent_cohort, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_adopted_master, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_dark_level_screening, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_pedestal_from_fits, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_pedestal_hint, module)?)?;
    module.add_function(wrap_pyfunction!(
        calibration_pedestal_sample_lights,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_set_label, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_describe_finding, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_family_key, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_pedestal_moved, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_flat_sessions, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_family_questions, module)?)?;
    module.add_function(wrap_pyfunction!(calibration_change_question, module)?)?;
    module.add_function(wrap_pyfunction!(master_flat_stability, module)?)?;
    module.add_function(wrap_pyfunction!(master_flat_stability_images, module)?)?;
    Ok(())
}
