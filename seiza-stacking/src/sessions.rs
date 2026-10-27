//! File calibration assignments shared by planning, integration and resume.
use crate::replay::SourceStamp;
use crate::{CalibrationMasters, Error, FitsFrame, Result, path_identity};
use seiza_calibration::MatchTolerances;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// One file-backed calibration recipe and its complete light-matching gate.
/// The rotation basis is part of `tolerances`; no previous session's settings
/// are inherited. Paths name already-built masters, not raw master inputs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FileCalibrationSession {
    pub bias: Option<PathBuf>,
    pub dark: Option<PathBuf>,
    pub flat: Option<PathBuf>,
    pub dark_exposure_seconds: Option<f64>,
    /// PSF Guard's fitted-pedestal fallback: a calibrated/normalized flat
    /// plus a synthetic constant bias applied to the lights. Requires no
    /// bias or dark file. The host retains the fit's evidence in its recipe.
    pub pedestal: Option<f32>,
    /// Convert only a declared 0:1 source range to this full scale, as PSF
    /// Guard's image loader does with 65535 ADU. None preserves stored units.
    pub normalized_full_scale: Option<f32>,
    pub tolerances: MatchTolerances,
}

impl FileCalibrationSession {
    pub(crate) fn paths(&self) -> impl Iterator<Item = &Path> {
        [&self.bias, &self.dark, &self.flat]
            .into_iter()
            .flatten()
            .map(PathBuf::as_path)
    }

    pub(crate) fn load(&self) -> Result<CalibrationMasters> {
        if let Some(pedestal) = self.pedestal {
            let flat = FitsFrame::open(
                self.flat
                    .as_ref()
                    .ok_or_else(|| error("a fitted pedestal requires a flat"))?,
            )?;
            let signature = flat.metadata().signature;
            let bias = crate::LinearImage::new(
                flat.image.width,
                flat.image.height,
                flat.image.channels,
                vec![pedestal; flat.image.sample_count()],
            )?;
            let flat = crate::MasterFlat::from_fits_frame(flat)?;
            if !flat.calibrated {
                return Err(error(
                    "a fitted pedestal requires a calibrated/normalized flat master",
                ));
            }
            // Moved from PSF Guard's MasterRecipe::PedestalFlat loader. Keep
            // the file's optical metadata as well as the synthesized pixels.
            let mut masters = CalibrationMasters::new(Some(bias), None, Some(flat))?;
            masters.flat_signature = Some(signature);
            return Ok(masters);
        }
        CalibrationMasters::from_fits_paths(
            self.bias.as_deref(),
            self.dark.as_deref(),
            self.flat.as_deref(),
            self.dark_exposure_seconds,
        )
    }

    pub(crate) fn record(&self) -> crate::replay::CalibrationRecord {
        crate::replay::CalibrationRecord::from_paths(
            self.bias.as_deref(),
            self.dark.as_deref(),
            self.flat.as_deref(),
            self.dark_exposure_seconds,
        )
    }
}

/// Immutable, fingerprinted file assignments. The same small record is saved
/// in contexts; historical master pixels are never held for every session.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileSessionPlan {
    pub(crate) sessions: Vec<FileCalibrationSession>,
    lights: BTreeMap<PathBuf, (usize, SessionStamp)>,
    masters: BTreeMap<PathBuf, SessionStamp>,
    /// Sidecar to the existing compact ledger: preserves recipe variants
    /// without changing context-format-3 postcard calibration records.
    calibrations: BTreeMap<u32, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SessionStamp {
    metadata: SourceStamp,
    identity: u64,
}

impl FileSessionPlan {
    pub(crate) fn new(
        paths: &[PathBuf],
        sessions: &[FileCalibrationSession],
        assignments: &[usize],
    ) -> Result<Self> {
        if paths.is_empty() || paths.len() != assignments.len() {
            return Err(error(
                "each light needs exactly one calibration session assignment",
            ));
        }
        if assignments.iter().any(|index| *index >= sessions.len()) {
            return Err(error("calibration session index is out of range"));
        }
        let mut masters = BTreeMap::new();
        let mut canonical = sessions.to_vec();
        for session in &mut canonical {
            session
                .tolerances
                .validate()
                .map_err(|e| error(e.to_string()))?;
            if session
                .normalized_full_scale
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            {
                return Err(error("normalized full scale must be positive and finite"));
            }
            if session.pedestal.is_some_and(|v| !v.is_finite() || v < 0.0)
                || (session.pedestal.is_some()
                    && (session.flat.is_none()
                        || session.bias.is_some()
                        || session.dark.is_some()
                        || session.dark_exposure_seconds.is_some()))
            {
                return Err(error(
                    "a finite nonnegative fitted pedestal requires a flat and no bias/dark files",
                ));
            }
            if session
                .dark_exposure_seconds
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
                || (session.dark.is_none() && session.dark_exposure_seconds.is_some())
            {
                return Err(error(
                    "a positive dark exposure override requires a dark path",
                ));
            }
            let mut seen = BTreeSet::new();
            for path in [&mut session.bias, &mut session.dark, &mut session.flat]
                .into_iter()
                .flatten()
            {
                *path = path_identity(path);
                if !seen.insert(path.clone()) {
                    return Err(error("a session repeats a master path"));
                }
                masters.insert(path.clone(), stamp(path)?);
            }
            let mut identities = BTreeMap::new();
            for path in session.paths() {
                insert_identity(&mut identities, path, masters[path].identity)?;
            }
        }
        let master_identities = masters.iter().fold(
            BTreeMap::<u64, Vec<PathBuf>>::new(),
            |mut map, (path, stamp)| {
                map.entry(stamp.identity).or_default().push(path.clone());
                map
            },
        );
        let mut lights = BTreeMap::new();
        let mut light_identities = BTreeMap::new();
        for (path, session) in paths.iter().zip(assignments) {
            if *session >= sessions.len() {
                return Err(error("calibration session index is out of range"));
            }
            let path = path_identity(path);
            if masters.contains_key(&path) {
                return Err(error("a light aliases a calibration master"));
            }
            let source = stamp(&path)?;
            if aliases(&master_identities, &path, source.identity)? {
                return Err(error("a light aliases a calibration master"));
            }
            insert_identity(&mut light_identities, &path, source.identity)?;
            if lights.insert(path, (*session, source)).is_some() {
                return Err(error("duplicate light in calibration session plan"));
            }
        }
        Ok(Self {
            sessions: canonical,
            lights,
            masters,
            calibrations: BTreeMap::new(),
        })
    }

    pub(crate) fn verify_masters(&self) -> Result<()> {
        for (path, expected) in &self.masters {
            verify(path, *expected)?;
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.verify_masters()?;
        if self.lights.is_empty() {
            return Err(error("session plan has no lights"));
        }
        if self
            .calibrations
            .values()
            .any(|index| *index >= self.sessions.len())
        {
            return Err(error("invalid calibration recipe sidecar"));
        }
        let paths = self.lights.keys().cloned().collect::<Vec<_>>();
        let assignments = self
            .lights
            .values()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();
        let rebuilt = Self::new(&paths, &self.sessions, &assignments)?;
        if rebuilt.lights != self.lights
            || rebuilt.masters != self.masters
            || rebuilt.sessions != self.sessions
        {
            return Err(error(
                "calibration session plan sources or assignments changed",
            ));
        }
        Ok(())
    }

    pub(crate) fn assignment(&self, path: &Path) -> Result<usize> {
        let path = path_identity(path);
        let (session, expected) = self.lights.get(&path).ok_or_else(|| {
            error(format!(
                "{} is not assigned by this calibration plan",
                path.display()
            ))
        })?;
        verify(&path, *expected)?;
        for master in self.sessions[*session].paths() {
            verify(master, self.masters[master])?;
        }
        Ok(*session)
    }

    pub(crate) fn protected_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.masters.keys().chain(self.lights.keys())
    }

    pub(crate) fn bind(&mut self, calibration: u32, session: usize) {
        self.calibrations.insert(calibration, session);
    }

    pub(crate) fn recipe(&self, calibration: u32) -> Result<&FileCalibrationSession> {
        self.calibrations
            .get(&calibration)
            .and_then(|index| self.sessions.get(*index))
            .ok_or_else(|| error("calibration recipe is missing from the session sidecar"))
    }

    pub(crate) fn verify_recipe(&self, calibration: u32) -> Result<()> {
        for path in self.recipe(calibration)?.paths() {
            verify(path, self.masters[path])?;
        }
        Ok(())
    }

    pub(crate) fn fingerprint_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&(&self.sessions, &self.lights, &self.masters))
            .map_err(|e| error(e.to_string()))
    }
}

/// One cached master set; workers can retain its Arc while another session
/// replaces it. Resident sets are bounded by planning concurrency, not by the
/// number of nights in the input list.
pub(crate) struct PlanningSessions<'a> {
    pub(crate) plan: &'a FileSessionPlan,
    cache: Mutex<Option<(usize, Arc<CalibrationMasters>)>>,
}

impl<'a> PlanningSessions<'a> {
    pub(crate) fn new(plan: &'a FileSessionPlan) -> Self {
        Self {
            plan,
            cache: Mutex::new(None),
        }
    }

    pub(crate) fn prepare(
        &self,
        path: &Path,
        mut frame: FitsFrame,
        options: &crate::StackOptions,
        bin: u32,
    ) -> Result<(FitsFrame, Option<crate::BayerLayout>)> {
        let index = self.plan.assignment(path)?;
        let session = &self.plan.sessions[index];
        if let Some(scale) = session.normalized_full_scale {
            frame.rescale_declared_unit_bounds(scale);
        }
        let masters = {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| error("calibration cache lock was poisoned"))?;
            if cache.as_ref().is_none_or(|(current, _)| *current != index) {
                let masters = session.load()?;
                self.plan.assignment(path)?;
                *cache = Some((index, Arc::new(masters)));
            }
            Arc::clone(&cache.as_ref().expect("loaded session").1)
        };
        let result =
            crate::stack::prepare_light_at(frame, &masters, &session.tolerances, options, bin)
                .map_err(crate::stack::LightFailure::into_error)?;
        self.plan.assignment(path)?;
        Ok(result)
    }
}

fn stamp(path: &Path) -> Result<SessionStamp> {
    let metadata = SourceStamp::of(path).ok_or_else(|| {
        error(format!(
            "cannot fingerprint calibration-plan source {}",
            path.display()
        ))
    })?;
    // Drop the handle immediately: large input lists must not consume one
    // open descriptor per light. The hash is a bucket key; alias checks below
    // compare actual file identities when a bucket matches.
    let handle = same_file::Handle::from_path(path).map_err(|e| error(e.to_string()))?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    handle.hash(&mut hasher);
    Ok(SessionStamp {
        metadata,
        identity: hasher.finish(),
    })
}

fn verify(path: &Path, expected: SessionStamp) -> Result<()> {
    if stamp(path).ok() != Some(expected) {
        return Err(error(format!(
            "calibration-plan source {} changed or is missing",
            path.display()
        )));
    }
    Ok(())
}

fn aliases(buckets: &BTreeMap<u64, Vec<PathBuf>>, path: &Path, identity: u64) -> Result<bool> {
    for previous in buckets.get(&identity).into_iter().flatten() {
        if same_file::is_same_file(previous, path).map_err(|e| error(e.to_string()))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn insert_identity(
    buckets: &mut BTreeMap<u64, Vec<PathBuf>>,
    path: &Path,
    identity: u64,
) -> Result<()> {
    if aliases(buckets, path, identity)? {
        return Err(error("duplicate physical file in calibration session plan"));
    }
    buckets
        .entry(identity)
        .or_default()
        .push(path.to_path_buf());
    Ok(())
}

fn error(message: impl Into<String>) -> Error {
    Error::Calibration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(root: &Path, name: &str) -> PathBuf {
        let path = root.join(name);
        std::fs::write(&path, b"source identity").unwrap();
        path
    }

    #[test]
    fn assignments_and_gate_errors_are_refused_before_file_reads() {
        let missing = vec![PathBuf::from("missing")];
        assert!(
            FileSessionPlan::new(&missing, &[], &[0])
                .unwrap_err()
                .to_string()
                .contains("index")
        );
        assert!(FileSessionPlan::new(&missing, &[Default::default()], &[]).is_err());
        let bad = FileCalibrationSession {
            tolerances: MatchTolerances {
                rotation_deg: f64::NAN,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            FileSessionPlan::new(&missing, &[bad], &[0])
                .unwrap_err()
                .to_string()
                .contains("rotation")
        );
    }

    #[test]
    fn physical_aliases_are_not_distinct_inputs_or_master_roles() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let alias = root.path().join("alias.data");
        std::fs::hard_link(&light, &alias).unwrap();
        assert!(
            FileSessionPlan::new(
                &[light.clone(), alias.clone()],
                &[Default::default()],
                &[0, 0]
            )
            .is_err()
        );
        let session = FileCalibrationSession {
            bias: Some(alias.clone()),
            ..Default::default()
        };
        assert!(
            FileSessionPlan::new(std::slice::from_ref(&light), &[session], &[0])
                .unwrap_err()
                .to_string()
                .contains("aliases")
        );
        let session = FileCalibrationSession {
            bias: Some(light),
            flat: Some(alias),
            ..Default::default()
        };
        let other = file(root.path(), "other.data");
        assert!(FileSessionPlan::new(&[other], &[session], &[0]).is_err());
    }

    #[test]
    fn replacing_a_file_with_preserved_length_and_time_invalidates_the_plan() {
        let root = tempfile::tempdir().unwrap();
        let path = file(root.path(), "light.data");
        let plan =
            FileSessionPlan::new(std::slice::from_ref(&path), &[Default::default()], &[0]).unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::rename(&path, root.path().join("old.data")).unwrap();
        file(root.path(), "light.data");
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(
            SourceStamp::of(&path),
            Some(plan.lights[&path_identity(&path)].1.metadata)
        );
        assert!(plan.assignment(&path).is_err());
    }

    #[test]
    fn large_assignment_lists_have_no_frame_or_file_descriptor_cap() {
        let root = tempfile::tempdir().unwrap();
        let paths = (0..600)
            .map(|i| file(root.path(), &format!("{i}.data")))
            .collect::<Vec<_>>();
        let plan =
            FileSessionPlan::new(&paths, &[Default::default()], &vec![0; paths.len()]).unwrap();
        assert_eq!(plan.lights.len(), 600);
        plan.validate().unwrap();
    }
}
