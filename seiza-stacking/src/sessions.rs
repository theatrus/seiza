//! File calibration assignments shared by planning, integration and resume.
use crate::replay::SourceStamp;
use crate::{CalibrationMasters, Error, FitsFrame, Result, path_identity};
use seiza_calibration::MatchTolerances;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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
    /// Refuse a gate or recipe no plan can apply, before any file is read.
    fn check(&self) -> Result<()> {
        self.tolerances
            .validate()
            .map_err(|e| error(e.to_string()))?;
        if self
            .normalized_full_scale
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(error("normalized full scale must be positive and finite"));
        }
        if self.pedestal.is_some_and(|v| !v.is_finite() || v < 0.0)
            || (self.pedestal.is_some()
                && (self.flat.is_none()
                    || self.bias.is_some()
                    || self.dark.is_some()
                    || self.dark_exposure_seconds.is_some()))
        {
            return Err(error(
                "a finite nonnegative fitted pedestal requires a flat and no bias/dark files",
            ));
        }
        if self
            .dark_exposure_seconds
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || (self.dark.is_none() && self.dark_exposure_seconds.is_some())
        {
            return Err(error(
                "a positive dark exposure override requires a dark path",
            ));
        }
        Ok(())
    }

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
///
/// A saved plan identifies each source by its size and modification time,
/// as an ordinary context does each frame, and checks it when it is used: a
/// light when it is pushed, a session's masters when they are loaded. So a
/// context reopens without a light it never took, or with a new master for
/// a session none of its frames used; only the frames that need them fail.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileSessionPlan {
    pub(crate) sessions: Vec<FileCalibrationSession>,
    lights: BTreeMap<PathBuf, (usize, SessionStamp)>,
    masters: BTreeMap<PathBuf, SessionStamp>,
    /// Sidecar to the existing compact ledger: preserves recipe variants
    /// without changing context-format-3 postcard calibration records.
    calibrations: BTreeMap<u32, usize>,
    /// Lights that are an earlier light's file under another name, such as
    /// a hard link. Each is refused when pushed, as a repeated path is.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    duplicates: BTreeSet<PathBuf>,
    /// The physical identities this process has seen; never saved.
    #[serde(skip)]
    live: LiveIdentities,
}

/// What a saved plan keeps of a source: the size and modification time an
/// ordinary context keeps of each frame. Plans saved by 0.46 also hold an
/// `identity`, a hash of the file's device and inode; a remounted volume
/// renumbers those and Rust does not promise to hash them alike, so serde
/// reads past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SessionStamp {
    metadata: SourceStamp,
}

/// Physical file identities this process has seen, by path: a hash of the
/// device and inode, or of the volume and file index. They hold only while
/// the volume stays mounted, so they are never saved; a reopened plan
/// learns each when it first checks the file. Clones share what they learn.
#[derive(Clone, Debug, Default)]
struct LiveIdentities(Arc<Mutex<HashMap<PathBuf, u64>>>);

impl LiveIdentities {
    fn get(&self, path: &Path) -> Option<u64> {
        self.lock().get(path).copied()
    }

    /// Record `key` as `path`'s identity, or whether it is the one recorded.
    fn confirm(&self, path: &Path, key: u64) -> bool {
        let mut known = self.lock();
        match known.get(path) {
            Some(recorded) => *recorded == key,
            None => {
                known.insert(path.to_path_buf(), key);
                true
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<PathBuf, u64>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl FileSessionPlan {
    /// Every light's session and every source's stamp. A path given twice
    /// keeps its first assignment, and a light that is an earlier light's
    /// file under another name is recorded as a duplicate: both are refused
    /// as duplicates when pushed, as the ordinary opener refuses a repeated
    /// path. A light that is a master's file is refused here.
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
        let mut keys = HashMap::new();
        let mut canonical = sessions.to_vec();
        for session in &mut canonical {
            session.check()?;
            let mut seen = BTreeSet::new();
            for path in [&mut session.bias, &mut session.dark, &mut session.flat]
                .into_iter()
                .flatten()
            {
                *path = path_identity(path);
                if !seen.insert(path.clone()) {
                    return Err(error("a session repeats a master path"));
                }
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    masters.entry(path.clone())
                {
                    let (metadata, key) = first_observation(path)?;
                    entry.insert(SessionStamp { metadata });
                    keys.insert(path.clone(), key);
                }
            }
            // Two roles of one session must not be one file by two names.
            let mut roles = Buckets::default();
            for path in session.paths() {
                if roles.aliases(path, keys[path])? {
                    return Err(error("duplicate physical file in calibration session plan"));
                }
                roles.insert(path, keys[path]);
            }
        }
        let mut master_files = Buckets::default();
        for (path, key) in &keys {
            master_files.insert(path, *key);
        }
        let mut lights = BTreeMap::new();
        let mut light_files = Buckets::default();
        let mut duplicates = BTreeSet::new();
        for (path, session) in paths.iter().zip(assignments) {
            let path = path_identity(path);
            if masters.contains_key(&path) {
                return Err(error("a light aliases a calibration master"));
            }
            if lights.contains_key(&path) {
                continue;
            }
            let (metadata, key) = first_observation(&path)?;
            if master_files.aliases(&path, key)? {
                return Err(error("a light aliases a calibration master"));
            }
            if light_files.aliases(&path, key)? {
                duplicates.insert(path.clone());
            }
            light_files.insert(&path, key);
            keys.insert(path.clone(), key);
            lights.insert(path, (*session, SessionStamp { metadata }));
        }
        Ok(Self {
            sessions: canonical,
            lights,
            masters,
            calibrations: BTreeMap::new(),
            duplicates,
            live: LiveIdentities(Arc::new(Mutex::new(keys))),
        })
    }

    /// Whether a saved plan is consistent in itself. Reads no file: each
    /// source is checked when it is used.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.lights.is_empty() {
            return Err(error("session plan has no lights"));
        }
        let count = self.sessions.len();
        if self.calibrations.values().any(|index| *index >= count) {
            return Err(error("invalid calibration recipe sidecar"));
        }
        if self.lights.values().any(|(index, _)| *index >= count) {
            return Err(error("calibration session index is out of range"));
        }
        let mut paths = BTreeSet::new();
        for session in &self.sessions {
            session.check()?;
            let mut seen = BTreeSet::new();
            for path in session.paths() {
                if !seen.insert(path) {
                    return Err(error("a session repeats a master path"));
                }
                paths.insert(path);
            }
        }
        if !paths
            .into_iter()
            .eq(self.masters.keys().map(PathBuf::as_path))
        {
            return Err(error(
                "calibration session plan masters do not match its sessions",
            ));
        }
        if self
            .lights
            .keys()
            .any(|path| self.masters.contains_key(path))
        {
            return Err(error("a light aliases a calibration master"));
        }
        if self
            .duplicates
            .iter()
            .any(|path| !self.lights.contains_key(path))
        {
            return Err(error("a duplicate light is not in its calibration plan"));
        }
        Ok(())
    }

    /// The plan's key for the light at `path`, its session and its stamp.
    /// A path spelled as the plan spells it costs no filesystem call.
    fn light(&self, path: &Path) -> Result<(&PathBuf, usize, SessionStamp)> {
        self.lights
            .get_key_value(path)
            .or_else(|| self.lights.get_key_value(&path_identity(path)))
            .map(|(light, (session, stamp))| (light, *session, *stamp))
            .ok_or_else(|| {
                error(format!(
                    "{} is not assigned by this calibration plan",
                    path.display()
                ))
            })
    }

    /// The session the light at `path` is assigned, without reading it.
    pub(crate) fn session_of(&self, path: &Path) -> Result<usize> {
        Ok(self.light(path)?.1)
    }

    /// Whether `path` is an earlier light's file under another name.
    pub(crate) fn is_duplicate(&self, path: &Path) -> bool {
        !self.duplicates.is_empty()
            && (self.duplicates.contains(path) || self.duplicates.contains(&path_identity(path)))
    }

    /// The session assigned to the light at `path`, once the light is found
    /// to be the file the plan was made from: one metadata read. Refuses a
    /// light this process sees is one of its session's masters by another
    /// name. Planning prepares a duplicate light as the ordinary planner
    /// does a repeated path; the stack refuses it when it is pushed.
    pub(crate) fn assignment(&self, path: &Path) -> Result<usize> {
        let (light, session, expected) = self.light(path)?;
        let key = self.verify(light, expected)?;
        for master in self.sessions[session].paths() {
            let master_key = match self.live.get(master) {
                Some(master_key) => master_key,
                None => self.verify(master, self.masters[master])?,
            };
            if master_key == key
                && same_file::is_same_file(master, light).map_err(|e| error(e.to_string()))?
            {
                return Err(error("a light aliases a calibration master"));
            }
        }
        Ok(session)
    }

    /// Check session `index`'s masters against the plan.
    fn verify_session(&self, index: usize) -> Result<()> {
        for path in self.sessions[index].paths() {
            self.verify(path, self.masters[path])?;
        }
        Ok(())
    }

    /// Session `index`'s masters, read between two checks of their files so
    /// that one replaced while it was read is refused.
    pub(crate) fn load_session(&self, index: usize) -> Result<CalibrationMasters> {
        self.verify_session(index)?;
        let masters = self.sessions[index].load()?;
        self.verify_session(index)?;
        Ok(masters)
    }

    pub(crate) fn protected_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.masters.keys().chain(self.lights.keys())
    }

    pub(crate) fn bind(&mut self, calibration: u32, session: usize) {
        self.calibrations.insert(calibration, session);
    }

    fn recipe_index(&self, calibration: u32) -> Result<usize> {
        self.calibrations
            .get(&calibration)
            .copied()
            .filter(|index| *index < self.sessions.len())
            .ok_or_else(|| error("calibration recipe is missing from the session sidecar"))
    }

    pub(crate) fn recipe(&self, calibration: u32) -> Result<&FileCalibrationSession> {
        Ok(&self.sessions[self.recipe_index(calibration)?])
    }

    /// Check the masters of the session behind ledger calibration set
    /// `calibration`.
    pub(crate) fn verify_recipe(&self, calibration: u32) -> Result<()> {
        self.verify_session(self.recipe_index(calibration)?)
    }

    /// [`Self::load_session`] for ledger calibration set `calibration`.
    pub(crate) fn load_recipe(&self, calibration: u32) -> Result<CalibrationMasters> {
        self.load_session(self.recipe_index(calibration)?)
    }

    pub(crate) fn fingerprint_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&(&self.sessions, &self.lights, &self.masters))
            .map_err(|e| error(e.to_string()))
    }

    /// Check `path` against its saved stamp, and against the physical
    /// identity this process first saw it with, returning that identity.
    fn verify(&self, path: &Path, expected: SessionStamp) -> Result<u64> {
        match observe(path) {
            Some((metadata, key))
                if metadata == expected.metadata && self.live.confirm(path, key) =>
            {
                Ok(key)
            }
            _ => Err(error(format!(
                "calibration-plan source {} changed or is missing",
                path.display()
            ))),
        }
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

    /// Calibrate `frame`, read from `path`, with its assigned session. The
    /// light is checked once; masters are checked only when they are read.
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
                *cache = Some((index, Arc::new(self.plan.load_session(index)?)));
            }
            Arc::clone(&cache.as_ref().expect("loaded session").1)
        };
        crate::stack::prepare_light_at(frame, &masters, &session.tolerances, options, bin)
            .map_err(crate::stack::LightFailure::into_error)
    }
}

/// A source's stamp and physical key, from one metadata read where the
/// platform allows. The key only buckets files: an alias check confirms a
/// match with the files themselves.
fn observe(path: &Path) -> Option<(SourceStamp, u64)> {
    let metadata = std::fs::metadata(path).ok()?;
    Some((
        SourceStamp::from_metadata(&metadata)?,
        physical_key(path, &metadata)?,
    ))
}

#[cfg(unix)]
fn physical_key(_: &Path, metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(bucket(&(metadata.dev(), metadata.ino())))
}

#[cfg(not(unix))]
fn physical_key(path: &Path, _: &std::fs::Metadata) -> Option<u64> {
    // The file index needs an open handle. Drop it at once: large input
    // lists must not hold one descriptor per light.
    Some(bucket(&same_file::Handle::from_path(path).ok()?))
}

fn bucket(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn first_observation(path: &Path) -> Result<(SourceStamp, u64)> {
    observe(path).ok_or_else(|| {
        error(format!(
            "cannot fingerprint calibration-plan source {}",
            path.display()
        ))
    })
}

/// Paths by physical key, so finding one file under two names compares the
/// files only where their keys match, rather than every pair.
#[derive(Default)]
struct Buckets(HashMap<u64, Vec<PathBuf>>);

impl Buckets {
    fn aliases(&self, path: &Path, key: u64) -> Result<bool> {
        for other in self.0.get(&key).into_iter().flatten() {
            if same_file::is_same_file(other, path).map_err(|e| error(e.to_string()))? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn insert(&mut self, path: &Path, key: u64) {
        self.0.entry(key).or_default().push(path.to_path_buf());
    }
}

fn error(message: impl Into<String>) -> Error {
    Error::Calibration(message.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn file(root: &Path, name: &str) -> PathBuf {
        let path = root.join(name);
        std::fs::write(&path, b"source identity").unwrap();
        path
    }

    /// Replace `path` with a copy of itself: a new file, as a remounted or
    /// restored volume presents, with the same size and modification time.
    pub(crate) fn recreate(path: &Path) {
        let modified = std::fs::metadata(path).unwrap().modified().unwrap();
        let moved = path.with_extension("moved");
        std::fs::rename(path, &moved).unwrap();
        std::fs::copy(&moved, path).unwrap();
        std::fs::remove_file(&moved).unwrap();
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }

    /// A plan as a context saves and restores it.
    fn reopened(plan: &FileSessionPlan) -> FileSessionPlan {
        let plan: FileSessionPlan =
            serde_json::from_slice(&serde_json::to_vec(plan).unwrap()).unwrap();
        plan.validate().unwrap();
        plan
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

    /// A light that is a master by another name refuses the plan, as do two
    /// roles of a session given one file. A light that is another light's
    /// file is the ordinary opener's duplicate: refused when pushed.
    #[test]
    fn physical_aliases_are_not_distinct_inputs_or_master_roles() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let alias = root.path().join("alias.data");
        std::fs::hard_link(&light, &alias).unwrap();
        let plan = FileSessionPlan::new(
            &[light.clone(), alias.clone()],
            &[Default::default()],
            &[0, 0],
        )
        .unwrap();
        assert!(plan.is_duplicate(&alias) && !plan.is_duplicate(&light));
        assert_eq!(plan.assignment(&light).unwrap(), 0);
        // Planning still prepares it, as the ordinary planner prepares a
        // repeated path.
        assert_eq!(plan.assignment(&alias).unwrap(), 0);
        // A saved plan keeps the duplicate.
        assert!(reopened(&plan).is_duplicate(&alias));
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

    /// Twin catalog rows: the first assignment stands, and the repeat is
    /// left to the stack's duplicate check.
    #[test]
    fn a_light_given_twice_keeps_its_first_assignment() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let other = file(root.path(), "other.data");
        let plan = FileSessionPlan::new(
            &[light.clone(), other.clone(), light.clone()],
            &[Default::default(), Default::default()],
            &[1, 0, 0],
        )
        .unwrap();
        assert_eq!(plan.lights.len(), 2);
        assert_eq!(plan.assignment(&light).unwrap(), 1);
        assert_eq!(plan.assignment(&other).unwrap(), 0);
        assert!(!plan.is_duplicate(&light));
    }

    /// Within one process a file replaced under the same size and time is
    /// still told apart by its physical identity.
    #[test]
    fn replacing_a_file_with_preserved_length_and_time_invalidates_the_plan() {
        let root = tempfile::tempdir().unwrap();
        let path = file(root.path(), "light.data");
        let plan =
            FileSessionPlan::new(std::slice::from_ref(&path), &[Default::default()], &[0]).unwrap();
        recreate(&path);
        assert_eq!(
            SourceStamp::of(&path),
            Some(plan.lights[&path_identity(&path)].1.metadata)
        );
        assert!(plan.assignment(&path).is_err());
    }

    /// A saved plan identifies its sources by size and modification time
    /// alone, so files that come back as new inodes or on a renumbered device
    /// still verify; one whose size or time changed does not.
    #[test]
    fn a_saved_plan_verifies_sources_by_size_and_time() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let master = file(root.path(), "bias.data");
        let session = FileCalibrationSession {
            bias: Some(master.clone()),
            ..Default::default()
        };
        let mut plan =
            FileSessionPlan::new(std::slice::from_ref(&light), &[session], &[0]).unwrap();
        plan.bind(0, 0);
        let saved = serde_json::to_string(&plan).unwrap();
        assert!(!saved.contains("identity"), "{saved}");
        let restored = reopened(&plan);
        recreate(&light);
        recreate(&master);
        assert_eq!(restored.assignment(&light).unwrap(), 0);
        restored.verify_recipe(0).unwrap();
        assert_eq!(
            restored.fingerprint_bytes().unwrap(),
            plan.fingerprint_bytes().unwrap()
        );
        // The live plan still sees both files were replaced.
        assert!(plan.assignment(&light).is_err());
        assert!(plan.verify_recipe(0).is_err());
        // Once seen, a file's identity holds for the rest of the process.
        recreate(&light);
        assert!(restored.assignment(&light).is_err());
        std::fs::write(&master, b"rebuilt master").unwrap();
        assert!(reopened(&plan).verify_recipe(0).is_err());
    }

    /// 0.46 saved each stamp with a hash of the device and inode. Such a plan
    /// still reads, and its sources verify by size and time.
    #[test]
    fn a_plan_saved_with_device_identities_still_reads() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let master = file(root.path(), "bias.data");
        let session = FileCalibrationSession {
            bias: Some(master),
            ..Default::default()
        };
        let mut plan =
            FileSessionPlan::new(std::slice::from_ref(&light), &[session], &[0]).unwrap();
        plan.bind(0, 0);
        let mut saved = serde_json::to_value(&plan).unwrap();
        with_device_identities(&mut saved);
        let old: FileSessionPlan = serde_json::from_value(saved).unwrap();
        old.validate().unwrap();
        recreate(&light);
        assert_eq!(old.assignment(&light).unwrap(), 0);
        old.verify_recipe(0).unwrap();
        assert_eq!(
            old.fingerprint_bytes().unwrap(),
            plan.fingerprint_bytes().unwrap()
        );
    }

    /// Give every stamp of a saved plan the `identity` 0.46 wrote.
    pub(crate) fn with_device_identities(plan: &mut serde_json::Value) {
        for (_, light) in plan["lights"].as_object_mut().unwrap() {
            light[1]["identity"] = serde_json::json!(0x0123_4567_89ab_cdef_u64);
        }
        for (_, master) in plan["masters"].as_object_mut().unwrap() {
            master["identity"] = serde_json::json!(0xfedc_ba98_7654_3210_u64);
        }
    }

    /// Validating a saved plan reads no source: a light gone since fails
    /// only itself, and a session whose master changed only its own lights.
    #[test]
    fn a_saved_plan_checks_each_source_only_when_it_is_used() {
        let root = tempfile::tempdir().unwrap();
        let lights = (0..3)
            .map(|i| file(root.path(), &format!("{i}.data")))
            .collect::<Vec<_>>();
        let sessions = (0..2)
            .map(|i| FileCalibrationSession {
                bias: Some(file(root.path(), &format!("bias-{i}.data"))),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let plan = FileSessionPlan::new(&lights, &sessions, &[0, 0, 1]).unwrap();
        std::fs::remove_file(&lights[1]).unwrap();
        std::fs::write(sessions[1].bias.as_ref().unwrap(), b"rebuilt master").unwrap();
        let restored = reopened(&plan);
        assert_eq!(restored.assignment(&lights[0]).unwrap(), 0);
        restored.verify_session(0).unwrap();
        assert!(restored.assignment(&lights[1]).is_err());
        let changed = restored.assignment(&lights[2]).unwrap_err().to_string();
        assert!(changed.contains("bias-1"), "{changed}");
        assert!(restored.verify_session(1).is_err());
    }

    #[test]
    fn a_saved_plan_must_agree_with_itself() {
        let root = tempfile::tempdir().unwrap();
        let light = file(root.path(), "light.data");
        let session = FileCalibrationSession {
            bias: Some(file(root.path(), "bias.data")),
            ..Default::default()
        };
        let plan = FileSessionPlan::new(std::slice::from_ref(&light), &[session], &[0]).unwrap();
        let saved = serde_json::to_value(&plan).unwrap();
        let read = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut value = saved.clone();
            edit(&mut value);
            serde_json::from_value::<FileSessionPlan>(value)
                .unwrap()
                .validate()
        };
        read(&|_| {}).unwrap();
        assert!(read(&|v| v["masters"] = serde_json::json!({})).is_err());
        assert!(read(&|v| v["sessions"][0]["bias"] = serde_json::Value::Null).is_err());
        assert!(read(&|v| v["calibrations"] = serde_json::json!({"0": 1})).is_err());
        let light = path_identity(&light).to_str().unwrap().to_owned();
        assert!(read(&|v| v["lights"][&light][0] = serde_json::json!(1)).is_err());
        assert!(read(&|v| v["duplicates"] = serde_json::json!(["elsewhere"])).is_err());
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
        for path in &paths {
            assert_eq!(plan.assignment(path).unwrap(), 0);
        }
    }
}
