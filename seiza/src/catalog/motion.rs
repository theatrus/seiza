//! Proper motions for a star tile file, kept in a sidecar.
//!
//! A [`super::TileCatalog`] holds positions at one epoch, 2025.5 for the
//! published Gaia files. Stars move: most by a few mas a year, the fastest
//! by arcseconds. `SEIZAPM1` holds each star's proper motion, so a solve
//! can move the catalog to the date of an image ([`EpochCatalog`]); the
//! star file itself is unchanged, and a solve without the sidecar or a
//! date works as before.
//!
//! The sidecar follows the star file's own record order and pairs only
//! with the file it was written with ([`super::TileSetBuilder::write_with_motions`]),
//! which its header records.
//!
//! ```text
//! magic        [u8; 8]  = b"SEIZAPM1"
//! n_bands      u32          the star file's
//! reserved     u32          0
//! epoch        f64          the star file's: motions start there
//! star_count   u64          the star file's
//! fingerprint  u64          FNV-1a of the star file's tile index
//! data         per tile, in the star file's order and record order:
//!              pmra[i16; n]  pmdec[i16; n]
//! ```
//!
//! Motions are whole mas/yr, `pmra` times cos(dec); `i16::MIN` marks a star
//! without one, which stays where the star file has it.

use super::tiles::NO_MOTION;
use super::{CatalogStar, StarCatalog, TileCatalog, propagate_proper_motion};
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SEIZAPM1";
const HEADER_SIZE: usize = 40;

pub(super) fn write_header(
    out: &mut impl Write,
    n_bands: u32,
    epoch: f64,
    star_count: u64,
    fingerprint: u64,
) -> io::Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&n_bands.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?;
    out.write_all(&epoch.to_le_bytes())?;
    out.write_all(&star_count.to_le_bytes())?;
    out.write_all(&fingerprint.to_le_bytes())
}

/// The proper motions of one star tile file, memory-mapped.
pub struct ProperMotions {
    map: memmap2::Mmap,
    epoch: f64,
    /// Where each tile's motions start in `map`, and how many stars it has.
    tiles: Vec<(usize, usize)>,
}

impl ProperMotions {
    /// Where the sidecar of `star_file` lives: `stars-gaia.bin` has
    /// `stars-gaia.pm.bin`.
    pub fn path_for(star_file: &Path) -> PathBuf {
        star_file.with_extension("pm.bin")
    }

    /// Open the sidecar at `path` for `catalog`, refusing one written with
    /// another star file.
    pub fn open(path: &Path, catalog: &TileCatalog) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation would
        // fault, which is acceptable for locally-managed data files.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < HEADER_SIZE || &map[..8] != MAGIC {
            return Err(invalid_data("not a seiza proper-motion file"));
        }
        let n_bands = u32::from_le_bytes(map[8..12].try_into().unwrap());
        let epoch = f64::from_le_bytes(map[16..24].try_into().unwrap());
        let star_count = u64::from_le_bytes(map[24..32].try_into().unwrap());
        let fingerprint = u64::from_le_bytes(map[32..40].try_into().unwrap());
        if n_bands != catalog.n_bands()
            || epoch.to_bits() != catalog.epoch().to_bits()
            || star_count != catalog.star_count()
            || fingerprint != catalog.index_fingerprint()
        {
            return Err(invalid_data(
                "the proper-motion file was written with another star file",
            ));
        }
        let mut tiles = Vec::new();
        let mut at = HEADER_SIZE;
        for count in catalog.tile_counts() {
            let count = count as usize;
            tiles.push((at, count));
            at += count * 4;
        }
        if at != map.len() {
            return Err(invalid_data(
                "the proper-motion file's length does not match its star file",
            ));
        }
        Ok(Self { map, epoch, tiles })
    }

    /// The sidecar beside `star_file` ([`Self::path_for`]), or `None` when
    /// it has none. One written with another star file is an error.
    pub fn open_beside(star_file: &Path, catalog: &TileCatalog) -> io::Result<Option<Self>> {
        match Self::open(&Self::path_for(star_file), catalog) {
            Ok(motions) => Ok(Some(motions)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// The epoch motions start from: the star file's.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    /// The motion of the star at `index` in `tile`, mas/yr (`pmra` times
    /// cos(dec), then `pmdec`), or `None` without one.
    fn motion(&self, tile: u32, index: usize) -> Option<(f64, f64)> {
        let (start, count) = *self.tiles.get(tile as usize)?;
        if index >= count {
            return None;
        }
        let read = |at: usize| i16::from_le_bytes([self.map[at], self.map[at + 1]]);
        let pmra = read(start + index * 2);
        let pmdec = read(start + count * 2 + index * 2);
        (pmra != NO_MOTION && pmdec != NO_MOTION).then(|| (f64::from(pmra), f64::from(pmdec)))
    }
}

/// A [`TileCatalog`] moved to another epoch by its [`ProperMotions`]: what a
/// solve searches when it knows an image's date.
pub struct EpochCatalog<'a> {
    stars: &'a TileCatalog,
    motions: &'a ProperMotions,
    years: f64,
}

impl<'a> EpochCatalog<'a> {
    /// `stars` as they were at `epoch`, a Julian year such as 2023.4.
    pub fn new(stars: &'a TileCatalog, motions: &'a ProperMotions, epoch: f64) -> Self {
        Self {
            stars,
            motions,
            years: epoch - motions.epoch(),
        }
    }
}

impl StarCatalog for EpochCatalog<'_> {
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        self.stars
            .cone_search_moved(ra, dec, radius_deg, limit, |tile, index, star| {
                match self.motions.motion(tile, index) {
                    Some((pmra, pmdec)) => {
                        let (ra, dec) =
                            propagate_proper_motion(star.ra, star.dec, pmra, pmdec, self.years);
                        CatalogStar { ra, dec, ..star }
                    }
                    None => star,
                }
            })
    }

    fn star_count(&self) -> u64 {
        self.stars.star_count()
    }
}

/// A star file as one image's solve searches it: moved to the image's date
/// when there are proper motions and a date, and as written otherwise.
pub enum DatedCatalog<'a> {
    AsWritten(&'a TileCatalog),
    Moved(EpochCatalog<'a>),
}

impl<'a> DatedCatalog<'a> {
    /// `stars` at `epoch` (a Julian year) when `motions` and `epoch` are
    /// both given, else `stars` at their own epoch.
    pub fn new(
        stars: &'a TileCatalog,
        motions: Option<&'a ProperMotions>,
        epoch: Option<f64>,
    ) -> Self {
        match (motions, epoch.filter(|epoch| epoch.is_finite())) {
            (Some(motions), Some(epoch)) => Self::Moved(EpochCatalog::new(stars, motions, epoch)),
            _ => Self::AsWritten(stars),
        }
    }

    /// The Julian year the positions searched are for.
    pub fn epoch(&self) -> f64 {
        match self {
            Self::AsWritten(stars) => stars.epoch(),
            Self::Moved(moved) => moved.motions.epoch() + moved.years,
        }
    }

    /// Whether the positions were moved from the star file's.
    pub fn is_moved(&self) -> bool {
        matches!(self, Self::Moved(_))
    }
}

impl StarCatalog for DatedCatalog<'_> {
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        match self {
            Self::AsWritten(stars) => stars.cone_search(ra, dec, radius_deg, limit),
            Self::Moved(moved) => moved.cone_search(ra, dec, radius_deg, limit),
        }
    }

    fn star_count(&self) -> u64 {
        match self {
            Self::AsWritten(stars) => stars.star_count(),
            Self::Moved(moved) => moved.star_count(),
        }
    }

    fn all_brighter_than(&self, mag_limit: f32) -> Vec<CatalogStar> {
        match self {
            Self::AsWritten(stars) => stars.all_brighter_than(mag_limit),
            Self::Moved(moved) => moved.all_brighter_than(mag_limit),
        }
    }
}

/// The Julian year of a Julian date, such as an image's capture time from
/// [`crate::field_objects::capture_julian_date`].
pub fn julian_year(julian_date: f64) -> f64 {
    2000.0 + (julian_date - 2_451_545.0) / 365.25
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::TileSetBuilder;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("seiza-motion-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Barnard's Star at its Gaia DR3 position and motion, with neighbours
    /// that do not move, one of them as bright so the stable order matters.
    fn builder() -> TileSetBuilder {
        let mut builder = TileSetBuilder::new(180, 2025.5, "test");
        builder.add(269.40, 4.70, 9.5);
        builder.add_moving(269.452, 4.6933, 8.2, Some((-801.6, 10362.4)));
        builder.add_moving(269.50, 4.68, 8.2, Some((0.0, 0.0)));
        builder.add_moving(269.47, 4.72, 11.0, None);
        builder
    }

    #[test]
    fn a_sidecar_leaves_the_star_file_as_it_was() {
        let dir = scratch("same");
        builder().write_to(&dir.join("plain.bin")).unwrap();
        builder()
            .write_with_motions(&dir.join("stars.bin"), &dir.join("stars.pm.bin"))
            .unwrap();
        assert_eq!(
            std::fs::read(dir.join("plain.bin")).unwrap(),
            std::fs::read(dir.join("stars.bin")).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_epoch_catalog_moves_each_star_by_its_own_motion() {
        let dir = scratch("move");
        let stars_path = dir.join("stars.bin");
        let motions_path = ProperMotions::path_for(&stars_path);
        builder()
            .write_with_motions(&stars_path, &motions_path)
            .unwrap();
        let stars = TileCatalog::open(&stars_path).unwrap();
        let motions = ProperMotions::open(&motions_path, &stars).unwrap();
        assert_eq!(motions.epoch(), 2025.5);

        let then = EpochCatalog::new(&stars, &motions, 2015.5);
        let now = stars.cone_search(269.45, 4.7, 1.0, 10);
        let before = then.cone_search(269.45, 4.7, 1.0, 10);
        assert_eq!(now.len(), 4);
        assert_eq!(before.len(), 4);
        for (now, before) in now.iter().zip(&before) {
            assert_eq!(now.mag, before.mag);
            let moved_arcsec =
                crate::catalog::angular_separation_deg(now.ra, now.dec, before.ra, before.dec)
                    * 3600.0;
            // Barnard's Star moved 103.9" in ten years, mostly north; the others did
            // not move, or have no motion.
            if (now.dec - 4.6933).abs() < 1e-3 {
                assert!((moved_arcsec - 103.93).abs() < 0.05, "{moved_arcsec}");
                assert!(before.dec < now.dec);
            } else {
                assert!(moved_arcsec < 1e-3, "{moved_arcsec}");
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_sidecar_pairs_only_with_its_own_star_file() {
        let dir = scratch("pair");
        builder()
            .write_with_motions(&dir.join("a.bin"), &dir.join("a.pm.bin"))
            .unwrap();
        let mut other = builder();
        other.add(10.0, 10.0, 12.0);
        other.write_to(&dir.join("b.bin")).unwrap();
        let other = TileCatalog::open(&dir.join("b.bin")).unwrap();
        let error = ProperMotions::open(&dir.join("a.pm.bin"), &other)
            .err()
            .expect("a sidecar of another star file is refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
