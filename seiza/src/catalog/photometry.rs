//! Star photometry tiles for colour calibration.
//!
//! Format `SEIZAPH1` (little-endian), the [`super::tiles`] layout with two
//! more columns:
//!
//! ```text
//! magic        [u8; 8]  = b"SEIZAPH1"
//! n_bands      u32          declination bands from -90° to +90°
//! epoch        f64          positions are proper-motion corrected to this year
//! star_count   u64
//! attribution  u16 len + UTF-8 bytes (source + license note)
//! padding      to an 8-byte boundary
//! index        n_tiles ×  { offset: u64, count: u32 }
//! data         per tile, columnar: ra[u32; n]  dec[u32; n]  g[u16; n]
//!              bp_rp[i16; n]  flags[u8; n]  (each tile padded to 4 bytes)
//! ```
//!
//! `g` is the Gaia G magnitude in the star tiles' quantization. `bp_rp` is
//! BP − RP in millimagnitudes, or `i16::MIN` without one. `flags` bit 0
//! marks a source whose colour should not calibrate: a RUWE of 1.4 or more,
//! which flags likely binaries and blends. Records within a tile are sorted
//! brightest-first.

use super::tiles::{Grid, pack_dec, pack_mag, pack_ra, unpack_dec, unpack_mag, unpack_ra};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZAPH1";
const RECORD_SIZE: u64 = 4 + 4 + 2 + 2 + 1;
const NO_COLOUR: i16 = i16::MIN;
const UNRELIABLE: u8 = 1;

/// A catalogue star with Gaia photometry, at the catalogue's epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotometricStar {
    /// ICRS degrees.
    pub ra: f64,
    pub dec: f64,
    /// Gaia G magnitude.
    pub g: f32,
    /// Gaia BP − RP colour, when measured.
    pub bp_rp: Option<f32>,
    /// False for a source whose colour should not calibrate (RUWE ≥ 1.4).
    pub reliable: bool,
}

/// Move a position `years` along its proper motion (mas/yr, `pmra` times
/// cos(dec)) on the sphere: along the great circle the motion starts on,
/// which stays right near the poles where stepping RA would not.
pub fn propagate_proper_motion(ra: f64, dec: f64, pmra: f64, pmdec: f64, years: f64) -> (f64, f64) {
    let (sin_ra, cos_ra) = ra.to_radians().sin_cos();
    let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
    let position = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
    // Unit vectors towards increasing RA (east) and Dec (north).
    let east = [-sin_ra, cos_ra, 0.0];
    let north = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];
    let to_radians = years / 3.6e6 * std::f64::consts::PI / 180.0;
    let (de, dn) = (pmra * to_radians, pmdec * to_radians);
    let moved = [0, 1, 2].map(|axis| position[axis] + de * east[axis] + dn * north[axis]);
    let norm = moved.iter().map(|value| value * value).sum::<f64>().sqrt();
    let [x, y, z] = moved.map(|value| value / norm);
    (
        y.atan2(x).to_degrees().rem_euclid(360.0),
        z.asin().to_degrees(),
    )
}

fn pack_colour(bp_rp: Option<f32>) -> i16 {
    match bp_rp {
        Some(colour) if colour.is_finite() => {
            (colour * 1000.0).round().clamp(-32_767.0, 32_767.0) as i16
        }
        _ => NO_COLOUR,
    }
}

fn unpack_colour(q: i16) -> Option<f32> {
    (q != NO_COLOUR).then(|| f32::from(q) / 1000.0)
}

/// One packed record: RA, Dec, G, BP − RP and flags.
type PackedStar = (u32, u32, u16, i16, u8);

/// Accumulates stars in memory, then writes a `SEIZAPH1` file.
pub struct PhotometryCatalogBuilder {
    grid: Grid,
    epoch: f64,
    attribution: String,
    tiles: Vec<Vec<PackedStar>>,
    count: u64,
}

impl PhotometryCatalogBuilder {
    /// `n_bands` sets the tile size: 90 bands gives tiles of about 2°.
    /// `attribution` records the data source and license inside the file.
    pub fn new(n_bands: u32, epoch: f64, attribution: &str) -> Self {
        let grid = Grid::new(n_bands);
        let tiles = vec![Vec::new(); grid.n_tiles() as usize];
        Self {
            grid,
            epoch,
            attribution: attribution.to_string(),
            tiles,
            count: 0,
        }
    }

    pub fn add(&mut self, star: PhotometricStar) {
        let tile = self.grid.tile_of(star.ra, star.dec);
        self.tiles[tile as usize].push((
            pack_ra(star.ra),
            pack_dec(star.dec),
            pack_mag(star.g),
            pack_colour(star.bp_rp),
            if star.reliable { 0 } else { UNRELIABLE },
        ));
        self.count += 1;
    }

    pub fn star_count(&self) -> u64 {
        self.count
    }

    pub fn write_to(mut self, path: &Path) -> io::Result<()> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&self.grid.n_bands.to_le_bytes())?;
        out.write_all(&self.epoch.to_le_bytes())?;
        out.write_all(&self.count.to_le_bytes())?;
        let attribution = self.attribution.as_bytes();
        let attr_len = attribution.len().min(u16::MAX as usize);
        out.write_all(&(attr_len as u16).to_le_bytes())?;
        out.write_all(&attribution[..attr_len])?;

        let mut position = 8 + 4 + 8 + 8 + 2 + attr_len as u64;
        let pad = position.next_multiple_of(8) - position;
        out.write_all(&vec![0u8; pad as usize])?;
        position += pad;

        let n_tiles = self.grid.n_tiles() as u64;
        let mut offset = position + n_tiles * 12;
        for tile in &mut self.tiles {
            tile.sort_by_key(|&(_, _, g, _, _)| g);
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&(tile.len() as u32).to_le_bytes())?;
            offset += (tile.len() as u64 * RECORD_SIZE).next_multiple_of(4);
        }
        for tile in &self.tiles {
            for &(ra, ..) in tile {
                out.write_all(&ra.to_le_bytes())?;
            }
            for &(_, dec, ..) in tile {
                out.write_all(&dec.to_le_bytes())?;
            }
            for &(_, _, g, ..) in tile {
                out.write_all(&g.to_le_bytes())?;
            }
            for &(_, _, _, colour, _) in tile {
                out.write_all(&colour.to_le_bytes())?;
            }
            for &(.., flags) in tile {
                out.write_all(&[flags])?;
            }
            let data = tile.len() as u64 * RECORD_SIZE;
            out.write_all(&vec![0u8; (data.next_multiple_of(4) - data) as usize])?;
        }
        out.flush()
    }
}

/// A read-only, memory-mapped `SEIZAPH1` catalogue.
pub struct PhotometryCatalog {
    map: memmap2::Mmap,
    grid: Grid,
    epoch: f64,
    star_count: u64,
    attribution: String,
    index: Vec<(u64, u32)>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

impl PhotometryCatalog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation would
        // fault, which is acceptable for locally managed data files.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < 30 || &map[..8] != MAGIC {
            return Err(invalid("not a SEIZAPH1 photometry catalog"));
        }
        let u32_at = |at: usize| u32::from_le_bytes(map[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(map[at..at + 8].try_into().unwrap());
        let n_bands = u32_at(8);
        if n_bands == 0 || n_bands > 10_000 {
            return Err(invalid("photometry catalog has an invalid band count"));
        }
        let epoch = f64::from_le_bytes(map[12..20].try_into().unwrap());
        let star_count = u64_at(20);
        let attr_len = u16::from_le_bytes(map[28..30].try_into().unwrap()) as usize;
        if map.len() < 30 + attr_len {
            return Err(invalid("photometry catalog header is truncated"));
        }
        let attribution = String::from_utf8_lossy(&map[30..30 + attr_len]).into_owned();
        let grid = Grid::new(n_bands);
        let index_start = ((30 + attr_len) as u64).next_multiple_of(8) as usize;
        let n_tiles = grid.n_tiles() as usize;
        let index_end = index_start + n_tiles * 12;
        if map.len() < index_end {
            return Err(invalid("photometry catalog index is truncated"));
        }
        let mut index = Vec::with_capacity(n_tiles);
        for tile in 0..n_tiles {
            let at = index_start + tile * 12;
            let (offset, count) = (u64_at(at), u32_at(at + 8));
            let end = offset
                .checked_add(u64::from(count) * RECORD_SIZE)
                .ok_or_else(|| invalid("photometry catalog tile overflows"))?;
            if offset < index_end as u64 || end > map.len() as u64 {
                return Err(invalid("photometry catalog tile lies outside the file"));
            }
            index.push((offset, count));
        }
        Ok(Self {
            map,
            grid,
            epoch,
            star_count,
            attribution,
            index,
        })
    }

    /// The year positions were proper-motion corrected to.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    pub fn star_count(&self) -> u64 {
        self.star_count
    }

    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// Stars within `radius_deg` of `(ra, dec)` down to G `max_mag`.
    pub fn cone_search(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Vec<PhotometricStar> {
        let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
        let cos_radius = radius_deg.to_radians().cos();
        let mut stars = Vec::new();
        for tile in self.grid.cone_tiles(ra, dec, radius_deg) {
            let (offset, count) = self.index[tile as usize];
            let (offset, n) = (offset as usize, count as usize);
            let column = |start: usize, width: usize| {
                &self.map[offset + start * n..offset + (start + width) * n]
            };
            let ras = column(0, 4);
            let decs = column(4, 4);
            let gs = &self.map[offset + 8 * n..offset + 10 * n];
            let colours = &self.map[offset + 10 * n..offset + 12 * n];
            let flags = &self.map[offset + 12 * n..offset + 13 * n];
            for record in 0..n {
                let g = unpack_mag(u16::from_le_bytes([gs[2 * record], gs[2 * record + 1]]));
                // Sorted brightest-first: the rest of the tile is fainter.
                if g > max_mag {
                    break;
                }
                let star_ra = unpack_ra(u32::from_le_bytes(
                    ras[4 * record..4 * record + 4].try_into().unwrap(),
                ));
                let star_dec = unpack_dec(u32::from_le_bytes(
                    decs[4 * record..4 * record + 4].try_into().unwrap(),
                ));
                let (sin_d, cos_d) = star_dec.to_radians().sin_cos();
                let cos_separation =
                    sin_dec * sin_d + cos_dec * cos_d * (star_ra - ra).to_radians().cos();
                if cos_separation < cos_radius {
                    continue;
                }
                stars.push(PhotometricStar {
                    ra: star_ra,
                    dec: star_dec,
                    g,
                    bp_rp: unpack_colour(i16::from_le_bytes([
                        colours[2 * record],
                        colours[2 * record + 1],
                    ])),
                    reliable: flags[record] & UNRELIABLE == 0,
                });
            }
        }
        stars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_round_trip_through_a_file_and_cone_search() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("phot.bin");
        let mut builder = PhotometryCatalogBuilder::new(90, 2026.0, "test");
        let stars = [
            PhotometricStar {
                ra: 56.75,
                dec: 24.1,
                g: 3.0,
                bp_rp: Some(-0.05),
                reliable: true,
            },
            PhotometricStar {
                ra: 56.8,
                dec: 24.2,
                g: 12.5,
                bp_rp: None,
                reliable: false,
            },
            PhotometricStar {
                ra: 359.95,
                dec: 0.0,
                g: 9.0,
                bp_rp: Some(1.234),
                reliable: true,
            },
            PhotometricStar {
                ra: 120.0,
                dec: -60.0,
                g: 8.0,
                bp_rp: Some(0.8),
                reliable: true,
            },
        ];
        for star in stars {
            builder.add(star);
        }
        assert_eq!(builder.star_count(), 4);
        builder.write_to(&path).unwrap();
        let catalog = PhotometryCatalog::open(&path).unwrap();
        assert_eq!((catalog.star_count(), catalog.epoch()), (4, 2026.0));
        let found = catalog.cone_search(56.75, 24.1, 0.5, 20.0);
        assert_eq!(found.len(), 2);
        assert!((found[0].ra - 56.75).abs() < 1e-6 && (found[0].dec - 24.1).abs() < 1e-6);
        assert_eq!(found[0].bp_rp, Some(-0.05));
        assert_eq!(found[1].bp_rp, None);
        assert!(!found[1].reliable);
        assert_eq!(catalog.cone_search(56.75, 24.1, 0.5, 10.0).len(), 1);
        // A cone across RA 0 finds the star just short of 360°.
        let wrapped = catalog.cone_search(0.02, 0.0, 0.2, 20.0);
        assert_eq!(wrapped.len(), 1);
        assert!((wrapped[0].bp_rp.unwrap() - 1.234).abs() < 1e-6);
        assert!(PhotometryCatalog::open(directory.path().join("missing").as_path()).is_err());
    }

    #[test]
    fn proper_motion_stays_on_the_sphere_at_the_pole() {
        // 10"/yr north for a century from 10" short of the north pole
        // carries the star across it: 990" past, on the far meridian.
        let (ra, dec) = propagate_proper_motion(30.0, 90.0 - 10.0 / 3600.0, 0.0, 10_000.0, 100.0);
        assert!((dec - (90.0 - 990.0 / 3600.0)).abs() < 1e-5, "{dec}");
        assert!((ra - 210.0).abs() < 1e-6, "{ra}");
        let (ra, dec) = propagate_proper_motion(56.75, 24.1, 1000.0, 0.0, 10.0);
        let expected = 56.75 + 10.0 / 3600.0 / 24.1_f64.to_radians().cos();
        assert!((ra - expected).abs() < 1e-7 && (dec - 24.1).abs() < 1e-6);
    }
}
