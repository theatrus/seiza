//! Star distance tiles: Gaia DR3 stars with their Bailer-Jones distances,
//! and the Hipparcos stars, for placing a field's stars in depth offline.
//!
//! Format `SEIZADI1` (little-endian), the [`super::tiles`] layout with two
//! more columns:
//!
//! ```text
//! magic        [u8; 8]  = b"SEIZADI1"
//! n_bands      u32          declination bands from -90° to +90°
//! epoch        f64          positions are proper-motion corrected to this year
//! star_count   u64
//! max_mag      f32          the faintest Gaia G magnitude the file holds
//! attribution  u16 len + UTF-8 bytes (source + license note)
//! padding      to an 8-byte boundary
//! index        n_tiles ×  { offset: u64, count: u32 }
//! data         per tile, columnar: ra[u32; n]  dec[u32; n]  mag[u16; n]
//!              distance[u16; n]  flags[u8; n]  (each tile padded to 4 bytes)
//! ```
//!
//! `mag` is Gaia G, or Hipparcos Hp for a Hipparcos star, in the star tiles'
//! quantization. `distance` is the distance in parsecs on a log scale from
//! 1 pc to 100 kpc, or 0 without one. `flags` bit 0 marks a Hipparcos star.
//! Records within a tile are sorted brightest-first.

use super::tiles::{Grid, pack_dec, pack_mag, pack_ra, unpack_dec, unpack_mag, unpack_ra};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZADI1";
const RECORD_SIZE: u64 = 4 + 4 + 2 + 2 + 1;
const HEADER: usize = 8 + 4 + 8 + 8 + 4 + 2;
const HIPPARCOS: u8 = 1;
/// The distance scale: decades above 1 pc that the 65,534 steps span.
const DECADES: f64 = 5.0;

/// A star with its distance, at the catalogue's epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceStar {
    /// ICRS degrees.
    pub ra: f64,
    pub dec: f64,
    /// Gaia G magnitude, or Hipparcos Hp for a Hipparcos star.
    pub mag: f32,
    /// Distance in parsecs, when known.
    pub distance_pc: Option<f64>,
    /// A star from the Hipparcos catalogue rather than Gaia: the brightest
    /// stars, which Gaia measures poorly or not at all.
    pub hipparcos: bool,
}

fn pack_distance(distance_pc: Option<f64>) -> u16 {
    match distance_pc {
        Some(distance) if distance.is_finite() && distance > 0.0 => {
            let step = (distance.max(1.0).log10() / DECADES * 65_534.0).round();
            1 + step.clamp(0.0, 65_534.0) as u16
        }
        _ => 0,
    }
}

fn unpack_distance(q: u16) -> Option<f64> {
    (q != 0).then(|| 10_f64.powf(f64::from(q - 1) / 65_534.0 * DECADES))
}

/// One packed record: RA, Dec, magnitude, distance and flags.
type PackedStar = (u32, u32, u16, u16, u8);

/// Accumulates stars in memory, then writes a `SEIZADI1` file.
pub struct StarDistanceCatalogBuilder {
    grid: Grid,
    epoch: f64,
    max_mag: f32,
    attribution: String,
    tiles: Vec<Vec<PackedStar>>,
    count: u64,
}

impl StarDistanceCatalogBuilder {
    /// `n_bands` sets the tile size: 90 bands gives tiles of about 2°.
    /// `max_mag` records the Gaia magnitude limit the stars were fetched to,
    /// and `attribution` the data sources and licenses, inside the file.
    pub fn new(n_bands: u32, epoch: f64, max_mag: f32, attribution: &str) -> Self {
        let grid = Grid::new(n_bands);
        let tiles = vec![Vec::new(); grid.n_tiles() as usize];
        Self {
            grid,
            epoch,
            max_mag,
            attribution: attribution.to_string(),
            tiles,
            count: 0,
        }
    }

    pub fn add(&mut self, star: DistanceStar) {
        let tile = self.grid.tile_of(star.ra, star.dec);
        self.tiles[tile as usize].push((
            pack_ra(star.ra),
            pack_dec(star.dec),
            pack_mag(star.mag),
            pack_distance(star.distance_pc),
            if star.hipparcos { HIPPARCOS } else { 0 },
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
        out.write_all(&self.max_mag.to_le_bytes())?;
        let attribution = self.attribution.as_bytes();
        let attr_len = attribution.len().min(u16::MAX as usize);
        out.write_all(&(attr_len as u16).to_le_bytes())?;
        out.write_all(&attribution[..attr_len])?;

        let mut position = (HEADER + attr_len) as u64;
        let pad = position.next_multiple_of(8) - position;
        out.write_all(&vec![0u8; pad as usize])?;
        position += pad;

        let n_tiles = self.grid.n_tiles() as u64;
        let mut offset = position + n_tiles * 12;
        for tile in &mut self.tiles {
            tile.sort_by_key(|&(_, _, mag, _, _)| mag);
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
            for &(_, _, mag, ..) in tile {
                out.write_all(&mag.to_le_bytes())?;
            }
            for &(_, _, _, distance, _) in tile {
                out.write_all(&distance.to_le_bytes())?;
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

/// A read-only, memory-mapped `SEIZADI1` catalogue.
pub struct StarDistanceCatalog {
    map: memmap2::Mmap,
    grid: Grid,
    epoch: f64,
    star_count: u64,
    max_mag: f32,
    attribution: String,
    index: Vec<(u64, u32)>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

impl StarDistanceCatalog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation would
        // fault, which is acceptable for locally managed data files.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < HEADER || &map[..8] != MAGIC {
            return Err(invalid("not a SEIZADI1 star distance catalog"));
        }
        let u32_at = |at: usize| u32::from_le_bytes(map[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(map[at..at + 8].try_into().unwrap());
        let n_bands = u32_at(8);
        if n_bands == 0 || n_bands > 10_000 {
            return Err(invalid("star distance catalog has an invalid band count"));
        }
        let epoch = f64::from_le_bytes(map[12..20].try_into().unwrap());
        let star_count = u64_at(20);
        let max_mag = f32::from_le_bytes(map[28..32].try_into().unwrap());
        let attr_len = u16::from_le_bytes(map[32..34].try_into().unwrap()) as usize;
        if map.len() < HEADER + attr_len {
            return Err(invalid("star distance catalog header is truncated"));
        }
        let attribution = String::from_utf8_lossy(&map[HEADER..HEADER + attr_len]).into_owned();
        let grid = Grid::new(n_bands);
        let index_start = ((HEADER + attr_len) as u64).next_multiple_of(8) as usize;
        let n_tiles = grid.n_tiles() as usize;
        let index_end = index_start + n_tiles * 12;
        if map.len() < index_end {
            return Err(invalid("star distance catalog index is truncated"));
        }
        let mut index = Vec::with_capacity(n_tiles);
        for tile in 0..n_tiles {
            let at = index_start + tile * 12;
            let (offset, count) = (u64_at(at), u32_at(at + 8));
            let end = offset
                .checked_add(u64::from(count) * RECORD_SIZE)
                .ok_or_else(|| invalid("star distance catalog tile overflows"))?;
            if offset < index_end as u64 || end > map.len() as u64 {
                return Err(invalid("star distance catalog tile lies outside the file"));
            }
            index.push((offset, count));
        }
        Ok(Self {
            map,
            grid,
            epoch,
            star_count,
            max_mag,
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

    /// The faintest Gaia G magnitude the file holds: a search to a fainter
    /// limit misses stars.
    pub fn max_mag(&self) -> f32 {
        self.max_mag
    }

    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// Stars within `radius_deg` of `(ra, dec)` down to magnitude
    /// `max_mag`, Gaia and Hipparcos alike.
    pub fn cone_search(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        max_mag: f32,
    ) -> Vec<DistanceStar> {
        let (sin_dec, cos_dec) = dec.to_radians().sin_cos();
        let cos_radius = radius_deg.to_radians().cos();
        let mut stars = Vec::new();
        for tile in self.grid.cone_tiles(ra, dec, radius_deg) {
            let (offset, count) = self.index[tile as usize];
            let (offset, n) = (offset as usize, count as usize);
            let ras = &self.map[offset..offset + 4 * n];
            let decs = &self.map[offset + 4 * n..offset + 8 * n];
            let mags = &self.map[offset + 8 * n..offset + 10 * n];
            let distances = &self.map[offset + 10 * n..offset + 12 * n];
            let flags = &self.map[offset + 12 * n..offset + 13 * n];
            for record in 0..n {
                let mag = unpack_mag(u16::from_le_bytes([mags[2 * record], mags[2 * record + 1]]));
                // Sorted brightest-first: the rest of the tile is fainter.
                if mag > max_mag {
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
                stars.push(DistanceStar {
                    ra: star_ra,
                    dec: star_dec,
                    mag,
                    distance_pc: unpack_distance(u16::from_le_bytes([
                        distances[2 * record],
                        distances[2 * record + 1],
                    ])),
                    hipparcos: flags[record] & HIPPARCOS != 0,
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
    fn distances_keep_a_ten_thousandth_from_a_parsec_to_a_hundred_kiloparsecs() {
        for distance in [1.3, 136.0, 410.0, 2790.0, 8_000.0, 50_000.0] {
            let back = unpack_distance(pack_distance(Some(distance))).unwrap();
            assert!((back / distance - 1.0).abs() < 1e-4, "{distance} {back}");
        }
        assert_eq!(unpack_distance(pack_distance(None)), None);
        assert_eq!(unpack_distance(pack_distance(Some(f64::NAN))), None);
        assert_eq!(unpack_distance(pack_distance(Some(0.5))), Some(1.0));
    }

    #[test]
    fn stars_round_trip_through_a_file_and_cone_search() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("distances.bin");
        let mut builder = StarDistanceCatalogBuilder::new(90, 2026.0, 16.0, "test");
        let stars = [
            DistanceStar {
                ra: 56.75,
                dec: 24.1,
                mag: 2.8,
                distance_pc: Some(136.2),
                hipparcos: true,
            },
            DistanceStar {
                ra: 56.8,
                dec: 24.2,
                mag: 12.5,
                distance_pc: None,
                hipparcos: false,
            },
            DistanceStar {
                ra: 359.95,
                dec: 0.0,
                mag: 9.0,
                distance_pc: Some(1200.0),
                hipparcos: false,
            },
        ];
        for star in stars {
            builder.add(star);
        }
        builder.write_to(&path).unwrap();
        let catalog = StarDistanceCatalog::open(&path).unwrap();
        assert_eq!(
            (catalog.star_count(), catalog.epoch(), catalog.max_mag()),
            (3, 2026.0, 16.0)
        );
        let found = catalog.cone_search(56.75, 24.1, 0.5, 20.0);
        assert_eq!(found.len(), 2);
        assert!(found[0].hipparcos && !found[1].hipparcos);
        assert!((found[0].distance_pc.unwrap() - 136.2).abs() < 0.02);
        assert_eq!(found[1].distance_pc, None);
        assert_eq!(catalog.cone_search(56.75, 24.1, 0.5, 10.0).len(), 1);
        // A cone across RA 0 finds the star just short of 360°.
        let wrapped = catalog.cone_search(0.02, 0.0, 0.2, 20.0);
        assert_eq!(wrapped.len(), 1);
        assert!((wrapped[0].distance_pc.unwrap() - 1200.0).abs() < 0.2);
        assert!(StarDistanceCatalog::open(directory.path().join("missing").as_path()).is_err());
    }
}
