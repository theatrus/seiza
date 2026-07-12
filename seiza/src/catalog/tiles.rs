//! File-backed star catalog tiles.
//!
//! Binary format `SEIZAST1` (little-endian):
//!
//! ```text
//! magic        [u8; 8]  = b"SEIZAST1"
//! n_bands      u32          declination bands from -90° to +90°
//! epoch        f64          positions are proper-motion corrected to this year
//! star_count   u64
//! index        n_tiles ×  { offset: u64, count: u32 }
//! data         packed star records, per tile, brightest first
//! ```
//!
//! The sky is split into `n_bands` equal-height declination bands; each band
//! is split into RA bins whose count shrinks with `cos(dec)` so bins stay
//! roughly equal-area. A star record is 10 bytes: quantized RA (u32 over
//! 360°), quantized Dec (u32 over 180°), magnitude (u16, millimags offset
//! by +3 mag).
//!
//! Readers use positioned reads (`read_exact_at`), so a [`TileCatalog`] is
//! shareable across threads without interior locking.

use super::{CatalogStar, StarCatalog, angular_separation_deg};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZAST1";
const RECORD_SIZE: u64 = 10;
const HEADER_SIZE: u64 = 8 + 4 + 8 + 8;
const MAG_OFFSET: f32 = 3.0;

/// Sky-to-tile geometry shared by the builder and the reader.
#[derive(Debug, Clone)]
struct Grid {
    n_bands: u32,
    /// Number of RA bins in each band
    bins: Vec<u32>,
    /// Flat tile index of the first bin of each band
    offsets: Vec<u32>,
}

impl Grid {
    fn new(n_bands: u32) -> Self {
        assert!(n_bands >= 1);
        let band_height = 180.0 / n_bands as f64;
        let mut bins = Vec::with_capacity(n_bands as usize);
        let mut offsets = Vec::with_capacity(n_bands as usize);
        let mut total = 0u32;
        for band in 0..n_bands {
            let dec_mid = -90.0 + (band as f64 + 0.5) * band_height;
            let circumference = 360.0 * dec_mid.to_radians().cos().max(1e-6);
            let count = (circumference / band_height).ceil().max(1.0) as u32;
            offsets.push(total);
            bins.push(count);
            total += count;
        }
        Self {
            n_bands,
            bins,
            offsets,
        }
    }

    fn n_tiles(&self) -> u32 {
        self.offsets[self.n_bands as usize - 1] + self.bins[self.n_bands as usize - 1]
    }

    fn band_of(&self, dec: f64) -> u32 {
        let band = ((dec + 90.0) / 180.0 * self.n_bands as f64) as i64;
        band.clamp(0, self.n_bands as i64 - 1) as u32
    }

    fn tile_of(&self, ra: f64, dec: f64) -> u32 {
        let band = self.band_of(dec);
        let n = self.bins[band as usize];
        let ra = ra.rem_euclid(360.0);
        let bin = ((ra / 360.0 * n as f64) as u32).min(n - 1);
        self.offsets[band as usize] + bin
    }

    /// Tiles intersecting a cone (a covering superset).
    fn cone_tiles(&self, ra: f64, dec: f64, radius_deg: f64) -> Vec<u32> {
        let band_height = 180.0 / self.n_bands as f64;
        let dec_lo = (dec - radius_deg).max(-90.0);
        let dec_hi = (dec + radius_deg).min(90.0);
        let band_lo = self.band_of(dec_lo);
        let band_hi = self.band_of(dec_hi);

        let mut tiles = Vec::new();
        for band in band_lo..=band_hi {
            let n = self.bins[band as usize];
            let offset = self.offsets[band as usize];

            // Widen the RA window by the shrinking of RA circles at the
            // band edge closest to the pole
            let band_dec_lo = -90.0 + band as f64 * band_height;
            let band_dec_hi = band_dec_lo + band_height;
            let max_abs_dec = band_dec_lo.abs().max(band_dec_hi.abs()).min(89.999);
            let cos_dec = max_abs_dec.to_radians().cos();
            let ra_radius = radius_deg / cos_dec;

            if ra_radius >= 180.0
                || dec_hi >= 90.0 && band == band_hi
                || dec_lo <= -90.0 && band == band_lo
            {
                tiles.extend(offset..offset + n);
                continue;
            }

            let bin_width = 360.0 / n as f64;
            let start = ((ra - ra_radius).rem_euclid(360.0) / bin_width) as u32 % n;
            let span = (2.0 * ra_radius / bin_width).ceil() as u32 + 1;
            if span >= n {
                tiles.extend(offset..offset + n);
            } else {
                for i in 0..span {
                    tiles.push(offset + (start + i) % n);
                }
            }
        }
        tiles
    }
}

fn pack_ra(ra: f64) -> u32 {
    ((ra.rem_euclid(360.0) / 360.0) * u32::MAX as f64) as u32
}

fn unpack_ra(q: u32) -> f64 {
    q as f64 / u32::MAX as f64 * 360.0
}

fn pack_dec(dec: f64) -> u32 {
    (((dec + 90.0) / 180.0).clamp(0.0, 1.0) * u32::MAX as f64) as u32
}

fn unpack_dec(q: u32) -> f64 {
    q as f64 / u32::MAX as f64 * 180.0 - 90.0
}

fn pack_mag(mag: f32) -> u16 {
    (((mag + MAG_OFFSET) * 1000.0).clamp(0.0, u16::MAX as f32)) as u16
}

fn unpack_mag(q: u16) -> f32 {
    q as f32 / 1000.0 - MAG_OFFSET
}

/// Accumulates stars in memory, then writes a tile file.
pub struct TileSetBuilder {
    grid: Grid,
    epoch: f64,
    tiles: Vec<Vec<(u32, u32, u16)>>,
    count: u64,
}

impl TileSetBuilder {
    /// `n_bands` controls tile granularity: 45 bands ≈ 4° tiles (lite),
    /// 90 bands ≈ 2° tiles (standard tier).
    pub fn new(n_bands: u32, epoch: f64) -> Self {
        let grid = Grid::new(n_bands);
        let tiles = vec![Vec::new(); grid.n_tiles() as usize];
        Self {
            grid,
            epoch,
            tiles,
            count: 0,
        }
    }

    pub fn add(&mut self, ra: f64, dec: f64, mag: f32) {
        let tile = self.grid.tile_of(ra, dec);
        self.tiles[tile as usize].push((pack_ra(ra), pack_dec(dec), pack_mag(mag)));
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

        let n_tiles = self.grid.n_tiles() as u64;
        let mut offset = HEADER_SIZE + n_tiles * 12;
        for tile in &mut self.tiles {
            tile.sort_by_key(|&(_, _, mag)| mag);
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&(tile.len() as u32).to_le_bytes())?;
            offset += tile.len() as u64 * RECORD_SIZE;
        }
        for tile in &self.tiles {
            for &(ra, dec, mag) in tile {
                out.write_all(&ra.to_le_bytes())?;
                out.write_all(&dec.to_le_bytes())?;
                out.write_all(&mag.to_le_bytes())?;
            }
        }
        out.flush()
    }
}

/// A read-only, file-backed star catalog.
pub struct TileCatalog {
    file: File,
    grid: Grid,
    epoch: f64,
    star_count: u64,
    index: Vec<(u64, u32)>,
}

impl TileCatalog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut header)?;
        if &header[0..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a seiza star tile file",
            ));
        }
        let n_bands = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let epoch = f64::from_le_bytes(header[12..20].try_into().unwrap());
        let star_count = u64::from_le_bytes(header[20..28].try_into().unwrap());
        if n_bands == 0 || n_bands > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("implausible band count {n_bands}"),
            ));
        }
        let grid = Grid::new(n_bands);

        let n_tiles = grid.n_tiles() as usize;
        let mut raw = vec![0u8; n_tiles * 12];
        file.seek(SeekFrom::Start(HEADER_SIZE))?;
        file.read_exact(&mut raw)?;
        let index = raw
            .chunks_exact(12)
            .map(|chunk| {
                (
                    u64::from_le_bytes(chunk[0..8].try_into().unwrap()),
                    u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                )
            })
            .collect();

        Ok(Self {
            file,
            grid,
            epoch,
            star_count,
            index,
        })
    }

    pub fn star_count(&self) -> u64 {
        self.star_count
    }

    /// Positions were proper-motion corrected to this Julian year.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    fn read_tile(&self, tile: u32) -> io::Result<Vec<CatalogStar>> {
        let (offset, count) = self.index[tile as usize];
        let mut raw = vec![0u8; count as usize * RECORD_SIZE as usize];
        self.file.read_exact_at(&mut raw, offset)?;
        Ok(raw
            .chunks_exact(RECORD_SIZE as usize)
            .map(|r| CatalogStar {
                ra: unpack_ra(u32::from_le_bytes(r[0..4].try_into().unwrap())),
                dec: unpack_dec(u32::from_le_bytes(r[4..8].try_into().unwrap())),
                mag: unpack_mag(u16::from_le_bytes(r[8..10].try_into().unwrap())),
            })
            .collect())
    }
}

impl StarCatalog for TileCatalog {
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        let mut found = Vec::new();
        for tile in self.grid.cone_tiles(ra, dec, radius_deg) {
            let Ok(stars) = self.read_tile(tile) else {
                continue;
            };
            found.extend(
                stars
                    .into_iter()
                    .filter(|s| angular_separation_deg(ra, dec, s.ra, s.dec) <= radius_deg),
            );
        }
        found.sort_by(|a, b| a.mag.total_cmp(&b.mag));
        found.truncate(limit);
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::MemoryCatalog;

    fn pseudo_random_stars(count: usize) -> Vec<CatalogStar> {
        // Deterministic xorshift; uniform-ish over the sphere via
        // dec = asin(2u - 1)
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..count)
            .map(|_| CatalogStar {
                ra: next() * 360.0,
                dec: (2.0 * next() - 1.0).asin().to_degrees(),
                mag: (next() * 12.0) as f32,
            })
            .collect()
    }

    #[test]
    fn quantization_error_is_below_a_milliarcsecond() {
        for &(ra, dec) in &[
            (0.0, -90.0),
            (359.9999, 89.9999),
            (180.0, 0.0),
            (83.63, 22.01),
        ] {
            assert!((unpack_ra(pack_ra(ra)) - ra).abs() * 3600.0 < 0.001);
            assert!((unpack_dec(pack_dec(dec)) - dec).abs() * 3600.0 < 0.001);
        }
        assert!((unpack_mag(pack_mag(11.234)) - 11.234).abs() < 0.001);
        assert!((unpack_mag(pack_mag(-1.46)) - -1.46).abs() < 0.001);
    }

    #[test]
    fn grid_assigns_every_position_to_a_valid_tile() {
        let grid = Grid::new(45);
        for star in pseudo_random_stars(5000) {
            let tile = grid.tile_of(star.ra, star.dec);
            assert!(tile < grid.n_tiles());
        }
        // Poles and RA wrap
        assert!(grid.tile_of(0.0, 90.0) < grid.n_tiles());
        assert!(grid.tile_of(360.0, -90.0) < grid.n_tiles());
        assert!(grid.tile_of(-10.0, 0.0) < grid.n_tiles());
    }

    #[test]
    fn tile_catalog_matches_brute_force_cone_search() {
        let stars = pseudo_random_stars(20000);
        let reference = MemoryCatalog::new(stars.clone());

        let mut builder = TileSetBuilder::new(45, 2025.5);
        for s in &stars {
            builder.add(s.ra, s.dec, s.mag);
        }
        let dir = std::env::temp_dir().join(format!("seiza-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiles.bin");
        builder.write_to(&path).unwrap();

        let catalog = TileCatalog::open(&path).unwrap();
        assert_eq!(catalog.star_count(), 20000);
        assert_eq!(catalog.epoch(), 2025.5);

        for &(ra, dec, radius) in &[
            (10.0, 20.0, 3.0),
            (0.1, -45.0, 5.0),
            (359.9, 0.0, 2.0),   // RA wrap
            (180.0, 88.5, 4.0),  // pole crossing
            (270.0, -89.0, 2.5), // south pole
            (83.6, 22.0, 0.5),
        ] {
            let expected = reference.cone_search(ra, dec, radius, usize::MAX);
            let actual = catalog.cone_search(ra, dec, radius, usize::MAX);
            assert_eq!(
                actual.len(),
                expected.len(),
                "cone ({ra}, {dec}, r={radius}) mismatch"
            );
            // Brightest-first and mag-faithful
            assert!(actual.windows(2).all(|w| w[0].mag <= w[1].mag));
        }

        // Limit applies after brightness sort
        let limited = catalog.cone_search(10.0, 20.0, 10.0, 5);
        assert_eq!(limited.len(), 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("seiza-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garbage.bin");
        std::fs::write(&path, b"definitely not a tile file").unwrap();
        assert!(TileCatalog::open(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
