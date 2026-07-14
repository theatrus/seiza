//! File-backed star catalog tiles.
//!
//! Current format `SEIZAST2` (little-endian), designed for memory-mapped,
//! vectorizable scans:
//!
//! ```text
//! magic        [u8; 8]  = b"SEIZAST2"
//! n_bands      u32          declination bands from -90° to +90°
//! epoch        f64          positions are proper-motion corrected to this year
//! star_count   u64
//! attribution  u16 len + UTF-8 bytes (source + license note)
//! padding      to an 8-byte boundary
//! index        n_tiles ×  { offset: u64, count: u32 }
//! data         per tile, columnar: ra[u32; n]  dec[u32; n]  mag[u16; n]
//!              (each tile padded to a 4-byte boundary)
//! ```
//!
//! Records within a tile are sorted brightest-first. Columnar (SoA) layout
//! keeps each field contiguous so decode loops auto-vectorize, and the file
//! is memory-mapped so only touched tiles are paged in.
//!
//! The legacy `SEIZAST1` interleaved format is still readable.
//!
//! The sky is split into `n_bands` equal-height declination bands; each band
//! is split into RA bins whose count shrinks with `cos(dec)` so bins stay
//! roughly equal-area. Quantization: RA as u32 over 360°, Dec as u32 over
//! 180° (sub-milliarcsecond), magnitude as u16 millimags offset by +3.

use super::{CatalogStar, StarCatalog, angular_separation_deg};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const MAGIC_V1: &[u8; 8] = b"SEIZAST1";
const MAGIC_V2: &[u8; 8] = b"SEIZAST2";
const V1_RECORD_SIZE: usize = 10;
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

/// Accumulates stars in memory, then writes a `SEIZAST2` tile file.
pub struct TileSetBuilder {
    grid: Grid,
    epoch: f64,
    attribution: String,
    tiles: Vec<Vec<(u32, u32, u16)>>,
    count: u64,
}

impl TileSetBuilder {
    /// `n_bands` controls tile granularity: 45 bands ≈ 4° tiles (lite),
    /// 180 bands ≈ 1° tiles (deep tiers). `attribution` records the data
    /// source and license inside the file.
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
        out.write_all(MAGIC_V2)?;
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
            tile.sort_by_key(|&(_, _, mag)| mag);
            out.write_all(&offset.to_le_bytes())?;
            out.write_all(&(tile.len() as u32).to_le_bytes())?;
            let data = tile.len() as u64 * 10;
            offset += data.next_multiple_of(4);
        }
        for tile in &self.tiles {
            // Columnar: all RA, then all Dec, then all magnitudes
            for &(ra, _, _) in tile {
                out.write_all(&ra.to_le_bytes())?;
            }
            for &(_, dec, _) in tile {
                out.write_all(&dec.to_le_bytes())?;
            }
            for &(_, _, mag) in tile {
                out.write_all(&mag.to_le_bytes())?;
            }
            let data = tile.len() as u64 * 10;
            let pad = data.next_multiple_of(4) - data;
            out.write_all(&vec![0u8; pad as usize])?;
        }
        out.flush()
    }
}

enum Layout {
    V1Interleaved,
    V2Columnar,
}

/// A read-only, memory-mapped star catalog.
pub struct TileCatalog {
    map: memmap2::Mmap,
    layout: Layout,
    grid: Grid,
    epoch: f64,
    star_count: u64,
    attribution: String,
    index: Vec<(u64, u32)>,
    data_offset: usize,
}

impl TileCatalog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is opened read-only; concurrent truncation would
        // fault, which is acceptable for locally-managed data files.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too short for a seiza star tile file",
            ));
        }

        let layout = match &map[0..8] {
            m if m == MAGIC_V1 => Layout::V1Interleaved,
            m if m == MAGIC_V2 => Layout::V2Columnar,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "not a seiza star tile file",
                ));
            }
        };

        let n_bands = u32::from_le_bytes(map[8..12].try_into().unwrap());
        let epoch = f64::from_le_bytes(map[12..20].try_into().unwrap());
        let star_count = u64::from_le_bytes(map[20..28].try_into().unwrap());
        if n_bands == 0 || n_bands > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("implausible band count {n_bands}"),
            ));
        }
        let grid = Grid::new(n_bands);

        let (attribution, index_start) = match layout {
            Layout::V1Interleaved => (String::new(), 28usize),
            Layout::V2Columnar => {
                if map.len() < 30 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated header",
                    ));
                }
                let len = u16::from_le_bytes(map[28..30].try_into().unwrap()) as usize;
                let end = 30 + len;
                if map.len() < end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated header",
                    ));
                }
                let attribution = String::from_utf8_lossy(&map[30..end]).into_owned();
                (attribution, end.next_multiple_of(8))
            }
        };

        let n_tiles = grid.n_tiles() as usize;
        let index_end = index_start + n_tiles * 12;
        if map.len() < index_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated index",
            ));
        }
        let index = map[index_start..index_end]
            .chunks_exact(12)
            .map(|chunk| {
                (
                    u64::from_le_bytes(chunk[0..8].try_into().unwrap()),
                    u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                )
            })
            .collect();

        Ok(Self {
            map,
            layout,
            grid,
            epoch,
            star_count,
            attribution,
            index,
            data_offset: index_end,
        })
    }

    /// Exhaustively validate tile ranges, total counts, and the documented
    /// brightest-first ordering. Normal [`Self::open`] does not scan star
    /// records; this method intentionally touches the complete mapped data.
    pub fn validate(&self) -> io::Result<()> {
        if !self.epoch.is_finite() {
            return Err(invalid_data("star tile epoch is not finite"));
        }
        let mut total = 0u64;
        let mut previous_end = self.data_offset;
        for &(offset, count) in &self.index {
            let start = usize::try_from(offset)
                .map_err(|_| invalid_data("star tile offset does not fit this platform"))?;
            let count = count as usize;
            let bytes = count
                .checked_mul(V1_RECORD_SIZE)
                .ok_or_else(|| invalid_data("star tile byte count overflows"))?;
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| invalid_data("star tile range overflows"))?;
            if start < previous_end
                || start > self.map.len()
                || end > self.map.len()
                || start - previous_end > 3
                || self.map[previous_end..start].iter().any(|byte| *byte != 0)
            {
                return Err(invalid_data(
                    "star tile range or alignment padding is invalid",
                ));
            }
            total = total
                .checked_add(count as u64)
                .ok_or_else(|| invalid_data("star tile count overflows"))?;

            let mut previous_mag = None;
            for index in 0..count {
                let mag_offset = match self.layout {
                    Layout::V1Interleaved => start + index * V1_RECORD_SIZE + 8,
                    Layout::V2Columnar => start + count * 8 + index * 2,
                };
                let magnitude =
                    u16::from_le_bytes(self.map[mag_offset..mag_offset + 2].try_into().unwrap());
                if previous_mag.is_some_and(|previous| previous > magnitude) {
                    return Err(invalid_data("stars within a tile are not magnitude-sorted"));
                }
                previous_mag = Some(magnitude);
            }
            previous_end = end;
        }
        if self.map.len() - previous_end > 3
            || self.map[previous_end..].iter().any(|byte| *byte != 0)
        {
            return Err(invalid_data("star tile has invalid trailing padding"));
        }
        if total != self.star_count {
            return Err(invalid_data(
                "star tile counts do not match the catalog header",
            ));
        }
        Ok(())
    }

    pub fn star_count(&self) -> u64 {
        self.star_count
    }

    /// Positions were proper-motion corrected to this Julian year.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    /// Data source and license note embedded in the file (empty for v1).
    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// Visit every star in a tile. Decoding runs over contiguous columns in
    /// the mapped file for the v2 layout.
    /// Visit stars in a tile until the callback returns false. Records are
    /// brightest-first, enabling early exit at a magnitude limit.
    fn for_each_in_tile_while(&self, tile: u32, mut visit: impl FnMut(CatalogStar) -> bool) {
        let (offset, count) = self.index[tile as usize];
        let (offset, count) = (offset as usize, count as usize);
        match self.layout {
            Layout::V1Interleaved => {
                let end = offset + count * V1_RECORD_SIZE;
                let Some(data) = self.map.get(offset..end) else {
                    return;
                };
                for r in data.chunks_exact(V1_RECORD_SIZE) {
                    let keep_going = visit(CatalogStar {
                        ra: unpack_ra(u32::from_le_bytes(r[0..4].try_into().unwrap())),
                        dec: unpack_dec(u32::from_le_bytes(r[4..8].try_into().unwrap())),
                        mag: unpack_mag(u16::from_le_bytes(r[8..10].try_into().unwrap())),
                    });
                    if !keep_going {
                        return;
                    }
                }
            }
            Layout::V2Columnar => {
                let ra_end = offset + count * 4;
                let dec_end = ra_end + count * 4;
                let mag_end = dec_end + count * 2;
                let Some(_) = self.map.get(offset..mag_end) else {
                    return;
                };
                let ra = &self.map[offset..ra_end];
                let dec = &self.map[ra_end..dec_end];
                let mag = &self.map[dec_end..mag_end];
                for i in 0..count {
                    visit(CatalogStar {
                        ra: unpack_ra(u32::from_le_bytes(ra[i * 4..i * 4 + 4].try_into().unwrap())),
                        dec: unpack_dec(u32::from_le_bytes(
                            dec[i * 4..i * 4 + 4].try_into().unwrap(),
                        )),
                        mag: unpack_mag(u16::from_le_bytes(
                            mag[i * 2..i * 2 + 2].try_into().unwrap(),
                        )),
                    });
                }
            }
        }
    }
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl StarCatalog for TileCatalog {
    fn star_count(&self) -> u64 {
        self.star_count()
    }

    fn all_brighter_than(&self, mag_limit: f32) -> Vec<CatalogStar> {
        // Tiles store brightest-first: stop at the first fainter record
        let mut found = Vec::new();
        for tile in 0..self.grid.n_tiles() {
            self.for_each_in_tile_while(tile, |star| {
                if star.mag <= mag_limit {
                    found.push(star);
                    true
                } else {
                    false
                }
            });
        }
        found
    }

    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        // Keep the brightest `limit` with a bounded max-heap. Tiles store
        // brightest-first, so once the heap is full each tile's scan stops
        // at the first record fainter than the faintest kept star — on a
        // deep catalog a wide cone would otherwise collect (and sort)
        // millions of faint stars just to keep a few hundred.
        struct ByMag(CatalogStar);
        impl PartialEq for ByMag {
            fn eq(&self, other: &Self) -> bool {
                self.0.mag.total_cmp(&other.0.mag).is_eq()
            }
        }
        impl Eq for ByMag {}
        impl PartialOrd for ByMag {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for ByMag {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.mag.total_cmp(&other.0.mag)
            }
        }
        let mut heap: std::collections::BinaryHeap<ByMag> =
            std::collections::BinaryHeap::with_capacity(limit.min(8192) + 1);
        for tile in self.grid.cone_tiles(ra, dec, radius_deg) {
            self.for_each_in_tile_while(tile, |star| {
                if heap.len() == limit && star.mag.total_cmp(&heap.peek().unwrap().0.mag).is_ge() {
                    return false;
                }
                if angular_separation_deg(ra, dec, star.ra, star.dec) <= radius_deg {
                    heap.push(ByMag(star));
                    if heap.len() > limit {
                        heap.pop();
                    }
                }
                true
            });
        }
        let mut found: Vec<CatalogStar> = heap.into_iter().map(|s| s.0).collect();
        found.sort_by(|a, b| a.mag.total_cmp(&b.mag));
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

    /// Write a legacy SEIZAST1 interleaved file for reader-compat testing.
    fn write_v1(stars: &[CatalogStar], n_bands: u32, epoch: f64, path: &Path) {
        let grid = Grid::new(n_bands);
        let mut tiles: Vec<Vec<(u32, u32, u16)>> = vec![Vec::new(); grid.n_tiles() as usize];
        for s in stars {
            tiles[grid.tile_of(s.ra, s.dec) as usize].push((
                pack_ra(s.ra),
                pack_dec(s.dec),
                pack_mag(s.mag),
            ));
        }
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC_V1);
        out.extend_from_slice(&n_bands.to_le_bytes());
        out.extend_from_slice(&epoch.to_le_bytes());
        out.extend_from_slice(&(stars.len() as u64).to_le_bytes());
        let mut offset = 28u64 + grid.n_tiles() as u64 * 12;
        for tile in &mut tiles {
            tile.sort_by_key(|&(_, _, mag)| mag);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(tile.len() as u32).to_le_bytes());
            offset += tile.len() as u64 * V1_RECORD_SIZE as u64;
        }
        for tile in &tiles {
            for &(ra, dec, mag) in tile {
                out.extend_from_slice(&ra.to_le_bytes());
                out.extend_from_slice(&dec.to_le_bytes());
                out.extend_from_slice(&mag.to_le_bytes());
            }
        }
        std::fs::write(path, out).unwrap();
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
    fn v2_catalog_matches_brute_force_cone_search() {
        let stars = pseudo_random_stars(20000);
        let reference = MemoryCatalog::new(stars.clone());

        let mut builder = TileSetBuilder::new(45, 2025.5, "test data (c) nobody");
        for s in &stars {
            builder.add(s.ra, s.dec, s.mag);
        }
        let dir = std::env::temp_dir().join(format!("seiza-test-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiles.bin");
        builder.write_to(&path).unwrap();

        let catalog = TileCatalog::open(&path).unwrap();
        catalog.validate().unwrap();
        assert_eq!(catalog.star_count(), 20000);
        assert_eq!(catalog.epoch(), 2025.5);
        assert_eq!(catalog.attribution(), "test data (c) nobody");

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
            assert!(actual.windows(2).all(|w| w[0].mag <= w[1].mag));
        }

        let limited = catalog.cone_search(10.0, 20.0, 10.0, 5);
        assert_eq!(limited.len(), 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reads_legacy_v1_files() {
        let stars = pseudo_random_stars(5000);
        let reference = MemoryCatalog::new(stars.clone());
        let dir = std::env::temp_dir().join(format!("seiza-test-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiles-v1.bin");
        write_v1(&stars, 45, 2025.0, &path);

        let catalog = TileCatalog::open(&path).unwrap();
        catalog.validate().unwrap();
        assert_eq!(catalog.star_count(), 5000);
        assert_eq!(catalog.attribution(), "");
        for &(ra, dec, radius) in &[(10.0, 20.0, 4.0), (300.0, 35.0, 2.0), (0.0, -88.0, 3.0)] {
            let expected = reference.cone_search(ra, dec, radius, usize::MAX);
            let actual = catalog.cone_search(ra, dec, radius, usize::MAX);
            assert_eq!(actual.len(), expected.len());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("seiza-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garbage.bin");
        std::fs::write(
            &path,
            b"definitely not a tile file with some length padding",
        )
        .unwrap();
        assert!(TileCatalog::open(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_defers_exhaustive_tile_validation() {
        let dir = std::env::temp_dir().join(format!("seiza-test-lazy-tile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiles.bin");
        let mut builder = TileSetBuilder::new(2, 2025.5, "test");
        builder.add(10.0, 20.0, 5.0);
        builder.write_to(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let attribution_len = u16::from_le_bytes(bytes[28..30].try_into().unwrap()) as usize;
        let index_offset = (30 + attribution_len).next_multiple_of(8);
        bytes[index_offset + 8..index_offset + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let catalog = TileCatalog::open(&path).unwrap();
        assert!(catalog.validate().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
