//! Reference star catalogs.
//!
//! [`tiles::TileCatalog`] is the file-backed store: packed
//! `(ra, dec, magnitude)` records binned into sky tiles,
//! proper-motion-corrected to a fixed epoch at build time.
//! [`MemoryCatalog`] provides the same interface from an in-memory list for
//! tests and synthetic solves.

pub mod tiles;

pub use tiles::{TileCatalog, TileSetBuilder};

/// A reference star position, ICRS degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogStar {
    pub ra: f64,
    pub dec: f64,
    pub mag: f32,
}

/// Source of reference stars around a sky position.
pub trait StarCatalog {
    /// Stars within `radius_deg` of `(ra, dec)`, brightest first, at most
    /// `limit` entries.
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar>;
}

/// In-memory catalog for tests and synthetic solves.
#[derive(Debug, Default, Clone)]
pub struct MemoryCatalog {
    stars: Vec<CatalogStar>,
}

impl MemoryCatalog {
    pub fn new(mut stars: Vec<CatalogStar>) -> Self {
        stars.sort_by(|a, b| a.mag.total_cmp(&b.mag));
        Self { stars }
    }
}

impl StarCatalog for MemoryCatalog {
    fn cone_search(&self, ra: f64, dec: f64, radius_deg: f64, limit: usize) -> Vec<CatalogStar> {
        self.stars
            .iter()
            .filter(|s| angular_separation_deg(ra, dec, s.ra, s.dec) <= radius_deg)
            .take(limit)
            .copied()
            .collect()
    }
}

/// Great-circle separation between two ICRS positions, degrees.
pub fn angular_separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, dec1, ra2, dec2) = (
        ra1.to_radians(),
        dec1.to_radians(),
        ra2.to_radians(),
        dec2.to_radians(),
    );
    // Vincenty formula: stable at small and antipodal separations
    let d_ra = ra2 - ra1;
    let (sin_d1, cos_d1) = dec1.sin_cos();
    let (sin_d2, cos_d2) = dec2.sin_cos();
    let (sin_dra, cos_dra) = d_ra.sin_cos();

    let num =
        ((cos_d2 * sin_dra).powi(2) + (cos_d1 * sin_d2 - sin_d1 * cos_d2 * cos_dra).powi(2)).sqrt();
    let den = sin_d1 * sin_d2 + cos_d1 * cos_d2 * cos_dra;
    num.atan2(den).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separation_basics() {
        assert!((angular_separation_deg(0.0, 0.0, 90.0, 0.0) - 90.0).abs() < 1e-9);
        assert!((angular_separation_deg(10.0, 20.0, 10.0, 20.0)).abs() < 1e-12);
        // one arcsecond in RA at the equator
        let sep = angular_separation_deg(0.0, 0.0, 1.0 / 3600.0, 0.0);
        assert!((sep * 3600.0 - 1.0).abs() < 1e-6);
        // RA circles shrink with declination
        let sep = angular_separation_deg(0.0, 60.0, 1.0, 60.0);
        assert!((sep - 0.5).abs() < 1e-3);
    }

    #[test]
    fn cone_search_filters_and_sorts() {
        let catalog = MemoryCatalog::new(vec![
            CatalogStar {
                ra: 10.0,
                dec: 20.0,
                mag: 5.0,
            },
            CatalogStar {
                ra: 10.1,
                dec: 20.1,
                mag: 3.0,
            },
            CatalogStar {
                ra: 50.0,
                dec: -10.0,
                mag: 1.0,
            },
        ]);
        let found = catalog.cone_search(10.0, 20.0, 1.0, 10);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].mag, 3.0);
    }
}
