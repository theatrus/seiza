//! World coordinate system: TAN (gnomonic) projection with a linear CD
//! matrix, following FITS WCS conventions (degrees, 1-indexed CRPIX is NOT
//! used here — pixel coordinates are 0-indexed image coordinates).

/// A TAN-projection WCS solution.
///
/// `pixel -> world`: intermediate coordinates `(xi, eta) = cd * (p - crpix)`
/// in degrees on the tangent plane, then de-projected around `crval`.
#[derive(Debug, Clone, PartialEq)]
pub struct Wcs {
    /// Sky coordinates of the reference point, degrees (RA, Dec)
    pub crval: (f64, f64),
    /// Pixel coordinates of the reference point (0-indexed)
    pub crpix: (f64, f64),
    /// Linear transform, degrees per pixel: [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    pub cd: [[f64; 2]; 2],
}

impl Wcs {
    /// Convenience constructor from center, scale, rotation, and parity.
    ///
    /// * `center`: sky position (RA, Dec) at pixel `crpix`, degrees
    /// * `scale_arcsec_px`: pixel scale in arcseconds per pixel
    /// * `rotation_deg`: position angle of north in the image, degrees E of N
    /// * `flipped`: true when the image parity is mirrored
    pub fn from_center_scale_rotation(
        center: (f64, f64),
        crpix: (f64, f64),
        scale_arcsec_px: f64,
        rotation_deg: f64,
        flipped: bool,
    ) -> Self {
        let s = scale_arcsec_px / 3600.0;
        let r = rotation_deg.to_radians();
        let (sin_r, cos_r) = r.sin_cos();
        let parity = if flipped { -1.0 } else { 1.0 };
        // Standard convention: xi increases to the east (negative RA axis
        // handled inside the projection), eta to the north.
        let cd = [
            [-s * parity * cos_r, s * sin_r],
            [-s * parity * sin_r, -s * cos_r],
        ];
        Self {
            crval: center,
            crpix,
            cd,
        }
    }

    /// Pixel scale in arcseconds per pixel (geometric mean of the two axes).
    pub fn scale_arcsec_per_px(&self) -> f64 {
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        det.abs().sqrt() * 3600.0
    }

    /// Map a pixel coordinate to sky coordinates (RA, Dec) in degrees.
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.crpix.0;
        let dy = y - self.crpix.1;
        let xi = (self.cd[0][0] * dx + self.cd[0][1] * dy).to_radians();
        let eta = (self.cd[1][0] * dx + self.cd[1][1] * dy).to_radians();

        let (ra0, dec0) = (self.crval.0.to_radians(), self.crval.1.to_radians());
        let (sin_d0, cos_d0) = dec0.sin_cos();

        let rho = (xi * xi + eta * eta).sqrt();
        if rho == 0.0 {
            return self.crval;
        }
        let c = rho.atan();
        let (sin_c, cos_c) = c.sin_cos();

        let dec = (cos_c * sin_d0 + eta * sin_c * cos_d0 / rho).asin();
        let ra = ra0 + (xi * sin_c).atan2(rho * cos_d0 * cos_c - eta * sin_d0 * sin_c);

        let mut ra_deg = ra.to_degrees() % 360.0;
        if ra_deg < 0.0 {
            ra_deg += 360.0;
        }
        (ra_deg, dec.to_degrees())
    }

    /// Map sky coordinates (RA, Dec, degrees) to a pixel coordinate.
    /// Returns `None` for points on or behind the tangent-plane horizon.
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let (ra0, dec0) = (self.crval.0.to_radians(), self.crval.1.to_radians());
        let (ra, dec) = (ra.to_radians(), dec.to_radians());
        let (sin_d0, cos_d0) = dec0.sin_cos();
        let (sin_d, cos_d) = dec.sin_cos();
        let dra = ra - ra0;

        let cos_c = sin_d0 * sin_d + cos_d0 * cos_d * dra.cos();
        if cos_c <= 1e-9 {
            return None;
        }
        let xi = (cos_d * dra.sin() / cos_c).to_degrees();
        let eta = ((cos_d0 * sin_d - sin_d0 * cos_d * dra.cos()) / cos_c).to_degrees();

        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        if det == 0.0 {
            return None;
        }
        let dx = (self.cd[1][1] * xi - self.cd[0][1] * eta) / det;
        let dy = (-self.cd[1][0] * xi + self.cd[0][0] * eta) / det;
        Some((self.crpix.0 + dx, self.crpix.1 + dy))
    }

    /// Sky footprint of an image of the given dimensions: the RA/Dec of the
    /// four corners, clockwise from (0, 0).
    pub fn footprint(&self, width: u32, height: u32) -> [(f64, f64); 4] {
        let (w, h) = (width as f64 - 1.0, height as f64 - 1.0);
        [
            self.pixel_to_world(0.0, 0.0),
            self.pixel_to_world(w, 0.0),
            self.pixel_to_world(w, h),
            self.pixel_to_world(0.0, h),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} != {b} (tol {tol})");
    }

    #[test]
    fn reference_point_maps_to_crval() {
        let wcs = Wcs::from_center_scale_rotation((83.63, 22.01), (100.0, 200.0), 1.5, 0.0, false);
        let (ra, dec) = wcs.pixel_to_world(100.0, 200.0);
        assert_close(ra, 83.63, 1e-9);
        assert_close(dec, 22.01, 1e-9);
        let (x, y) = wcs.world_to_pixel(83.63, 22.01).unwrap();
        assert_close(x, 100.0, 1e-6);
        assert_close(y, 200.0, 1e-6);
    }

    #[test]
    fn round_trips_across_the_frame() {
        for rotation in [0.0, 33.5, 180.0, 271.25] {
            for flipped in [false, true] {
                let wcs = Wcs::from_center_scale_rotation(
                    (10.68, 41.27),
                    (2000.0, 1500.0),
                    0.73,
                    rotation,
                    flipped,
                );
                for (x, y) in [(0.0, 0.0), (4000.0, 0.0), (123.4, 2987.6), (2000.0, 1500.0)] {
                    let (ra, dec) = wcs.pixel_to_world(x, y);
                    let (x2, y2) = wcs.world_to_pixel(ra, dec).unwrap();
                    assert_close(x2, x, 1e-6);
                    assert_close(y2, y, 1e-6);
                }
            }
        }
    }

    #[test]
    fn round_trips_near_the_pole() {
        let wcs = Wcs::from_center_scale_rotation((37.95, 89.26), (500.0, 500.0), 2.0, 45.0, false);
        let (ra, dec) = wcs.pixel_to_world(0.0, 0.0);
        let (x, y) = wcs.world_to_pixel(ra, dec).unwrap();
        assert_close(x, 0.0, 1e-6);
        assert_close(y, 0.0, 1e-6);
        // ~1000 px diagonal at 2"/px stays within a degree of the pole center
        assert!((dec - 89.26).abs() < 1.0);
    }

    #[test]
    fn scale_is_recovered_from_cd() {
        let wcs = Wcs::from_center_scale_rotation((180.0, 0.0), (0.0, 0.0), 1.23, 77.0, true);
        assert_close(wcs.scale_arcsec_per_px(), 1.23, 1e-9);
    }

    #[test]
    fn scale_matches_angular_separation() {
        let wcs = Wcs::from_center_scale_rotation((180.0, 20.0), (0.0, 0.0), 2.0, 0.0, false);
        let (ra1, dec1) = wcs.pixel_to_world(0.0, 0.0);
        let (ra2, dec2) = wcs.pixel_to_world(1.0, 0.0);
        // one pixel apart => ~2 arcsec on the sky
        let d = angular_separation_deg(ra1, dec1, ra2, dec2) * 3600.0;
        assert_close(d, 2.0, 1e-3);
    }

    #[test]
    fn behind_horizon_is_none() {
        let wcs = Wcs::from_center_scale_rotation((0.0, 0.0), (0.0, 0.0), 1.0, 0.0, false);
        assert!(wcs.world_to_pixel(180.0, 0.0).is_none());
    }

    fn angular_separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
        let (ra1, dec1, ra2, dec2) = (
            ra1.to_radians(),
            dec1.to_radians(),
            ra2.to_radians(),
            dec2.to_radians(),
        );
        let s = (dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * (ra1 - ra2).cos())
            .clamp(-1.0, 1.0);
        s.acos().to_degrees()
    }
}
