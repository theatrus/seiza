//! World coordinate system: TAN (gnomonic) projection with a linear CD
//! matrix, following FITS WCS conventions (degrees, 1-indexed CRPIX is NOT
//! used here — pixel coordinates are 0-indexed image coordinates).

/// SIP polynomial distortion terms (Shupe et al. 2005).
///
/// Forward model: with `u = x - crpix.0` and `v = y - crpix.1`,
/// `(xi, eta) = cd * (u + f(u, v), v + g(u, v))` where
/// `f(u, v) = sum A_pq u^p v^q` over [`Sip::forward_terms`] and `g` uses the
/// `B` coefficients. The inverse polynomials `AP`/`BP` approximate the
/// reverse mapping over [`Sip::inverse_terms`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sip {
    /// Polynomial order (2..=5); forward terms have `2 <= p + q <= order`.
    pub order: u8,
    /// `A_p_q` coefficients in [`Sip::forward_terms`] order.
    pub a: Vec<f64>,
    /// `B_p_q` coefficients in [`Sip::forward_terms`] order.
    pub b: Vec<f64>,
    /// `AP_p_q` inverse coefficients in [`Sip::inverse_terms`] order.
    pub ap: Vec<f64>,
    /// `BP_p_q` inverse coefficients in [`Sip::inverse_terms`] order.
    pub bp: Vec<f64>,
}

impl Sip {
    /// `(p, q)` exponent pairs for the forward `A`/`B` polynomials:
    /// `2 <= p + q <= order`, ascending in `p` then `q`.
    pub fn forward_terms(order: u8) -> Vec<(u8, u8)> {
        Self::terms(order, 2)
    }

    /// `(p, q)` exponent pairs for the inverse `AP`/`BP` polynomials:
    /// `0 <= p + q <= order`, ascending in `p` then `q`. The inverse
    /// deliberately includes constant and linear terms — the inverse of an
    /// identity-plus-polynomial is not itself identity-plus-polynomial.
    pub fn inverse_terms(order: u8) -> Vec<(u8, u8)> {
        Self::terms(order, 0)
    }

    fn terms(order: u8, min_total: u8) -> Vec<(u8, u8)> {
        let mut terms = Vec::new();
        for p in 0..=order {
            for q in 0..=order.saturating_sub(p) {
                if p + q >= min_total {
                    terms.push((p, q));
                }
            }
        }
        terms
    }

    /// `(f(u, v), g(u, v))`: the forward distortion correction in pixels.
    pub fn forward(&self, u: f64, v: f64) -> (f64, f64) {
        Self::eval(&self.a, &self.b, Self::forward_terms(self.order), u, v)
    }

    /// `(F(U, V), G(U, V))`: the inverse correction in pixels.
    pub fn inverse(&self, u: f64, v: f64) -> (f64, f64) {
        Self::eval(&self.ap, &self.bp, Self::inverse_terms(self.order), u, v)
    }

    fn eval(a: &[f64], b: &[f64], terms: Vec<(u8, u8)>, u: f64, v: f64) -> (f64, f64) {
        let mut f = 0.0;
        let mut g = 0.0;
        for (index, (p, q)) in terms.iter().enumerate() {
            let monomial = u.powi(*p as i32) * v.powi(*q as i32);
            f += a.get(index).copied().unwrap_or(0.0) * monomial;
            g += b.get(index).copied().unwrap_or(0.0) * monomial;
        }
        (f, g)
    }
}

/// A TAN-projection WCS solution.
///
/// `pixel -> world`: intermediate coordinates `(xi, eta) = cd * (p - crpix)`
/// in degrees on the tangent plane, then de-projected around `crval`. When
/// `sip` is present the SIP forward polynomial corrects `(p - crpix)` first.
#[derive(Debug, Clone, PartialEq)]
pub struct Wcs {
    /// Sky coordinates of the reference point, degrees (RA, Dec)
    pub crval: (f64, f64),
    /// Pixel coordinates of the reference point (0-indexed)
    pub crpix: (f64, f64),
    /// Linear transform, degrees per pixel: [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    pub cd: [[f64; 2]; 2],
    /// Optional SIP distortion polynomials.
    pub sip: Option<Sip>,
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
            sip: None,
        }
    }

    /// Pixel scale in arcseconds per pixel (geometric mean of the two axes).
    pub fn scale_arcsec_per_px(&self) -> f64 {
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        det.abs().sqrt() * 3600.0
    }

    /// Map a pixel coordinate to sky coordinates (RA, Dec) in degrees.
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let mut dx = x - self.crpix.0;
        let mut dy = y - self.crpix.1;
        if let Some(sip) = &self.sip {
            let (f, g) = sip.forward(dx, dy);
            dx += f;
            dy += g;
        }
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
        let mut dx = (self.cd[1][1] * xi - self.cd[0][1] * eta) / det;
        let mut dy = (-self.cd[1][0] * xi + self.cd[0][0] * eta) / det;
        if let Some(sip) = &self.sip {
            let (f, g) = sip.inverse(dx, dy);
            dx += f;
            dy += g;
        }
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
    fn sip_forward_terms_follow_the_documented_order() {
        assert_eq!(Sip::forward_terms(2), vec![(0, 2), (1, 1), (2, 0)]);
        assert_eq!(
            Sip::forward_terms(3),
            vec![(0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]
        );
        assert_eq!(Sip::inverse_terms(2).len(), 6);
        assert_eq!(Sip::inverse_terms(2)[0], (0, 0));
    }

    #[test]
    fn sip_distortion_shifts_pixels_and_round_trips() {
        let mut wcs =
            Wcs::from_center_scale_rotation((150.0, 35.0), (1000.0, 800.0), 2.0, 15.0, false);
        let undistorted = wcs.pixel_to_world(200.0, 300.0);
        // A pure quadratic barrel-like term. The inverse coefficients are a
        // first-order approximation: -A for the same terms, plus exact
        // agreement is verified only to the tolerance such a small
        // distortion permits.
        let a = 1e-6;
        wcs.sip = Some(Sip {
            order: 2,
            a: vec![a, 0.0, a],
            b: vec![0.0, a, 0.0],
            ap: vec![0.0, 0.0, -a, 0.0, 0.0, -a],
            bp: vec![0.0, 0.0, 0.0, 0.0, -a, 0.0],
        });
        let distorted = wcs.pixel_to_world(200.0, 300.0);
        // u = -800, v = -500: f = a(v^2 + u^2) = 0.889 px at 2"/px
        let separation =
            angular_separation_deg(undistorted.0, undistorted.1, distorted.0, distorted.1) * 3600.0;
        assert!((1.0..3.0).contains(&separation), "{separation}");

        let (x, y) = wcs.world_to_pixel(distorted.0, distorted.1).unwrap();
        // The hand-written inverse is approximate; the round trip must land
        // within a small fraction of the applied distortion.
        assert!((x - 200.0).abs() < 0.01, "{x}");
        assert!((y - 300.0).abs() < 0.01, "{y}");
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
