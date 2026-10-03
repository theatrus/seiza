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
        Self::eval(&self.a, &self.b, self.order, 2, u, v)
    }

    /// `(F(U, V), G(U, V))`: the inverse correction in pixels.
    pub fn inverse(&self, u: f64, v: f64) -> (f64, f64) {
        Self::eval(&self.ap, &self.bp, self.order, 0, u, v)
    }

    /// Allocation-free evaluation: transforms run per pixel in overlay and
    /// fitting loops. Term visit order matches [`Sip::terms`] exactly.
    fn eval(a: &[f64], b: &[f64], order: u8, min_total: u8, u: f64, v: f64) -> (f64, f64) {
        let mut f = 0.0;
        let mut g = 0.0;
        let mut index = 0;
        for p in 0..=order {
            for q in 0..=order.saturating_sub(p) {
                if p + q >= min_total {
                    let monomial = u.powi(i32::from(p)) * v.powi(i32::from(q));
                    f += a.get(index).copied().unwrap_or(0.0) * monomial;
                    g += b.get(index).copied().unwrap_or(0.0) * monomial;
                    index += 1;
                }
            }
        }
        (f, g)
    }
}

/// One FITS header card value emitted by [`Wcs::fits_header_cards`].
#[derive(Debug, Clone, PartialEq)]
pub enum FitsCardValue {
    Text(&'static str),
    Integer(u8),
    Number(f64),
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

    /// FITS WCS keywords describing this solution: 1-indexed `CRPIX`, TAN
    /// or TAN-SIP `CTYPE`, the CD matrix, and the complete
    /// `A_p_q`/`B_p_q`/`AP_p_q`/`BP_p_q` set when distortion is present.
    /// Shared by every serializer (FITS header text, `.wcs` files, Python
    /// dicts) so the keyword contract has one implementation.
    pub fn fits_header_cards(&self) -> Vec<(String, FitsCardValue)> {
        use FitsCardValue::{Integer, Number, Text};
        let sip = self.sip.as_ref();
        let mut cards = vec![
            (
                "CTYPE1".into(),
                Text(if sip.is_some() {
                    "RA---TAN-SIP"
                } else {
                    "RA---TAN"
                }),
            ),
            (
                "CTYPE2".into(),
                Text(if sip.is_some() {
                    "DEC--TAN-SIP"
                } else {
                    "DEC--TAN"
                }),
            ),
            ("CUNIT1".into(), Text("deg")),
            ("CUNIT2".into(), Text("deg")),
            ("EQUINOX".into(), Number(2000.0)),
            ("CRVAL1".into(), Number(self.crval.0)),
            ("CRVAL2".into(), Number(self.crval.1)),
            ("CRPIX1".into(), Number(self.crpix.0 + 1.0)),
            ("CRPIX2".into(), Number(self.crpix.1 + 1.0)),
            ("CD1_1".into(), Number(self.cd[0][0])),
            ("CD1_2".into(), Number(self.cd[0][1])),
            ("CD2_1".into(), Number(self.cd[1][0])),
            ("CD2_2".into(), Number(self.cd[1][1])),
        ];
        if let Some(sip) = sip {
            cards.push(("A_ORDER".into(), Integer(sip.order)));
            cards.push(("B_ORDER".into(), Integer(sip.order)));
            for (prefix, terms, values) in [
                ("A", Sip::forward_terms(sip.order), &sip.a),
                ("B", Sip::forward_terms(sip.order), &sip.b),
            ] {
                for ((p, q), value) in terms.iter().zip(values) {
                    cards.push((format!("{prefix}_{p}_{q}"), Number(*value)));
                }
            }
            cards.push(("AP_ORDER".into(), Integer(sip.order)));
            cards.push(("BP_ORDER".into(), Integer(sip.order)));
            for (prefix, terms, values) in [
                ("AP", Sip::inverse_terms(sip.order), &sip.ap),
                ("BP", Sip::inverse_terms(sip.order), &sip.bp),
            ] {
                for ((p, q), value) in terms.iter().zip(values) {
                    cards.push((format!("{prefix}_{p}_{q}"), Number(*value)));
                }
            }
        }
        cards
    }

    /// Read a TAN or TAN-SIP solution from FITS WCS keywords, the inverse of
    /// [`Self::fits_header_cards`]. `number` looks up a numeric card and
    /// `text` a string card by keyword.
    ///
    /// The linear part may be a `CD` matrix, or `CDELT` with a `PC` matrix
    /// or a `CROTA2` angle. `None` when the projection is not TAN, a required
    /// card is missing, or the matrix is singular. SIP terms of the forward
    /// and inverse polynomials are read up to their own `*_ORDER`; a header
    /// with forward terms but no inverse ones maps world to pixel without
    /// the distortion correction.
    pub fn from_fits_values(
        number: impl Fn(&str) -> Option<f64>,
        text: impl Fn(&str) -> Option<String>,
    ) -> Option<Self> {
        let ctype1 = text("CTYPE1")?;
        let ctype2 = text("CTYPE2")?;
        if !ctype1.trim().starts_with("RA---TAN") || !ctype2.trim().starts_with("DEC--TAN") {
            return None;
        }
        let crval = (number("CRVAL1")?, number("CRVAL2")?);
        let crpix = (number("CRPIX1")? - 1.0, number("CRPIX2")? - 1.0);
        let cd = match (
            number("CD1_1"),
            number("CD1_2"),
            number("CD2_1"),
            number("CD2_2"),
        ) {
            (None, None, None, None) => {
                let (cdelt1, cdelt2) = (number("CDELT1")?, number("CDELT2")?);
                let pc = match (
                    number("PC1_1"),
                    number("PC1_2"),
                    number("PC2_1"),
                    number("PC2_2"),
                ) {
                    (None, None, None, None) => {
                        let (sin, cos) = number("CROTA2").unwrap_or(0.0).to_radians().sin_cos();
                        [[cos, -sin * cdelt2 / cdelt1], [sin * cdelt1 / cdelt2, cos]]
                    }
                    (pc11, pc12, pc21, pc22) => [
                        [pc11.unwrap_or(1.0), pc12.unwrap_or(0.0)],
                        [pc21.unwrap_or(0.0), pc22.unwrap_or(1.0)],
                    ],
                };
                [
                    [cdelt1 * pc[0][0], cdelt1 * pc[0][1]],
                    [cdelt2 * pc[1][0], cdelt2 * pc[1][1]],
                ]
            }
            (cd11, cd12, cd21, cd22) => [
                [cd11.unwrap_or(0.0), cd12.unwrap_or(0.0)],
                [cd21.unwrap_or(0.0), cd22.unwrap_or(0.0)],
            ],
        };
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        if !det.is_finite() || det == 0.0 || !crval.0.is_finite() || !crval.1.is_finite() {
            return None;
        }
        let order_of = |key: &str| {
            number(key)
                .filter(|order| order.is_finite() && *order >= 0.0 && *order <= 9.0)
                .map(|order| order as u8)
        };
        let forward_order = ctype1
            .contains("-SIP")
            .then(|| order_of("A_ORDER").max(order_of("B_ORDER")))
            .flatten()
            .unwrap_or(0);
        let sip = (forward_order >= 2).then(|| {
            let inverse_order = order_of("AP_ORDER").max(order_of("BP_ORDER")).unwrap_or(0);
            let order = forward_order.max(inverse_order);
            let read = |prefix: &str, terms: Vec<(u8, u8)>, limit: u8| {
                terms
                    .into_iter()
                    .map(|(p, q)| {
                        if p + q > limit {
                            0.0
                        } else {
                            number(&format!("{prefix}_{p}_{q}")).unwrap_or(0.0)
                        }
                    })
                    .collect::<Vec<_>>()
            };
            Sip {
                order,
                a: read("A", Sip::forward_terms(order), forward_order),
                b: read("B", Sip::forward_terms(order), forward_order),
                ap: read("AP", Sip::inverse_terms(order), inverse_order),
                bp: read("BP", Sip::inverse_terms(order), inverse_order),
            }
        });
        Some(Self {
            crval,
            crpix,
            cd,
            sip,
        })
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
    fn header_cards_read_back_into_the_same_solution() {
        let mut wcs =
            Wcs::from_center_scale_rotation((150.0, 35.0), (1000.0, 800.0), 2.0, 15.0, true);
        let a = 1e-6;
        wcs.sip = Some(Sip {
            order: 2,
            a: vec![a, 0.0, a],
            b: vec![0.0, a, 0.0],
            ap: vec![0.0, 0.0, -a, 0.0, 0.0, -a],
            bp: vec![0.0, 0.0, 0.0, 0.0, -a, 0.0],
        });
        let cards = wcs.fits_header_cards();
        let number = |key: &str| {
            cards
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| match value {
                    FitsCardValue::Number(number) => Some(*number),
                    FitsCardValue::Integer(number) => Some(f64::from(*number)),
                    FitsCardValue::Text(_) => None,
                })
        };
        let text = |key: &str| {
            cards
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| match value {
                    FitsCardValue::Text(text) => Some((*text).to_owned()),
                    _ => None,
                })
        };
        assert_eq!(Wcs::from_fits_values(number, text), Some(wcs.clone()));

        // CDELT with CROTA2 describes the same linear part as a CD matrix.
        let plain = Wcs::from_center_scale_rotation((10.0, -20.0), (5.0, 7.0), 3.0, 30.0, false);
        let scale = 3.0 / 3600.0;
        let values = [
            ("CRVAL1", 10.0),
            ("CRVAL2", -20.0),
            ("CRPIX1", 6.0),
            ("CRPIX2", 8.0),
            ("CDELT1", -scale),
            ("CDELT2", -scale),
            ("CROTA2", 30.0),
        ];
        let from_cdelt = Wcs::from_fits_values(
            |key| {
                values
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, value)| *value)
            },
            |key| match key {
                "CTYPE1" => Some("RA---TAN".to_owned()),
                "CTYPE2" => Some("DEC--TAN".to_owned()),
                _ => None,
            },
        )
        .unwrap();
        for (x, y) in [(0.0, 0.0), (100.0, -40.0)] {
            let (expected, actual) = (plain.pixel_to_world(x, y), from_cdelt.pixel_to_world(x, y));
            assert_close(expected.0, actual.0, 1e-9);
            assert_close(expected.1, actual.1, 1e-9);
        }
        assert!(Wcs::from_fits_values(|_| Some(1.0), |_| Some("RA---SIN".to_owned())).is_none());
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
