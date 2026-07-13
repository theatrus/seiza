//! Solar-system minor bodies: comets and numbered asteroids.
//!
//! Orbital elements (from the Minor Planet Center) are propagated with
//! two-body mechanics to the image's acquisition time — these objects
//! move arcminutes to degrees per day, so a plate solve alone cannot
//! label them; the exact time matters. Geocentric positions include a
//! single light-time iteration; topocentric parallax is neglected
//! (significant only for very close approaches).

use crate::wcs::Wcs;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SEIZAMB1";
/// Obliquity of the ecliptic, J2000, degrees.
const OBLIQUITY_DEG: f64 = 23.439_291_1;
/// Gaussian gravitational constant: mean motion (rad/day) of a 1 AU orbit.
const GAUSS_K: f64 = 0.017_202_098_95;
/// Light travel time for 1 AU, days.
const LIGHT_DAYS_PER_AU: f64 = 0.005_775_518;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinorBodyKind {
    Comet = 0,
    Asteroid = 1,
}

/// Orbital elements in the J2000 ecliptic frame.
#[derive(Debug, Clone)]
pub struct MinorBody {
    pub kind: MinorBodyKind,
    /// Display name: "C/2025 A6 (Lemmon)", "(1) Ceres"
    pub name: String,
    /// Asteroids: epoch of the mean anomaly (JD TT).
    /// Comets: time of perihelion passage (JD TT).
    pub epoch_jd: f64,
    /// Asteroids: semi-major axis a (AU). Comets: perihelion distance q (AU).
    pub q_or_a: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub node_deg: f64,
    pub arg_perihelion_deg: f64,
    /// Mean anomaly at epoch, degrees (asteroids; zero for comets)
    pub mean_anomaly_deg: f64,
    /// Asteroids: absolute magnitude H. Comets: total magnitude m1.
    pub h_mag: f32,
    /// Asteroids: slope G. Comets: magnitude slope k1.
    pub slope: f32,
}

/// A minor body projected into an image.
#[derive(Debug, Clone)]
pub struct PlacedMinorBody {
    pub body: MinorBody,
    pub ra: f64,
    pub dec: f64,
    pub x: f64,
    pub y: f64,
    /// Estimated visual magnitude at the acquisition time
    pub mag: f64,
    /// Distance from Earth, AU
    pub delta_au: f64,
    /// Sky position angle (degrees east of north) of the characteristic
    /// direction: anti-solar for comets (the tail), apparent motion for
    /// asteroids (the trail)
    pub direction_pa_deg: Option<f64>,
}

pub struct MinorBodyCatalog {
    bodies: Vec<MinorBody>,
}

impl MinorBodyCatalog {
    pub fn new(bodies: Vec<MinorBody>) -> Self {
        Self { bodies }
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    pub fn bodies(&self) -> &[MinorBody] {
        &self.bodies
    }

    /// Geocentric J2000 position and estimated magnitude at `jd` (TT).
    /// `None` if the propagation fails to converge (freak orbits).
    pub fn position_at(body: &MinorBody, jd: f64) -> Option<(f64, f64, f64, f64)> {
        // One light-time iteration: evaluate, step back, re-evaluate
        let first = geocentric(body, jd)?;
        let retarded = jd - first.3 * LIGHT_DAYS_PER_AU;
        let (ra, dec, r_sun, delta) = geocentric(body, retarded)?;
        let mag = match body.kind {
            MinorBodyKind::Asteroid => body.h_mag as f64 + 5.0 * (r_sun * delta).log10(),
            MinorBodyKind::Comet => {
                body.h_mag as f64 + 5.0 * delta.log10() + 2.5 * body.slope as f64 * r_sun.log10()
            }
        };
        Some((ra, dec, mag, delta))
    }

    /// Bodies inside the WCS footprint at `jd`, brightest first.
    pub fn objects_in_footprint(
        &self,
        wcs: &Wcs,
        dimensions: (u32, u32),
        jd: f64,
        limit_mag: f64,
    ) -> Vec<PlacedMinorBody> {
        let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
        let sun = sun_geocentric_equatorial(jd);
        let mut placed: Vec<PlacedMinorBody> = self
            .bodies
            .iter()
            .filter_map(|body| {
                let (ra, dec, mag, delta_au) = Self::position_at(body, jd)?;
                if mag > limit_mag {
                    return None;
                }
                let (x, y) = wcs.world_to_pixel(ra, dec)?;
                if x < 0.0 || y < 0.0 || x >= width || y >= height {
                    return None;
                }
                let direction_pa_deg = match body.kind {
                    // Comet tails point anti-sunward (first order: both
                    // ion and dust tails)
                    MinorBodyKind::Comet => {
                        Some((bearing_deg(ra, dec, sun.0, sun.1) + 180.0).rem_euclid(360.0))
                    }
                    // Asteroids trail along their apparent motion
                    MinorBodyKind::Asteroid => Self::position_at(body, jd + 1.0 / 24.0)
                        .map(|(ra2, dec2, _, _)| bearing_deg(ra, dec, ra2, dec2)),
                };
                Some(PlacedMinorBody {
                    body: body.clone(),
                    ra,
                    dec,
                    x,
                    y,
                    mag,
                    delta_au,
                    direction_pa_deg,
                })
            })
            .collect();
        placed.sort_by(|a, b| a.mag.total_cmp(&b.mag));
        placed
    }

    pub fn write_to(&self, path: &Path) -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(self.bodies.len() as u32).to_le_bytes())?;
        for body in &self.bodies {
            out.write_all(&[body.kind as u8])?;
            let name = body.name.as_bytes();
            out.write_all(&(name.len() as u16).to_le_bytes())?;
            out.write_all(name)?;
            for value in [
                body.epoch_jd,
                body.q_or_a,
                body.eccentricity,
                body.inclination_deg,
                body.node_deg,
                body.arg_perihelion_deg,
                body.mean_anomaly_deg,
            ] {
                out.write_all(&value.to_le_bytes())?;
            }
            out.write_all(&body.h_mag.to_le_bytes())?;
            out.write_all(&body.slope.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn open(path: &Path) -> std::io::Result<Self> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let bad = |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string());
        if data.len() < 12 || &data[0..8] != MAGIC {
            return Err(bad("not a SEIZAMB1 file"));
        }
        let count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let mut bodies = Vec::with_capacity(count);
        let mut at = 12usize;
        for _ in 0..count {
            let kind = match data.get(at) {
                Some(0) => MinorBodyKind::Comet,
                Some(1) => MinorBodyKind::Asteroid,
                _ => return Err(bad("bad body kind")),
            };
            at += 1;
            let name_len = u16::from_le_bytes(
                data.get(at..at + 2)
                    .ok_or_else(|| bad("truncated"))?
                    .try_into()
                    .unwrap(),
            ) as usize;
            at += 2;
            let name = String::from_utf8_lossy(
                data.get(at..at + name_len)
                    .ok_or_else(|| bad("truncated"))?,
            )
            .to_string();
            at += name_len;
            let mut values = [0.0f64; 7];
            for value in &mut values {
                *value = f64::from_le_bytes(
                    data.get(at..at + 8)
                        .ok_or_else(|| bad("truncated"))?
                        .try_into()
                        .unwrap(),
                );
                at += 8;
            }
            let h_mag = f32::from_le_bytes(
                data.get(at..at + 4)
                    .ok_or_else(|| bad("truncated"))?
                    .try_into()
                    .unwrap(),
            );
            at += 4;
            let slope = f32::from_le_bytes(
                data.get(at..at + 4)
                    .ok_or_else(|| bad("truncated"))?
                    .try_into()
                    .unwrap(),
            );
            at += 4;
            bodies.push(MinorBody {
                kind,
                name,
                epoch_jd: values[0],
                q_or_a: values[1],
                eccentricity: values[2],
                inclination_deg: values[3],
                node_deg: values[4],
                arg_perihelion_deg: values[5],
                mean_anomaly_deg: values[6],
                h_mag,
                slope,
            });
        }
        Ok(Self { bodies })
    }
}

/// Geocentric J2000 RA/Dec (degrees) plus heliocentric and geocentric
/// distances (AU) at `jd`.
fn geocentric(body: &MinorBody, jd: f64) -> Option<(f64, f64, f64, f64)> {
    let helio = heliocentric_ecliptic(body, jd)?;
    let earth = earth_heliocentric_ecliptic(jd);
    let geo = [
        helio[0] - earth[0],
        helio[1] - earth[1],
        helio[2] - earth[2],
    ];

    // Ecliptic → equatorial J2000
    let (sin_ob, cos_ob) = OBLIQUITY_DEG.to_radians().sin_cos();
    let x = geo[0];
    let y = geo[1] * cos_ob - geo[2] * sin_ob;
    let z = geo[1] * sin_ob + geo[2] * cos_ob;

    let delta = (x * x + y * y + z * z).sqrt();
    let r_sun = (helio[0] * helio[0] + helio[1] * helio[1] + helio[2] * helio[2]).sqrt();
    let ra = y.atan2(x).to_degrees().rem_euclid(360.0);
    let dec = (z / delta).asin().to_degrees();
    Some((ra, dec, r_sun, delta))
}

/// Heliocentric J2000-ecliptic position of the body, AU.
fn heliocentric_ecliptic(body: &MinorBody, jd: f64) -> Option<[f64; 3]> {
    let e = body.eccentricity;
    // Perifocal position (xi toward perihelion, eta in-plane)
    let (xi, eta) = match body.kind {
        MinorBodyKind::Asteroid => {
            let a = body.q_or_a;
            let n = GAUSS_K / (a * a * a).sqrt();
            let mean = body.mean_anomaly_deg.to_radians() + n * (jd - body.epoch_jd);
            let ecc_anom = solve_kepler_elliptic(mean, e)?;
            (
                a * (ecc_anom.cos() - e),
                a * (1.0 - e * e).sqrt() * ecc_anom.sin(),
            )
        }
        MinorBodyKind::Comet => {
            let q = body.q_or_a;
            let dt = jd - body.epoch_jd; // days since perihelion
            if (e - 1.0).abs() < 1e-6 {
                // Parabolic: Barker's equation
                let w = 3.0 * GAUSS_K / (2.0 * q * (2.0 * q).sqrt()) * dt;
                let s = barker(w);
                (q * (1.0 - s * s), 2.0 * q * s)
            } else if e < 1.0 {
                let a = q / (1.0 - e);
                let n = GAUSS_K / (a * a * a).sqrt();
                let mean = n * dt;
                let ecc_anom = solve_kepler_elliptic(mean, e)?;
                (
                    a * (ecc_anom.cos() - e),
                    a * (1.0 - e * e).sqrt() * ecc_anom.sin(),
                )
            } else {
                // Hyperbolic
                let a = q / (e - 1.0); // positive
                let n = GAUSS_K / (a * a * a).sqrt();
                let mean = n * dt;
                let hyp_anom = solve_kepler_hyperbolic(mean, e)?;
                (
                    a * (e - hyp_anom.cosh()),
                    a * (e * e - 1.0).sqrt() * hyp_anom.sinh(),
                )
            }
        }
    };

    Some(rotate_to_ecliptic(
        xi,
        eta,
        body.arg_perihelion_deg,
        body.inclination_deg,
        body.node_deg,
    ))
}

/// Rotate perifocal (xi, eta) into the J2000 ecliptic frame.
fn rotate_to_ecliptic(
    xi: f64,
    eta: f64,
    arg_peri_deg: f64,
    incl_deg: f64,
    node_deg: f64,
) -> [f64; 3] {
    let (sin_w, cos_w) = arg_peri_deg.to_radians().sin_cos();
    let (sin_i, cos_i) = incl_deg.to_radians().sin_cos();
    let (sin_o, cos_o) = node_deg.to_radians().sin_cos();

    let px = cos_w * cos_o - sin_w * sin_o * cos_i;
    let py = cos_w * sin_o + sin_w * cos_o * cos_i;
    let pz = sin_w * sin_i;
    let qx = -sin_w * cos_o - cos_w * sin_o * cos_i;
    let qy = -sin_w * sin_o + cos_w * cos_o * cos_i;
    let qz = cos_w * sin_i;

    [px * xi + qx * eta, py * xi + qy * eta, pz * xi + qz * eta]
}

/// Solve Kepler's equation M = E - e sin E. Newton first; f is strictly
/// monotonic for e < 1, so bisection is a guaranteed fallback for the
/// near-parabolic corner cases where Newton oscillates.
fn solve_kepler_elliptic(mean_anomaly: f64, e: f64) -> Option<f64> {
    let mean = mean_anomaly.rem_euclid(std::f64::consts::TAU);
    let mut ecc = if e > 0.8 { std::f64::consts::PI } else { mean };
    for _ in 0..60 {
        let delta = (ecc - e * ecc.sin() - mean) / (1.0 - e * ecc.cos());
        ecc -= delta;
        if delta.abs() < 1e-12 {
            return Some(ecc);
        }
    }
    let (mut lo, mut hi) = (0.0f64, std::f64::consts::TAU);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid - e * mid.sin() < mean {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Newton solve of the hyperbolic Kepler equation M = e sinh H - H.
fn solve_kepler_hyperbolic(mean: f64, e: f64) -> Option<f64> {
    let mut h = (2.0 * mean.abs() / e + 1.8).ln().copysign(mean);
    for _ in 0..80 {
        let delta = (e * h.sinh() - h - mean) / (e * h.cosh() - 1.0);
        h -= delta;
        if delta.abs() < 1e-12 {
            return Some(h);
        }
    }
    None
}

/// Barker's equation for parabolic orbits: solve s^3/3 + s = w.
fn barker(w: f64) -> f64 {
    // Closed form: with s = y - 1/y, the cubic becomes y^3 - 1/y^3 = 3w
    let y = ((3.0 * w + (9.0 * w * w + 4.0).sqrt()) / 2.0).cbrt();
    y - 1.0 / y
}

/// Heliocentric J2000-ecliptic position of the Earth-Moon barycenter, AU.
/// JPL "Approximate Positions of the Planets" Keplerian elements, valid
/// 1800–2050 with errors well under an arcminute as seen from typical
/// minor-body distances.
fn earth_heliocentric_ecliptic(jd: f64) -> [f64; 3] {
    let t = (jd - 2_451_545.0) / 36_525.0; // Julian centuries from J2000
    let a = 1.000_002_61 + 0.000_005_62 * t;
    let e = 0.016_711_23 - 0.000_043_92 * t;
    let incl: f64 = -0.000_015_31 - 0.012_946_68 * t;
    let mean_longitude = 100.464_571_66 + 35_999.372_449_81 * t;
    let peri_longitude = 102.937_681_93 + 0.323_273_64 * t;
    let node: f64 = 0.0;

    let arg_peri = peri_longitude - node;
    let mean_anomaly = (mean_longitude - peri_longitude).to_radians();
    let ecc_anom = solve_kepler_elliptic(mean_anomaly, e).unwrap_or(mean_anomaly);
    let xi = a * (ecc_anom.cos() - e);
    let eta = a * (1.0 - e * e).sqrt() * ecc_anom.sin();
    rotate_to_ecliptic(xi, eta, arg_peri, incl, node)
}

/// Geocentric equatorial (RA, Dec) of the Sun at `jd`, degrees.
fn sun_geocentric_equatorial(jd: f64) -> (f64, f64) {
    let earth = earth_heliocentric_ecliptic(jd);
    let (x, y, z) = (-earth[0], -earth[1], -earth[2]);
    let (sin_ob, cos_ob) = OBLIQUITY_DEG.to_radians().sin_cos();
    let (eq_y, eq_z) = (y * cos_ob - z * sin_ob, y * sin_ob + z * cos_ob);
    let r = (x * x + eq_y * eq_y + eq_z * eq_z).sqrt();
    (
        eq_y.atan2(x).to_degrees().rem_euclid(360.0),
        (eq_z / r).asin().to_degrees(),
    )
}

/// Initial bearing (position angle east of north, degrees) from point 1
/// toward point 2 on the celestial sphere.
fn bearing_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let d_ra = (ra2 - ra1).to_radians();
    let (sin_d1, cos_d1) = dec1.to_radians().sin_cos();
    let (sin_d2, cos_d2) = dec2.to_radians().sin_cos();
    (d_ra.sin() * cos_d2)
        .atan2(cos_d1 * sin_d2 - sin_d1 * cos_d2 * d_ra.cos())
        .to_degrees()
        .rem_euclid(360.0)
}

/// Julian date (UTC ≈ TT for labeling purposes) from a calendar date.
pub fn julian_date(year: i32, month: u32, day: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor() + day + b - 1524.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_dates() {
        // J2000.0
        assert!((julian_date(2000, 1, 1.5) - 2_451_545.0).abs() < 1e-9);
        // A modern date
        assert!((julian_date(2026, 7, 13.0) - 2_461_234.5).abs() < 1.0);
    }

    #[test]
    fn earth_distance_is_about_one_au() {
        for jd in [2_451_545.0, 2_460_000.0, 2_461_000.0] {
            let p = earth_heliocentric_ecliptic(jd);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((0.98..1.02).contains(&r), "r = {r} at {jd}");
        }
    }

    #[test]
    fn kepler_solvers_converge() {
        for e in [0.0, 0.1, 0.6, 0.95, 0.999] {
            for m in [-2.5, 0.0, 0.3, 3.0, 6.0] {
                let ecc = solve_kepler_elliptic(m, e).unwrap();
                assert!((ecc - e * ecc.sin() - m.rem_euclid(std::f64::consts::TAU)).abs() < 1e-9);
            }
        }
        for e in [1.01, 1.2, 3.0] {
            for m in [-4.0, 0.5, 8.0] {
                let h = solve_kepler_hyperbolic(m, e).unwrap();
                assert!((e * h.sinh() - h - m).abs() < 1e-8);
            }
        }
        // Barker round-trip
        for w in [-5.0, -0.2, 0.0, 0.4, 7.0] {
            let s = barker(w);
            assert!((s * s * s / 3.0 + s - w).abs() < 1e-9);
        }
    }

    #[test]
    fn orbit_radius_matches_conic_geometry() {
        // r from the propagated position must satisfy the conic equation
        // r = p / (1 + e cos(nu)) at every time — a strong internal check
        let body = MinorBody {
            kind: MinorBodyKind::Asteroid,
            name: "test".into(),
            epoch_jd: 2_460_000.5,
            q_or_a: 2.5,
            eccentricity: 0.2,
            inclination_deg: 12.0,
            node_deg: 45.0,
            arg_perihelion_deg: 110.0,
            mean_anomaly_deg: 30.0,
            h_mag: 10.0,
            slope: 0.15,
        };
        let a = body.q_or_a;
        let e = body.eccentricity;
        for offset in [-800.0, -3.0, 0.0, 42.0, 1234.5] {
            let p = heliocentric_ecliptic(&body, body.epoch_jd + offset).unwrap();
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                r >= a * (1.0 - e) - 1e-9 && r <= a * (1.0 + e) + 1e-9,
                "r = {r}"
            );
        }
    }
}
