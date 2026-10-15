//! What an image's headers say about where it points: a WCS from an
//! earlier solve, the mount's RA and Dec, or a pixel scale from the
//! camera's pixel size and the telescope's focal length. `seiza`'s
//! `prior::solve_with_prior` solves with it.

use crate::wcs::Wcs;

/// What an image's headers say about where it points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolvePrior {
    /// A TAN or TAN-SIP solution the headers carry, as written. It may be
    /// stale.
    pub wcs: Option<Wcs>,
    /// Where the image points, (RA, Dec) in degrees: from `RA`/`DEC`,
    /// `OBJCTRA`/`OBJCTDEC`, or `CRVAL1`/`CRVAL2` when the WCS is unusable.
    pub center: Option<(f64, f64)>,
    /// Arcseconds per pixel: from the WCS, `PIXSCALE`/`SCALE`/`SECPIX`, or
    /// `XPIXSZ` (micrometres, binning included, as capture programs write
    /// it) over `FOCALLEN` (millimetres).
    pub scale_arcsec_px: Option<f64>,
}

/// Arcseconds per radian over 1000, for micrometres over millimetres.
const ARCSEC_PER_MICRON_PER_MM: f64 = 206.264_806_247;

impl SolvePrior {
    /// Read a prior from header cards: `number` looks a numeric card up by
    /// keyword, `text` a string card. Values that cannot be right — a
    /// declination past a pole, a scale that is not positive — are left out.
    pub fn from_header(
        number: impl Fn(&str) -> Option<f64>,
        text: impl Fn(&str) -> Option<String>,
    ) -> Self {
        let wcs = Wcs::from_fits_values(&number, &text).filter(|wcs| {
            wcs.crval.0.is_finite()
                && wcs.crval.1.is_finite()
                && plausible_scale(wcs.scale_arcsec_per_px())
        });
        let numeric_center = match (number("RA"), number("DEC")) {
            (Some(ra), Some(dec)) => sky_position(ra, dec),
            _ => None,
        };
        let sexagesimal_center = || {
            let ra = parse_sexagesimal(&text("OBJCTRA").or_else(|| text("RA"))?)? * 15.0;
            let dec = parse_sexagesimal(&text("OBJCTDEC").or_else(|| text("DEC"))?)?;
            sky_position(ra, dec)
        };
        let crval_center = || match (number("CRVAL1"), number("CRVAL2")) {
            (Some(ra), Some(dec)) => sky_position(ra, dec),
            _ => None,
        };
        let center = numeric_center
            .or_else(sexagesimal_center)
            .or_else(crval_center);

        let first = |keys: &[&str]| keys.iter().find_map(|key| number(key));
        let scale = wcs
            .as_ref()
            .map(Wcs::scale_arcsec_per_px)
            .or_else(|| first(&["PIXSCALE", "SCALE", "SECPIX"]))
            .or_else(|| {
                let pixel_um = first(&["XPIXSZ", "PIXSIZE1"])?;
                let focal_mm = first(&["FOCALLEN", "FOCALLENGTH", "FOCAL"])?;
                Some(ARCSEC_PER_MICRON_PER_MM * pixel_um / focal_mm)
            })
            .filter(|scale| plausible_scale(*scale));
        Self {
            wcs,
            center,
            scale_arcsec_px: scale,
        }
    }

    /// Whether the headers said nothing useful.
    pub fn is_empty(&self) -> bool {
        self.wcs.is_none() && self.center.is_none() && self.scale_arcsec_px.is_none()
    }
}

fn plausible_scale(scale: f64) -> bool {
    scale.is_finite() && scale > 1e-3 && scale < 3600.0
}

fn sky_position(ra: f64, dec: f64) -> Option<(f64, f64)> {
    (ra.is_finite() && dec.is_finite() && dec.abs() <= 90.0).then(|| (ra.rem_euclid(360.0), dec))
}

/// `"05 35 17.3"`, `"-05:23:28"` or `"12h30m"` as a decimal number in the
/// first field's unit; `None` for anything else, including a plain decimal
/// number, which the numeric card already gave.
fn parse_sexagesimal(text: &str) -> Option<f64> {
    let text = text.trim();
    let negative = text.starts_with('-');
    let fields: Vec<f64> = text
        .trim_start_matches(['+', '-'])
        .split(|c: char| c.is_whitespace() || matches!(c, ':' | 'h' | 'm' | 's' | 'd' | '°'))
        .filter(|field| !field.is_empty())
        .map(|field| field.trim_end_matches(['\'', '"']).parse::<f64>())
        .collect::<Result<_, _>>()
        .ok()?;
    if !(2..=3).contains(&fields.len())
        || fields
            .iter()
            .any(|field| !field.is_finite() || *field < 0.0)
        || fields[1..].iter().any(|field| *field >= 60.0)
    {
        return None;
    }
    let value = fields[0] + fields[1] / 60.0 + fields.get(2).copied().unwrap_or(0.0) / 3600.0;
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn prior(cards: &[(&str, &str)]) -> SolvePrior {
        let cards: HashMap<&str, &str> = cards.iter().copied().collect();
        SolvePrior::from_header(
            |key| cards.get(key).and_then(|value| value.trim().parse().ok()),
            |key| cards.get(key).map(|value| value.to_string()),
        )
    }

    #[test]
    fn a_header_gives_its_position_and_scale() {
        // Degrees, as N.I.N.A. writes RA and DEC.
        let mount = prior(&[("RA", "83.82"), ("DEC", "-5.39")]);
        assert_eq!(mount.center, Some((83.82, -5.39)));
        assert_eq!(mount.scale_arcsec_px, None);

        // Hours and degrees, as OBJCTRA and OBJCTDEC are written.
        let object = prior(&[("OBJCTRA", "05 35 17.3"), ("OBJCTDEC", "-05 23 28")]);
        let (ra, dec) = object.center.unwrap();
        assert!((ra - 83.822_08).abs() < 1e-4, "{ra}");
        assert!((dec + 5.391_11).abs() < 1e-4, "{dec}");
        let colons = prior(&[("OBJCTRA", "5:35:17.3"), ("OBJCTDEC", "+22:00:52")]);
        assert!((colons.center.unwrap().1 - 22.014_44).abs() < 1e-4);

        // Pixel size over focal length: 3.76 µm at 530 mm.
        let optics = prior(&[("XPIXSZ", "3.76"), ("FOCALLEN", "530")]);
        let scale = optics.scale_arcsec_px.unwrap();
        assert!((scale - 1.4633).abs() < 1e-3, "{scale}");
        assert_eq!(
            prior(&[
                ("PIXSCALE", "1.25"),
                ("XPIXSZ", "3.76"),
                ("FOCALLEN", "530")
            ])
            .scale_arcsec_px,
            Some(1.25)
        );

        // Nothing usable.
        assert!(prior(&[("DEC", "95"), ("RA", "10")]).is_empty());
        assert!(prior(&[("FOCALLEN", "0"), ("XPIXSZ", "3.76")]).is_empty());
        assert!(prior(&[("OBJCTRA", "12 75 00"), ("OBJCTDEC", "10 00")]).is_empty());
    }

    #[test]
    fn a_header_wcs_gives_position_and_scale_too() {
        let with_wcs = prior(&[
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRVAL1", "150.5"),
            ("CRVAL2", "33.2"),
            ("CRPIX1", "2000.5"),
            ("CRPIX2", "1500.5"),
            ("CD1_1", "-0.0005"),
            ("CD1_2", "0"),
            ("CD2_1", "0"),
            ("CD2_2", "0.0005"),
        ]);
        let wcs = with_wcs.wcs.as_ref().unwrap();
        assert_eq!(wcs.crval, (150.5, 33.2));
        assert!((with_wcs.scale_arcsec_px.unwrap() - 1.8).abs() < 1e-9);
        // The pointing comes from RA/DEC when given, else CRVAL.
        assert_eq!(with_wcs.center, Some((150.5, 33.2)));

        // An unusable projection still leaves CRVAL as a position.
        let sin = prior(&[
            ("CTYPE1", "RA---SIN"),
            ("CTYPE2", "DEC--SIN"),
            ("CRVAL1", "10"),
            ("CRVAL2", "20"),
        ]);
        assert!(sin.wcs.is_none());
        assert_eq!(sin.center, Some((10.0, 20.0)));
    }
}
