//! Catalog data builders.

use anyhow::{Context, Result, bail};
use seiza::catalog::TileSetBuilder;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Build a star tile file from the Tycho-2 catalogue (CDS I/259).
///
/// Expects the distribution files `tyc2.dat.NN[.gz]` in `input`. Mean
/// positions (ICRS, epoch J2000) are proper-motion corrected to `epoch`;
/// entries without a mean position fall back to the observed position.
pub fn build_tycho2(input: &Path, output: &Path, epoch: f64, max_mag: f32) -> Result<()> {
    let mut parts: Vec<_> = std::fs::read_dir(input)
        .with_context(|| format!("cannot read {}", input.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("tyc2.dat."))
        })
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!("no tyc2.dat.* files found in {}", input.display());
    }

    // 45 declination bands ≈ 4° tiles: right-sized for the ~2.5M star
    // lite tier
    let mut builder = TileSetBuilder::new(45, epoch);
    let mut skipped_no_mag = 0u64;
    let mut too_faint = 0u64;

    for part in &parts {
        let file =
            std::fs::File::open(part).with_context(|| format!("cannot open {}", part.display()))?;
        let reader: Box<dyn std::io::Read> = if part.extension().is_some_and(|ext| ext == "gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else {
            Box::new(file)
        };

        for line in BufReader::new(reader).lines() {
            let line = line?;
            let Some(star) = parse_tycho2_line(&line, epoch) else {
                skipped_no_mag += 1;
                continue;
            };
            if star.2 > max_mag {
                too_faint += 1;
                continue;
            }
            builder.add(star.0, star.1, star.2);
        }
    }

    let count = builder.star_count();
    builder.write_to(output)?;
    println!(
        "{} stars written to {} (epoch {epoch}, {} unusable records skipped, {} fainter than {max_mag})",
        count,
        output.display(),
        skipped_no_mag,
        too_faint
    );
    Ok(())
}

/// Parse one fixed-width Tycho-2 record into (ra, dec, mag) at `epoch`.
fn parse_tycho2_line(line: &str, epoch: f64) -> Option<(f64, f64, f32)> {
    // Byte ranges from the CDS ReadMe are 1-indexed inclusive
    let field =
        |from: usize, to: usize| -> &str { line.get(from - 1..to).map(str::trim).unwrap_or("") };

    // VT magnitude, falling back to BT
    let mag: f32 = field(124, 129)
        .parse()
        .or_else(|_| field(111, 116).parse())
        .ok()?;

    // Mean position (may be absent when pflag is X), else observed position
    if let (Ok(ra), Ok(dec)) = (field(16, 27).parse::<f64>(), field(29, 40).parse::<f64>()) {
        let dt = epoch - 2000.0;
        // mas/yr; pmRA includes cos(dec)
        let pm_ra: f64 = field(42, 48).parse().unwrap_or(0.0);
        let pm_dec: f64 = field(50, 56).parse().unwrap_or(0.0);
        let cos_dec = dec.to_radians().cos().max(1e-6);
        let ra = (ra + pm_ra * dt / 3_600_000.0 / cos_dec).rem_euclid(360.0);
        let dec = (dec + pm_dec * dt / 3_600_000.0).clamp(-90.0, 90.0);
        return Some((ra, dec, mag));
    }

    let ra = field(153, 164).parse::<f64>().ok()?;
    let dec = field(166, 177).parse::<f64>().ok()?;
    Some((ra, dec, mag))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real Tycho-2 record (TYC 1-1-1)
    const SAMPLE: &str = "0001 00008 1| |  2.31750494|  2.23184345|  -16.3|   -9.0| 68| 73| 1.7| 1.8|1958.89|1951.94| 4|1.0|1.0|0.9|1.0|12.146|0.158|12.146|0.223|999| |         |  2.31754222|  2.23186444|1.67|1.54| 88.0|100.8| |-0.2";

    #[test]
    fn parses_a_real_record() {
        let (ra, dec, mag) = parse_tycho2_line(SAMPLE, 2000.0).unwrap();
        assert!((ra - 2.31750494).abs() < 1e-8);
        assert!((dec - 2.23184345).abs() < 1e-8);
        assert!((mag - 12.146).abs() < 1e-3);
    }

    #[test]
    fn applies_proper_motion() {
        let (ra, dec, _) = parse_tycho2_line(SAMPLE, 2025.0).unwrap();
        // pmRA = -16.3 mas/yr over 25 years ≈ -0.41" of RA*cos(dec)
        let d_ra_arcsec = (ra - 2.31750494) * 3600.0 * dec.to_radians().cos();
        assert!((d_ra_arcsec - -0.4075).abs() < 0.01, "{d_ra_arcsec}");
        let d_dec_arcsec = (dec - 2.23184345) * 3600.0;
        assert!((d_dec_arcsec - -0.225).abs() < 0.01, "{d_dec_arcsec}");
    }

    #[test]
    fn rejects_magnitude_free_records() {
        let mut broken = SAMPLE.to_string();
        broken.replace_range(110..135, &" ".repeat(25));
        assert!(parse_tycho2_line(&broken, 2000.0).is_none());
    }
}
