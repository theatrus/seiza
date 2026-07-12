//! Catalog data builders.

use anyhow::{Context, Result, bail};
use seiza::catalog::TileSetBuilder;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Build a star tile file from the Tycho-2 catalogue (CDS I/259).
///
/// Expects the distribution files `tyc2.dat.NN[.gz]` in `input`, plus
/// `suppl_1.dat[.gz]` — the supplement holds most stars brighter than
/// magnitude ~2 (Sirius is not in the main catalogue). Mean positions
/// (ICRS, epoch J2000; supplement epoch J1991.25) are proper-motion
/// corrected to `epoch`; entries without a mean position fall back to the
/// observed position.
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

    let mut suppl_count = 0u64;
    for name in ["suppl_1.dat.gz", "suppl_1.dat"] {
        let path = input.join(name);
        if !path.exists() {
            continue;
        }
        let file = std::fs::File::open(&path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let reader: Box<dyn std::io::Read> = if name.ends_with(".gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let Some((ra, dec, mag)) = parse_tycho2_suppl_line(&line, epoch) else {
                skipped_no_mag += 1;
                continue;
            };
            if mag > max_mag {
                too_faint += 1;
                continue;
            }
            builder.add(ra, dec, mag);
            suppl_count += 1;
        }
        break;
    }
    if suppl_count == 0 {
        eprintln!(
            "warning: no supplement-1 stars ingested — the brightest stars \
             (including Sirius) will be missing; run download-data tycho2"
        );
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

/// Build a star tile file from an ASTAP `.1476` star database directory
/// (e.g. D80, Gaia DR3). The format is documented in ASTAP's
/// unit_star_database.pas: each of the 1476 sky-area files has a 110-byte
/// header (text description, record size in the final byte) followed by
/// 5-byte records; `FF FF FF` section headers carry the dec high byte
/// (offset +128) and the section magnitude ((byte - 16) / 10).
pub fn build_astap(
    input: &Path,
    output: &Path,
    epoch: f64,
    max_mag: f32,
    bands: u32,
) -> Result<()> {
    let mut parts: Vec<_> = std::fs::read_dir(input)
        .with_context(|| format!("cannot read {}", input.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "1476"))
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!("no .1476 files found in {}", input.display());
    }

    let mut builder = TileSetBuilder::new(bands, epoch);
    let mut too_faint = 0u64;

    for part in &parts {
        let mut reader = BufReader::with_capacity(
            1 << 20,
            std::fs::File::open(part).with_context(|| format!("cannot open {}", part.display()))?,
        );
        let mut header = [0u8; 110];
        std::io::Read::read_exact(&mut reader, &mut header)
            .with_context(|| format!("{} is too short for a header", part.display()))?;
        let record_size = header[109];
        if record_size != 5 {
            bail!(
                "{}: unsupported record size {record_size} (only the 5-byte \
                 1476 format is supported)",
                part.display()
            );
        }

        let mut record = [0u8; 5];
        let mut dec9: i32 = 0;
        let mut mag: f32 = 0.0;
        loop {
            match std::io::Read::read_exact(&mut reader, &mut record) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            if record[0] == 0xFF && record[1] == 0xFF && record[2] == 0xFF {
                dec9 = record[3] as i32 - 128;
                mag = (record[4] as f32 - 16.0) / 10.0;
                continue;
            }
            if mag > max_mag {
                too_faint += 1;
                continue;
            }
            let ra = (record[0] as f64 + record[1] as f64 * 256.0 + record[2] as f64 * 65536.0)
                * 360.0
                / ((1u32 << 24) - 1) as f64;
            let dec_int = record[3] as i32 + record[4] as i32 * 256 + dec9 * 65536;
            let dec = dec_int as f64 * 90.0 / ((128 * 65536) - 1) as f64;
            builder.add(ra, dec, mag);
        }
    }

    let count = builder.star_count();
    builder.write_to(output)?;
    println!(
        "{} stars written to {} (epoch {epoch}, {} fainter than {max_mag})",
        count,
        output.display(),
        too_faint
    );
    Ok(())
}

/// Parse one supplement-1/2 record: positions are ICRS at epoch J1991.25.
fn parse_tycho2_suppl_line(line: &str, epoch: f64) -> Option<(f64, f64, f32)> {
    let field =
        |from: usize, to: usize| -> &str { line.get(from - 1..to).map(str::trim).unwrap_or("") };

    let mag: f32 = field(97, 102)
        .parse()
        .or_else(|_| field(84, 89).parse())
        .ok()?;
    let ra = field(16, 27).parse::<f64>().ok()?;
    let dec = field(29, 40).parse::<f64>().ok()?;

    let dt = epoch - 1991.25;
    let pm_ra: f64 = field(42, 48).parse().unwrap_or(0.0);
    let pm_dec: f64 = field(50, 56).parse().unwrap_or(0.0);
    let cos_dec = dec.to_radians().cos().max(1e-6);
    let ra = (ra + pm_ra * dt / 3_600_000.0 / cos_dec).rem_euclid(360.0);
    let dec = (dec + pm_dec * dt / 3_600_000.0).clamp(-90.0, 90.0);
    Some((ra, dec, mag))
}

/// Build an object catalog from OpenNGC, VizieR Sharpless/Barnard TSVs, and
/// the IAU star-name list, whichever are present in `input`.
pub fn build_objects(input: &Path, output: &Path) -> Result<()> {
    use seiza::objects::{ObjectCatalog, ObjectKind, SkyObject};

    let mut objects = Vec::new();
    let mut sources = 0;

    for name in ["NGC.csv", "addendum.csv"] {
        let path = input.join(name);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines().skip(1) {
            if let Some(object) = parse_openngc_line(line) {
                objects.push(object);
            }
        }
    }

    for (file, prefix, kind) in [
        ("sh2.tsv", "Sh2-", ObjectKind::HiiRegion),
        ("barnard.tsv", "B", ObjectKind::DarkNebula),
    ] {
        let path = input.join(file);
        if !path.exists() {
            continue;
        }
        sources += 1;
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if fields.len() < 3 {
                continue;
            }
            let (Ok(ra), Ok(dec)) = (fields[0].parse::<f64>(), fields[1].parse::<f64>()) else {
                continue; // column header / units / separator rows
            };
            let Ok(number) = fields[2].parse::<u32>() else {
                continue;
            };
            let diam: Option<f32> = fields.get(3).and_then(|d| d.parse().ok());
            objects.push(SkyObject {
                kind,
                ra,
                dec,
                mag: None,
                major_arcmin: diam.filter(|d| *d > 0.0),
                minor_arcmin: None,
                position_angle_deg: None,
                name: format!("{prefix}{number}"),
                common_name: String::new(),
            });
        }
    }

    let csn = input.join("IAU-CSN.txt");
    if csn.exists() {
        sources += 1;
        let content = std::fs::read_to_string(&csn)?;
        for line in content.lines() {
            if let Some(object) = parse_iau_csn_line(line) {
                objects.push(object);
            }
        }
    }

    if sources == 0 {
        bail!(
            "no catalog sources found in {} (expected NGC.csv, sh2.tsv, \
             barnard.tsv, IAU-CSN.txt); run download-data objects first",
            input.display()
        );
    }

    let catalog = ObjectCatalog::new(objects);
    let count = catalog.len();
    catalog.write_to(output)?;
    println!(
        "{count} objects from {sources} sources written to {}",
        output.display()
    );
    Ok(())
}

/// One `;`-separated OpenNGC row. Skips duplicates and non-existent entries.
fn parse_openngc_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, SkyObject};

    let fields: Vec<&str> = line.split(';').collect();
    if fields.len() < 30 {
        return None;
    }
    let kind = match fields[1] {
        "G" | "GPair" | "GTrpl" | "GGroup" => ObjectKind::Galaxy,
        "OCl" => ObjectKind::OpenCluster,
        "GCl" => ObjectKind::GlobularCluster,
        "PN" => ObjectKind::PlanetaryNebula,
        "HII" => ObjectKind::HiiRegion,
        "SNR" => ObjectKind::SupernovaRemnant,
        "DrkN" => ObjectKind::DarkNebula,
        "Neb" | "EmN" | "RfN" => ObjectKind::Nebula,
        "Cl+N" => ObjectKind::ClusterWithNebula,
        // Bare star entries are catalog-number noise next to the IAU
        // named-star list (e.g. IC 1318 is typed as the star gamma Cyg)
        "*" | "**" => return None,
        "*Ass" => ObjectKind::Association,
        "Dup" | "NonEx" => return None,
        _ => ObjectKind::Other,
    };

    // RA "HH:MM:SS.ss", Dec "+DD:MM:SS.s"
    let ra = parse_sexagesimal(fields[2])? * 15.0;
    let dec = parse_sexagesimal(fields[3])?;

    // Prefer the Messier designation, prettify the NGC/IC name
    let name = match fields.get(23).map(|m| m.trim_start_matches('0')) {
        Some(m) if !m.is_empty() => format!("M {m}"),
        _ => {
            let raw = fields[0];
            if let Some(rest) = raw.strip_prefix("NGC") {
                format!("NGC {}", rest.trim_start_matches('0'))
            } else if let Some(rest) = raw.strip_prefix("IC") {
                format!("IC {}", rest.trim_start_matches('0'))
            } else {
                raw.to_string()
            }
        }
    };
    let common_name = fields
        .get(28)
        .and_then(|names| names.split(',').next())
        .unwrap_or("")
        .trim()
        .to_string();

    Some(SkyObject {
        kind,
        ra,
        dec,
        mag: fields[9].parse().ok().or_else(|| fields[8].parse().ok()),
        major_arcmin: fields[5].parse().ok().filter(|v: &f32| *v > 0.0),
        minor_arcmin: fields[6].parse().ok().filter(|v: &f32| *v > 0.0),
        position_angle_deg: fields[7].parse().ok(),
        name,
        common_name,
    })
}

/// "HH:MM:SS.ss" or "+DD:MM:SS.s" to a float in the leading unit.
fn parse_sexagesimal(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let parts: Vec<&str> = value.trim_start_matches(['-', '+']).split(':').collect();
    let mut total = 0.0;
    let mut scale = 1.0;
    for part in parts {
        total += part.parse::<f64>().ok()? * scale;
        scale /= 60.0;
    }
    Some(if negative { -total } else { total })
}

/// One line of the IAU-CSN list. The ASCII name occupies the first 18
/// bytes; RA/Dec (J2000, degrees) are anchored by the date column.
fn parse_iau_csn_line(line: &str) -> Option<seiza::objects::SkyObject> {
    use seiza::objects::{ObjectKind, SkyObject};

    if line.starts_with('#') || line.len() < 40 || !line.is_ascii() && line.get(..18).is_none() {
        return None;
    }
    let name = line.get(..18)?.trim();
    if name.is_empty() {
        return None;
    }
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let date_index = tokens
        .iter()
        .position(|t| t.len() == 10 && t.as_bytes()[4] == b'-' && t.as_bytes()[7] == b'-')?;
    if date_index < 6 {
        return None;
    }
    let ra: f64 = tokens[date_index - 2].parse().ok()?;
    let dec: f64 = tokens[date_index - 1].parse().ok()?;
    let mag: Option<f32> = tokens[date_index - 6].parse().ok();

    Some(SkyObject {
        kind: ObjectKind::Star,
        ra,
        dec,
        mag,
        major_arcmin: None,
        minor_arcmin: None,
        position_angle_deg: None,
        name: name.to_string(),
        common_name: name.to_string(),
    })
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

    // The real supplement-1 record for Sirius (TYC 5949-2777-1)
    const SIRIUS: &str = "5949 02777 1|H|101.28854105|-16.71314306| -546.0|-1223.1|  1.2|  1.0|  1.3|  1.2|H|      |     |-1.088|0.002|999| | 32349 ";

    #[test]
    fn parses_a_supplement_record_with_proper_motion() {
        let (ra, dec, mag) = parse_tycho2_suppl_line(SIRIUS, 1991.25).unwrap();
        assert!((ra - 101.28854105).abs() < 1e-8);
        assert!((dec - -16.71314306).abs() < 1e-8);
        assert!((mag - -1.088).abs() < 1e-3);

        // Sirius moves fast: ~-546 mas/yr (RA*cos dec), -1223.1 mas/yr (Dec)
        let (ra, dec, _) = parse_tycho2_suppl_line(SIRIUS, 2025.5).unwrap();
        let dt = 2025.5 - 1991.25;
        let d_dec_arcsec = (dec - -16.71314306) * 3600.0;
        assert!((d_dec_arcsec - -1.2231 * dt).abs() < 0.01);
        assert!(ra < 101.28854105); // moving in -RA
    }

    #[test]
    fn astap_record_decoding_matches_the_documented_sirius_example() {
        // From unit_star_database.pas: RA bytes C3 06 48, DEC bytes D7 39
        // with section dec9 = -24 (0xE8)
        let ra = (0xC3 as f64 + 0x06 as f64 * 256.0 + 0x48 as f64 * 65536.0) * 360.0
            / ((1u32 << 24) - 1) as f64;
        assert!((ra - 101.2871).abs() < 0.001, "{ra}");
        let dec_int = 0xD7_i32 + 0x39_i32 * 256 + (-24) * 65536;
        let dec = dec_int as f64 * 90.0 / ((128 * 65536) - 1) as f64;
        assert!((dec - -16.71614).abs() < 0.0001, "{dec}");
    }

    #[test]
    fn astap_builder_reads_a_synthetic_area_file() {
        let dir = std::env::temp_dir().join(format!("seiza-astap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut data = vec![b' '; 110];
        data[..9].copy_from_slice(b"TEST FILE");
        data[109] = 5;
        // Section: dec9 = -24, magnitude byte 6 => -1.0
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, (-24i32 + 128) as u8, 6]);
        // Sirius
        data.extend_from_slice(&[0xC3, 0x06, 0x48, 0xD7, 0x39]);
        // Fainter section at dec9 = 0, magnitude byte 116 => 10.0
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 128, 116]);
        data.extend_from_slice(&[0x00, 0x00, 0x80, 0x00, 0x40]); // ra=180°, dec small
        std::fs::write(dir.join("test_0101.1476"), &data).unwrap();

        let out = dir.join("out.bin");
        build_astap(&dir, &out, 2025.0, 21.0, 45).unwrap();

        let catalog = seiza::catalog::TileCatalog::open(&out).unwrap();
        assert_eq!(catalog.star_count(), 2);
        use seiza::catalog::StarCatalog;
        let sirius = catalog.cone_search(101.287, -16.716, 0.01, 5);
        assert_eq!(sirius.len(), 1);
        assert!((sirius[0].mag - -1.0).abs() < 0.01);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_magnitude_free_records() {
        let mut broken = SAMPLE.to_string();
        broken.replace_range(110..135, &" ".repeat(25));
        assert!(parse_tycho2_line(&broken, 2000.0).is_none());
    }
}
