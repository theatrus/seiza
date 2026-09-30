//! The seiza half of the PixInsight conformance run in
//! `conformance/pixinsight`: reads every file PixInsight wrote, checks its
//! samples against the generating pattern, then writes round-trip copies and
//! files of its own for PixInsight to check.
//!
//! Usage: `pixinsight_probe <from_pi dir> <to_pi dir>`

use seiza_fits::{F32ImageData, Pixels};
use seiza_xisf::{ChecksumAlgorithm, CompressionCodec, WriteCompression, WriteOptions};
use std::fmt::Write as _;
use std::path::Path;

fn max_k(bits: u32, float: bool) -> f64 {
    if float {
        65535.0
    } else {
        2f64.powi(bits as i32) - 1.0
    }
}

fn sample_k(x: usize, y: usize, c: usize, seed: u64, mk: f64) -> f64 {
    ((x as u64 * 7919 + y as u64 * 104729 + c as u64 * 1299709 + seed * 31) % (mk as u64 + 1))
        as f64
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let (from_pi, to_pi) = (Path::new(&args[1]), Path::new(&args[2]));
    std::fs::create_dir_all(to_pi).unwrap();
    let manifest = std::fs::read_to_string(from_pi.join("manifest.txt")).unwrap();
    let mut out_manifest = String::new();
    let (mut passed, mut failed) = (0, 0);
    for line in manifest.lines().filter(|line| !line.is_empty()) {
        let fields = line.split('|').collect::<Vec<_>>();
        let (name, index) = match fields[0].split_once('#') {
            Some((name, index)) => (name, index.parse::<usize>().unwrap()),
            None => (fields[0], 0),
        };
        let (w, h, ch): (usize, usize, usize) = (
            fields[1].parse().unwrap(),
            fields[2].parse().unwrap(),
            fields[3].parse().unwrap(),
        );
        let (bits, float, seed): (u32, bool, u64) = (
            fields[4].parse().unwrap(),
            fields[5] == "1",
            fields[6].parse().unwrap(),
        );
        let lab = fields.get(8) == Some(&"lab");
        let path = from_pi.join(format!("{name}.xisf"));
        let read = match seiza_xisf::read_image_at(&path, index) {
            Ok(read) => read,
            Err(error) => {
                failed += 1;
                println!("FAIL {}: read error {error}", fields[0]);
                continue;
            }
        };
        let image = &read.image;
        let mk = max_k(bits, float);
        let samples: Vec<f64> = match &image.pixels {
            Pixels::U8(v) => v.iter().map(|&v| f64::from(v)).collect(),
            Pixels::U16(v) => v.iter().map(|&v| f64::from(v)).collect(),
            Pixels::F32(v) => v.iter().map(|&v| f64::from(v)).collect(),
            Pixels::F64(v) => v.clone(),
            Pixels::I32(v) => v.iter().map(|&v| f64::from(v)).collect(),
        };
        let mut max_error = 0f64;
        let mut mismatches = 0;
        for c in 0..ch {
            for y in 0..h {
                for x in 0..w {
                    let k = sample_k(x, y, c, seed, mk);
                    let expected = if lab {
                        k / mk
                    } else if float && bits == 32 {
                        f64::from((k / mk) as f32)
                    } else if float {
                        k / mk
                    } else {
                        k
                    };
                    let actual = samples[(c * h + y) * w + x];
                    let error = (actual - expected).abs();
                    max_error = max_error.max(error);
                    if error > if lab { 1e-4 } else { 0.0 } {
                        mismatches += 1;
                    }
                }
            }
        }
        let geometry_ok = (image.width, image.height, image.planes) == (w, h, ch);
        let info = &read.info;
        let location = match &info.location {
            seiza_xisf::BlockLocation::Attachment { .. } => "attached",
            seiza_xisf::BlockLocation::Inline { .. } => "inline",
            seiza_xisf::BlockLocation::Embedded { .. } => "embedded",
            seiza_xisf::BlockLocation::External { .. } => "external",
        };
        let ok = geometry_ok && mismatches == 0 && read.metadata.dropped.is_empty();
        if ok {
            passed += 1
        } else {
            failed += 1
        }
        println!(
            "{} {:28} {:?} {location:8} {:?} max_error={max_error:.3e} mismatches={mismatches} dropped={:?}",
            if ok { "ok  " } else { "FAIL" },
            fields[0],
            info.sample_format,
            info.compression
                .as_ref()
                .map(|c| (c.codec, c.shuffled_item_bytes)),
            read.metadata.dropped
        );
        if name.starts_with("props") {
            for element in &read.metadata.image_elements {
                println!(
                    "      {} {:?} type={:?} value={:?} text={:?} location={:?} block={:?}",
                    element.name,
                    element.attribute("id").or(element.attribute("name")),
                    element.attribute("type"),
                    element.attribute("value"),
                    element.text.chars().take(60).collect::<String>(),
                    element.attribute("location"),
                    element.block.as_ref().map(Vec::len)
                );
            }
            println!(
                "      headers: {:?}",
                image
                    .headers
                    .iter()
                    .map(|(k, v)| format!("{k}={v:?}"))
                    .collect::<Vec<_>>()
            );
        }
        if lab || index != 0 {
            continue;
        }
        // Round trip: the samples normalized to [0,1], with every carried
        // field, compressed and checksummed.
        let normalized = samples
            .iter()
            .map(|&v| if float { v as f32 } else { (v / mk) as f32 })
            .collect::<Vec<_>>();
        let data = if ch == 3 {
            F32ImageData::RgbPlanar(&normalized)
        } else {
            F32ImageData::Mono(&normalized)
        };
        let options = WriteOptions {
            compression: Some(WriteCompression::recommended()),
            checksum: Some(ChecksumAlgorithm::Sha256),
            metadata: Some(&read.metadata),
            bounds: Some((0.0, 1.0)),
        };
        let out = format!("rt_{name}");
        seiza_xisf::write_f32_image_with_options(
            to_pi.join(format!("{out}.xisf")),
            w,
            h,
            data,
            &[],
            &options,
        )
        .unwrap();
        let _ = writeln!(
            out_manifest,
            "{out}|{w}|{h}|{ch}|{bits}|{}|{seed}|{name}",
            u8::from(float)
        );
    }
    // seiza's own files, one per codec and checksum.
    let codecs = [
        ("none", None),
        ("zlib", Some((CompressionCodec::Zlib, false))),
        ("zlibsh", Some((CompressionCodec::Zlib, true))),
        ("lz4", Some((CompressionCodec::Lz4, false))),
        ("lz4sh", Some((CompressionCodec::Lz4, true))),
        ("zstd", Some((CompressionCodec::Zstd, false))),
        ("zstdsh", Some((CompressionCodec::Zstd, true))),
    ];
    let sums = [
        ChecksumAlgorithm::Sha1,
        ChecksumAlgorithm::Sha256,
        ChecksumAlgorithm::Sha512,
    ];
    for (i, (label, codec)) in codecs.into_iter().enumerate() {
        for ch in [1_usize, 3] {
            let (w, h, seed) = (57_usize, 43_usize, 500 + i as u64 * 2 + ch as u64);
            let mut interleaved = vec![0f32; w * h * ch];
            for c in 0..ch {
                for y in 0..h {
                    for x in 0..w {
                        interleaved[(y * w + x) * ch + c] =
                            (sample_k(x, y, c, seed, 65535.0) / 65535.0) as f32;
                    }
                }
            }
            let data = if ch == 3 {
                F32ImageData::RgbInterleaved(&interleaved)
            } else {
                F32ImageData::Mono(&interleaved)
            };
            let options = WriteOptions {
                compression: codec.map(|(codec, byte_shuffle)| WriteCompression {
                    codec,
                    byte_shuffle,
                    level: None,
                }),
                checksum: Some(sums[i % 3]),
                metadata: None,
                bounds: Some((0.0, 1.0)),
            };
            let out = format!("seiza_{label}_{ch}");
            let headers = [seiza_fits::WriteHeaderCard::new(
                "OBJECT",
                seiza_fits::HeaderValue::String("seiza probe".into()),
            )];
            seiza_xisf::write_f32_image_with_options(
                to_pi.join(format!("{out}.xisf")),
                w,
                h,
                data,
                &headers,
                &options,
            )
            .unwrap();
            let _ = writeln!(out_manifest, "{out}|{w}|{h}|{ch}|32|1|{seed}|");
        }
    }
    std::fs::write(to_pi.join("manifest.txt"), out_manifest).unwrap();
    println!("read checks: {passed} passed, {failed} failed");
}
