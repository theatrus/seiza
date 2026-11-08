use seiza::Wcs;
use seiza::catalog::TileSetBuilder;
use seiza_fits::{F32ImageData, HeaderValue, WriteHeaderCard};
use seiza_image_io::{SaveOptions, save_f32};
use std::path::PathBuf;
use std::process::{Command, Output};

struct Field {
    directory: tempfile::TempDir,
    image: PathBuf,
    catalog: PathBuf,
}

impl Field {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let image = directory.path().join("field.fits");
        let wcs = Wcs::from_center_scale_rotation((10.0, 20.0), (31.5, 23.5), 1.5, 15.0, false);
        let headers = wcs
            .fits_header_cards()
            .into_iter()
            .map(|(key, value)| {
                let value = match value {
                    seiza::FitsCardValue::Text(value) => HeaderValue::String(value.into()),
                    seiza::FitsCardValue::Integer(value) => HeaderValue::Integer(i64::from(value)),
                    seiza::FitsCardValue::Number(value) => HeaderValue::Float(value),
                };
                WriteHeaderCard::new(&key, value)
            })
            .collect::<Vec<_>>();
        save_f32(
            &image,
            64,
            48,
            F32ImageData::Mono(&vec![100.0; 64 * 48]),
            &SaveOptions {
                headers: &headers,
                ..SaveOptions::default()
            },
        )
        .unwrap();
        let catalog = directory.path().join("stars.bin");
        let mut stars = TileSetBuilder::new(2, 2000.0, "output collision regression");
        stars.add(10.0, 20.0, 5.0);
        stars.write_to(&catalog).unwrap();
        Self {
            directory,
            image,
            catalog,
        }
    }

    fn command(&self, subcommand: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_seiza"));
        command
            .arg(subcommand)
            .arg(&self.image)
            .args(["--detection-backend", "f32"]);
        if subcommand != "detect" {
            command.arg("--data").arg(&self.catalog);
        }
        command
    }
}

#[test]
fn blind_wcs_output_cannot_replace_the_input_image() {
    let field = Field::new();
    let before = std::fs::read(&field.image).unwrap();
    let result = field
        .command("solve-blind")
        .arg("--wcs")
        .arg(&field.image)
        .output()
        .unwrap();
    let after = std::fs::read(&field.image).unwrap();
    assert!(
        after == before,
        "input changed from {} to {} bytes; stdout: {}; stderr: {}",
        before.len(),
        after.len(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("same file"));
}

fn assert_collision(result: &Output, role: &str) {
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(
        error.contains("same file") && error.contains(role),
        "{error}"
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("stars detected"));
}

#[test]
fn detection_and_solve_annotations_cannot_replace_the_input() {
    for command in ["detect", "solve", "solve-blind"] {
        let field = Field::new();
        let before = std::fs::read(&field.image).unwrap();
        let result = field
            .command(command)
            .arg("--annotate")
            .arg(&field.image)
            .output()
            .unwrap();
        assert_collision(&result, "annotation output");
        assert_eq!(std::fs::read(&field.image).unwrap(), before);
    }
}

#[test]
fn sky_maps_cannot_replace_the_input() {
    for command in ["solve", "solve-blind"] {
        let field = Field::new();
        let before = std::fs::read(&field.image).unwrap();
        let result = field
            .command(command)
            .arg("--sky-map")
            .arg(&field.image)
            .output()
            .unwrap();
        assert_collision(&result, "sky-map output");
        assert_eq!(std::fs::read(&field.image).unwrap(), before);
    }
}

#[test]
fn blind_wcs_output_cannot_replace_the_catalog_or_index() {
    for role in ["star catalog", "blind index"] {
        let field = Field::new();
        let mut command = field.command("solve-blind");
        let path = if role == "star catalog" {
            field.catalog.clone()
        } else {
            let path = field.directory.path().join("index.bin");
            std::fs::write(&path, b"index must not be opened or overwritten").unwrap();
            command.arg("--index").arg(&path);
            path
        };
        let before = std::fs::read(&path).unwrap();
        let result = command.arg("--wcs").arg(&path).output().unwrap();
        assert_collision(&result, role);
        assert_eq!(std::fs::read(path).unwrap(), before);
    }
}

#[test]
fn solve_outputs_must_have_distinct_destinations() {
    for (command, flags) in [
        ("solve", ["--annotate", "--sky-map"]),
        ("solve-blind", ["--wcs", "--annotate"]),
        ("solve-blind", ["--wcs", "--sky-map"]),
        ("solve-blind", ["--annotate", "--sky-map"]),
    ] {
        let field = Field::new();
        let child = field.directory.path().join("child");
        std::fs::create_dir(&child).unwrap();
        let output = field.directory.path().join("output.fits");
        let alias = child.join("../output.fits");
        let before = std::fs::read(&field.image).unwrap();
        let result = field
            .command(command)
            .arg(flags[0])
            .arg(&output)
            .arg(flags[1])
            .arg(alias)
            .output()
            .unwrap();
        assert_collision(&result, "output");
        assert!(!output.exists());
        assert_eq!(std::fs::read(&field.image).unwrap(), before);
    }
}

#[test]
fn hard_link_outputs_cannot_replace_the_input_or_share_a_destination() {
    let field = Field::new();
    let alias = field.directory.path().join("linked.fits");
    std::fs::hard_link(&field.image, &alias).unwrap();
    let before = std::fs::read(&field.image).unwrap();
    let result = field
        .command("solve-blind")
        .arg("--wcs")
        .arg(&alias)
        .output()
        .unwrap();
    assert_collision(&result, "image input");
    assert_eq!(std::fs::read(&field.image).unwrap(), before);
    assert_eq!(std::fs::read(&alias).unwrap(), before);

    let output = field.directory.path().join("output.fits");
    let linked_output = field.directory.path().join("linked-output.fits");
    std::fs::write(&output, b"existing output").unwrap();
    std::fs::hard_link(&output, &linked_output).unwrap();
    let result = field
        .command("solve-blind")
        .arg("--wcs")
        .arg(&output)
        .arg("--annotate")
        .arg(&linked_output)
        .output()
        .unwrap();
    assert_collision(&result, "output");
    assert_eq!(std::fs::read(output).unwrap(), b"existing output");
    assert_eq!(std::fs::read(linked_output).unwrap(), b"existing output");
}

#[cfg(unix)]
#[test]
fn symlink_outputs_cannot_replace_the_input() {
    let field = Field::new();
    let alias = field.directory.path().join("linked.fits");
    std::os::unix::fs::symlink(&field.image, &alias).unwrap();
    let before = std::fs::read(&field.image).unwrap();
    let result = field
        .command("solve-blind")
        .arg("--wcs")
        .arg(&alias)
        .output()
        .unwrap();
    assert_collision(&result, "image input");
    assert_eq!(std::fs::read(&field.image).unwrap(), before);
    assert_eq!(std::fs::read(alias).unwrap(), before);
}

#[test]
fn distinct_outputs_still_write_the_wcs_and_annotations() {
    let field = Field::new();
    let wcs = field.directory.path().join("solution.wcs");
    let annotation = field.directory.path().join("annotation.png");
    let before = std::fs::read(&field.image).unwrap();
    let result = field
        .command("solve-blind")
        .arg("--wcs")
        .arg(&wcs)
        .arg("--annotate")
        .arg(&annotation)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let header = std::fs::read(wcs).unwrap();
    assert!(header.starts_with(b"SIMPLE"));
    assert!(header.windows(7).any(|value| value == b"CD1_1  "));
    assert_eq!(
        seiza_image_io::open_raster(&annotation)
            .unwrap()
            .coordinates
            .oriented_dimensions,
        (64, 48)
    );
    assert_eq!(std::fs::read(&field.image).unwrap(), before);
}
