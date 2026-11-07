//! Malformed headers must return errors through native readers, including
//! when used on an ordinary Rust worker thread with a 2 MiB stack.

use seiza_xisf::XisfError;
use std::path::Path;

fn image_bytes(depth: usize, suffix: &str) -> Vec<u8> {
    let mut xml = String::from(
        "<xisf version=\"1.0\"><Image geometry=\"1:1:1\" sampleFormat=\"UInt8\" \
         colorSpace=\"Gray\" location=\"inline:hex\">7f</Image>",
    );
    xml.push_str(&"<Extension>".repeat(depth));
    xml.push_str(&"</Extension>".repeat(depth));
    xml.push_str("</xisf>");
    xml.push_str(suffix);
    let mut bytes = b"XISF0100".to_vec();
    bytes.extend_from_slice(&(xml.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(xml.as_bytes());
    bytes
}

#[test]
fn excessive_nesting_returns_an_error_without_aborting() {
    const CHILD: &str = "SEIZA_XISF_NESTING_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let result = seiza_xisf::from_bytes(&image_bytes(16384, ""));
                assert!(matches!(result, Err(XisfError::Malformed(_))));
            })
            .unwrap()
            .join()
            .unwrap();
        return;
    }
    // A stack overflow aborts the entire process; isolate the regression so
    // it is reported as a failed test rather than killing the suite.
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "excessive_nesting_returns_an_error_without_aborting",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "reader subprocess exited with {}:\n{}\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn unfinished_trailing_signature_is_refused_by_native_readers() {
    let bytes = image_bytes(0, "<Signature><SignedInfo>");
    assert!(matches!(
        seiza_xisf::from_bytes(&bytes),
        Err(XisfError::Malformed(_))
    ));
    assert!(matches!(
        seiza_xisf::read_image_from_bytes(&bytes, 0),
        Err(XisfError::Malformed(_))
    ));
    type Reader = fn(&Path) -> Result<(), XisfError>;
    let readers: [(&str, Reader); 5] = [
        ("inspect", |path| seiza_xisf::inspect(path).map(|_| ())),
        ("header", |path| seiza_xisf::read_header(path).map(|_| ())),
        ("pixels", |path| seiza_xisf::open(path).map(|_| ())),
        ("metadata", |path| seiza_xisf::read_image(path).map(|_| ())),
        ("window", |path| {
            seiza_xisf::read_window(path, 0, 0, 1, 1).map(|_| ())
        }),
    ];
    for extension in [".xisf", ".xish"] {
        let file = tempfile::Builder::new()
            .suffix(extension)
            .tempfile()
            .unwrap();
        if extension == ".xish" {
            let mut header = b"<?xml version=\"1.0\"?>".to_vec();
            header.extend_from_slice(&bytes[16..]);
            std::fs::write(file.path(), header).unwrap();
        } else {
            std::fs::write(file.path(), &bytes).unwrap();
        }
        for (name, read) in readers {
            assert!(
                matches!(read(file.path()), Err(XisfError::Malformed(_))),
                "{name} {extension}"
            );
        }
    }
}

#[test]
fn complete_detached_signature_keeps_the_image_readable() {
    let bytes = image_bytes(16, "<Signature><SignedInfo/></Signature>");
    let image = seiza_xisf::from_bytes(&bytes).unwrap();
    assert_eq!((image.width, image.height, image.planes), (1, 1, 1));
    assert!(matches!(image.pixels, seiza_fits::Pixels::U8(values) if values == [127]));
}

fn nested_element(depth: usize) -> seiza_xisf::XisfElement {
    let mut element = seiza_xisf::XisfElement {
        name: "Extension".into(),
        ..Default::default()
    };
    for _ in 1..depth {
        element = seiza_xisf::XisfElement {
            name: "Extension".into(),
            children: vec![element],
            ..Default::default()
        };
    }
    element
}

fn referenced_image_bytes(depth: usize) -> Vec<u8> {
    let bytes = image_bytes(depth, "");
    let xml = std::str::from_utf8(&bytes[16..])
        .unwrap()
        .replacen("<Extension>", "<Extension uid=\"shared\">", 1)
        .replace("7f</Image>", "7f<Reference ref=\"shared\"/></Image>");
    let mut bytes = b"XISF0100".to_vec();
    bytes.extend_from_slice(&(xml.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(xml.as_bytes());
    bytes
}

#[test]
fn resolved_references_obey_the_image_nesting_limit() {
    // A root child moves one level deeper when used inside an Image.
    assert!(matches!(
        seiza_xisf::read_image_from_bytes(&referenced_image_bytes(127), 0),
        Err(XisfError::Malformed(_))
    ));
    let read = seiza_xisf::read_image_from_bytes(&referenced_image_bytes(126), 0).unwrap();
    let mut output = Vec::new();
    seiza_xisf::write_f32_image_to_with_options(
        &mut output,
        1,
        1,
        seiza_fits::F32ImageData::Mono(&[127.0]),
        &[],
        &seiza_xisf::WriteOptions {
            metadata: Some(&read.metadata),
            ..Default::default()
        },
    )
    .unwrap();
    let again = seiza_xisf::read_image_from_bytes(&output, 0).unwrap();
    assert_eq!(read.metadata.image_elements, again.metadata.image_elements);
}

#[test]
fn writer_rejects_excessive_metadata_nesting_before_writing() {
    for (location, parent_depth) in [("root", 1), ("image", 2), ("unit", 2)] {
        let mut metadata = seiza_xisf::XisfMetadata {
            source_geometry: (1, 1, 1),
            ..Default::default()
        };
        let element = nested_element(129 - parent_depth);
        match location {
            "root" => metadata.root_elements.push(element),
            "image" => metadata.image_elements.push(element),
            "unit" => metadata.unit_properties.push(element),
            _ => unreachable!(),
        }
        let mut output = Vec::new();
        let result = seiza_xisf::write_f32_image_to_with_options(
            &mut output,
            1,
            1,
            seiza_fits::F32ImageData::Mono(&[0.5]),
            &[],
            &seiza_xisf::WriteOptions {
                metadata: Some(&metadata),
                ..Default::default()
            },
        );
        assert!(matches!(result, Err(XisfError::Malformed(_))));
        assert!(
            output.is_empty(),
            "invalid metadata must be refused before writing"
        );
    }
}

#[test]
fn deeply_nested_supported_metadata_round_trips() {
    // Root plus 127 extension elements is the supported 128-level boundary.
    let read = seiza_xisf::read_image_from_bytes(&image_bytes(127, ""), 0).unwrap();
    let mut output = Vec::new();
    seiza_xisf::write_f32_image_to_with_options(
        &mut output,
        1,
        1,
        seiza_fits::F32ImageData::Mono(&[127.0]),
        &[],
        &seiza_xisf::WriteOptions {
            metadata: Some(&read.metadata),
            ..Default::default()
        },
    )
    .unwrap();
    let again = seiza_xisf::read_image_from_bytes(&output, 0).unwrap();
    assert_eq!(read.metadata.root_elements, again.metadata.root_elements);
    assert!(matches!(again.image.pixels, seiza_fits::Pixels::F32(values) if values == [127.0]));
}
