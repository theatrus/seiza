//! What a stack's file says of how its frames were taken.

use seiza_fits::{F32ImageData, FitsImage, HeaderValue, WriteHeaderCard, write_f32_image};
use seiza_stacking::{
    Acquisition, BatchStackOptions, Continue, DrizzleOptions, FitsFrame, FrameDisposition,
    LiveStacker, NormalizationMode, PipelineOptions, StackOptions, write_drizzle_fits_f32,
    write_fits_f32, write_processed_image_fits_f32, write_stack_export_fits_f32,
};
use std::path::{Path, PathBuf};

const WIDTH: usize = 160;
const HEIGHT: usize = 128;

fn star_field() -> Vec<f32> {
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX)
    };
    let mut values = vec![100.0_f32; WIDTH * HEIGHT];
    for value in &mut values {
        *value += (next() * 2.0) as f32;
    }
    for _ in 0..40 {
        let (x, y) = (
            8.0 + next() * (WIDTH as f64 - 16.0),
            8.0 + next() * (HEIGHT as f64 - 16.0),
        );
        let peak = 400.0 + next() * 4_000.0;
        for row in 0..HEIGHT {
            for column in 0..WIDTH {
                let distance = (column as f64 - x).powi(2) + (row as f64 - y).powi(2);
                values[row * WIDTH + column] += (peak * (-distance / 4.5).exp()) as f32;
            }
        }
    }
    values
}

/// A light with `cards`, all of the same field.
fn light(directory: &Path, name: &str, cards: &[(&str, HeaderValue)]) -> PathBuf {
    let path = directory.join(format!("{name}.fits"));
    let cards = cards
        .iter()
        .map(|(key, value)| WriteHeaderCard::new(*key, value.clone()))
        .collect::<Vec<_>>();
    write_f32_image(
        &path,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&star_field()),
        &cards,
    )
    .unwrap();
    path
}

fn seconds(value: f64) -> (&'static str, HeaderValue) {
    ("EXPTIME", HeaderValue::Float(value))
}

fn filter(value: &str) -> (&'static str, HeaderValue) {
    ("FILTER", HeaderValue::String(value.into()))
}

fn options() -> StackOptions {
    StackOptions {
        normalization: NormalizationMode::None,
        ..StackOptions::default()
    }
}

fn open(reference: &Path) -> LiveStacker {
    LiveStacker::open_fits(reference, None, None, None, None, options()).unwrap()
}

/// The stack of `lights`, the first its reference.
fn stacked(lights: &[PathBuf]) -> LiveStacker {
    let mut stacker = open(&lights[0]);
    for light in &lights[1..] {
        let disposition = stacker.push_fits(light).unwrap();
        assert!(
            matches!(disposition, FrameDisposition::Accepted(_)),
            "{disposition:?}"
        );
    }
    stacker
}

fn summary(acquisition: &Acquisition) -> (Option<f64>, Option<&str>) {
    (
        acquisition.total_exposure_seconds,
        acquisition.filter.as_deref(),
    )
}

/// `LIVETIME` and `FILTER` as a file holds them.
fn written(path: &Path) -> (Option<f64>, Option<String>) {
    let image = FitsImage::open(path).unwrap();
    (
        image.header_f64("LIVETIME"),
        image.header_str("FILTER").map(str::to_owned),
    )
}

#[test]
fn a_stack_records_its_total_exposure_and_filter() {
    let directory = tempfile::tempdir().unwrap();
    let at = |name: &str| directory.path().join(name);
    let lights = [
        light(directory.path(), "a", &[seconds(300.0), filter("Ha")]),
        light(directory.path(), "b", &[seconds(300.0), filter("HA")]),
        light(directory.path(), "c", &[seconds(120.0), filter("Ha ")]),
    ];
    let expected = (Some(720.0), Some("Ha"));
    let on_disk = (Some(720.0), Some("Ha".to_owned()));
    let stacker = stacked(&lights);

    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(summary(&snapshot.acquisition), expected);
    write_fits_f32(at("stack.fits"), &snapshot, stacker.reference_headers()).unwrap();
    assert_eq!(written(&at("stack.fits")), on_disk);
    // Neither the length of one frame nor a date of one.
    let file = FitsImage::open(&at("stack.fits")).unwrap();
    assert!(file.header("EXPTIME").is_none() && file.header("DATE-OBS").is_none());

    let export = stacker.export_snapshot().unwrap();
    write_stack_export_fits_f32(at("export.fits"), &export, stacker.reference_headers()).unwrap();
    assert_eq!(written(&at("export.fits")), on_disk);

    let batch = BatchStackOptions::default();
    let drizzle = DrizzleOptions {
        scale: 2,
        ..DrizzleOptions::default()
    };
    let (result, drizzled) = stacker
        .reintegrate_drizzled(&batch, &drizzle, |_, _, _| {})
        .unwrap();
    assert_eq!(summary(&result.snapshot.acquisition), expected);
    write_drizzle_fits_f32(at("drizzle.fits"), &drizzled, 3, 0, &[]).unwrap();
    assert_eq!(written(&at("drizzle.fits")), on_disk);

    // A subset sums only its own frames.
    let some = stacker
        .reintegrate_frames(&lights[1..], &batch, |_, _, _| {})
        .unwrap();
    assert_eq!(
        summary(&some.snapshot.acquisition),
        (Some(420.0), Some("HA"))
    );

    // A processed stack keeps both.
    let source = FitsFrame::open(at("stack.fits")).unwrap();
    write_processed_image_fits_f32(at("processed.fits"), &source.image, &source.headers, &[])
        .unwrap();
    assert_eq!(written(&at("processed.fits")), on_disk);

    let finished = stacker.into_snapshot().unwrap();
    assert_eq!(summary(&finished.acquisition), expected);
}

#[test]
fn every_way_of_pushing_records_the_frame_and_a_rejected_one_is_left_out() {
    let directory = tempfile::tempdir().unwrap();
    let lights = [
        light(directory.path(), "a", &[seconds(60.0), filter("L")]),
        light(directory.path(), "b", &[seconds(60.0), filter("L")]),
        light(directory.path(), "c", &[seconds(30.0), filter("L")]),
    ];
    // Starless: refused.
    let blank = directory.path().join("blank.fits");
    write_f32_image(
        &blank,
        WIDTH,
        HEIGHT,
        F32ImageData::Mono(&vec![100.0; WIDTH * HEIGHT]),
        &[
            WriteHeaderCard::new("EXPTIME", HeaderValue::Float(900.0)),
            WriteHeaderCard::new("FILTER", HeaderValue::String("R".into())),
        ],
    )
    .unwrap();
    let mut stacker = open(&lights[0]);
    stacker.push(FitsFrame::open(&lights[1]).unwrap()).unwrap();
    let report = stacker
        .push_fits_pipelined(
            &[blank, lights[2].clone()],
            &PipelineOptions::default(),
            |_, _| Continue::Yes,
        )
        .unwrap();
    assert_eq!((report.integrated, report.rejected), (1, 1));
    let snapshot = stacker.snapshot().unwrap();
    assert_eq!(snapshot.accepted_frames, 3);
    assert_eq!(summary(&snapshot.acquisition), (Some(150.0), Some("L")));

    // A frame supplied as pixels says nothing, and the rest cannot speak for
    // it.
    let pixels = FitsFrame::open(&lights[1]).unwrap().image;
    let mut stacker =
        LiveStacker::from_prepared_frame(FitsFrame::open(&lights[0]).unwrap(), options()).unwrap();
    assert_eq!(
        summary(&stacker.snapshot().unwrap().acquisition),
        (Some(60.0), Some("L"))
    );
    stacker.push_linear(pixels).unwrap();
    assert_eq!(
        summary(&stacker.snapshot().unwrap().acquisition),
        (None, None)
    );
}

#[test]
fn an_unknown_or_mixed_value_is_left_out() {
    let directory = tempfile::tempdir().unwrap();
    let a = light(directory.path(), "a", &[seconds(60.0), filter("R")]);
    for (name, cards, expected) in [
        ("no exposure", vec![filter("R")], (None, Some("R"))),
        ("no filter", vec![seconds(60.0)], (Some(120.0), None)),
        (
            "other filter",
            vec![seconds(60.0), filter("G")],
            (Some(120.0), None),
        ),
        // A frame that is itself a stack: its total, or else unknown.
        (
            "stack",
            vec![
                seconds(60.0),
                ("LIVETIME", HeaderValue::Float(1_200.0)),
                ("STACKCNT", HeaderValue::Integer(20)),
                filter("R"),
            ],
            (Some(1_260.0), Some("R")),
        ),
        (
            "stack without a total",
            vec![
                seconds(60.0),
                ("NUMFRAME", HeaderValue::Integer(20)),
                filter("R"),
            ],
            (None, Some("R")),
        ),
    ] {
        let b = light(directory.path(), name, &cards);
        let stacker = stacked(&[a.clone(), b]);
        let snapshot = stacker.snapshot().unwrap();
        assert_eq!(summary(&snapshot.acquisition), expected, "{name}");
        let path = directory.path().join(format!("{name} stack.fits"));
        write_fits_f32(&path, &snapshot, stacker.reference_headers()).unwrap();
        assert_eq!(
            written(&path),
            (expected.0, expected.1.map(str::to_owned)),
            "{name}"
        );
    }
}

/// An external reference image lends its grid, never its filter.
#[test]
fn an_external_reference_image_is_not_counted() {
    let directory = tempfile::tempdir().unwrap();
    let reference = light(directory.path(), "lum", &[seconds(600.0), filter("L")]);
    let lights = [
        light(directory.path(), "a", &[seconds(300.0), filter("Ha")]),
        light(directory.path(), "b", &[seconds(300.0), filter("Ha")]),
    ];
    let mut stacker = LiveStacker::with_reference_image(
        FitsFrame::open(&reference).unwrap(),
        FitsFrame::open(&lights[0]).unwrap(),
        Default::default(),
        options(),
    )
    .unwrap();
    assert_eq!(
        summary(&stacker.snapshot().unwrap().acquisition),
        (None, None)
    );
    for light in &lights {
        stacker.push_fits(light).unwrap();
    }
    assert_eq!(
        summary(&stacker.snapshot().unwrap().acquisition),
        (Some(600.0), Some("Ha"))
    );
}

#[test]
fn a_saved_context_keeps_what_its_frames_recorded() {
    let directory = tempfile::tempdir().unwrap();
    let lights = [
        light(directory.path(), "a", &[seconds(300.0), filter("Ha")]),
        light(directory.path(), "b", &[seconds(120.0), filter("Ha")]),
        light(directory.path(), "c", &[seconds(60.0), filter("Ha")]),
    ];
    let stacker = stacked(&lights[..2]);
    let context = directory.path().join("stack.seizastack");
    stacker.save_context(&context).unwrap();
    let mut resumed = LiveStacker::open_context(&context).unwrap();
    assert_eq!(
        resumed.snapshot().unwrap().acquisition,
        stacker.snapshot().unwrap().acquisition
    );
    resumed.push_fits(&lights[2]).unwrap();
    let batch = BatchStackOptions::default();
    let result = resumed.reintegrate(&batch, |_, _, _| {}).unwrap();
    assert_eq!(
        summary(&result.snapshot.acquisition),
        (Some(480.0), Some("Ha"))
    );
    // The report of a resumed stack has every exposure too.
    let exposures = resumed
        .report_context()
        .frames
        .iter()
        .map(|frame| frame.exposure_seconds)
        .collect::<Vec<_>>();
    assert_eq!(exposures, [Some(300.0), Some(120.0), Some(60.0)]);
}

#[test]
fn cards_read_back_and_leave_out_what_fits_cannot_hold() {
    let cards = |acquisition: &Acquisition| {
        acquisition
            .cards()
            .iter()
            .map(|card| (card.keyword().to_owned(), card.value().clone()))
            .collect::<Vec<_>>()
    };
    let read = Acquisition::from_cards(&[
        ("LIVETIME".into(), HeaderValue::Integer(90)),
        ("FILTER".into(), HeaderValue::String(" Ha ".into())),
    ]);
    assert_eq!(summary(&read), (Some(90.0), Some("Ha")));
    assert_eq!(Acquisition::from_cards(&cards(&read)), read);

    let mut greek = read.clone();
    greek.filter = Some("Hα".into());
    assert_eq!(
        cards(&greek),
        [("LIVETIME".to_owned(), HeaderValue::Float(90.0))]
    );
    for nonsense in [-1.0, 0.0, f64::NAN] {
        let read = Acquisition::from_cards(&[("LIVETIME".into(), HeaderValue::Float(nonsense))]);
        assert_eq!(read, Acquisition::default());
    }
    assert_eq!(
        Acquisition::combined([]),
        Acquisition::default(),
        "nothing combined says nothing"
    );
    assert_eq!(
        summary(&Acquisition::combined([&read, &read, &read])),
        (Some(270.0), Some("Ha"))
    );
}
