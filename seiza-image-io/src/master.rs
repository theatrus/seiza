//! What a stacked image records of its integration.
//!
//! Every integrator writes it its own way, and some in ways that contradict
//! each other, so each value says where it came from, and one that cannot be
//! told is left out.

use crate::{Error, FitsHeader, Format, HeaderValue};
use seiza_xisf::{XisfElement, XisfMetadata};
use std::path::Path;

/// A value a file records, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct Recorded<T> {
    pub value: T,
    /// The card, property or line it came from, such as `LIVETIME`, or how it
    /// follows from others.
    pub source: String,
}

impl<T> Recorded<T> {
    fn new(value: T, source: impl Into<String>) -> Self {
        Self {
            value,
            source: source.into(),
        }
    }
}

/// What a stacked image records of its integration, from [`master_info`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MasterInfo {
    /// Exposure over all its frames, in seconds. In a colour image, the
    /// channel with the most.
    pub total_exposure_seconds: Option<Recorded<f64>>,
    /// Frames it was made from.
    pub frames: Option<Recorded<u64>>,
    /// One frame's exposure in seconds: a recorded one, or the total over the
    /// count, which is the mean of frames that differ.
    pub sub_exposure_seconds: Option<Recorded<f64>>,
    pub filter: Option<Recorded<String>>,
    /// The program that integrated it.
    pub integrator: Option<Recorded<String>>,
    /// Output pixels per input pixel, when it was drizzled.
    pub drizzle_scale: Option<Recorded<f64>>,
    /// What to know before relying on the values: records that contradict
    /// each other, or that may count more than was used.
    pub notes: Vec<String>,
}

/// Cards that count an image's frames.
const COUNT_CARDS: [&str; 3] = ["STACKCNT", "NUMFRAME", "NCOMBINE"];
/// How far, as a share, two exposures may differ and still be one.
const SAME_EXPOSURE: f64 = 0.005;

/// Read what the FITS or XISF image at `path` records of its integration,
/// without decoding its pixels: XISF when the extension says so, FITS
/// otherwise. A file that records none of it, such as a single frame's, gives
/// a [`MasterInfo`] without those values.
pub fn read_master_info(path: &Path) -> Result<MasterInfo, Error> {
    if Format::astronomy(path) == Format::Xisf {
        let (cards, metadata) =
            seiza_xisf::read_header_and_metadata(path).map_err(|error| Error::open(path, error))?;
        let header = FitsHeader {
            cards,
            ..FitsHeader::default()
        };
        return Ok(master_info(&header, Some(&metadata)));
    }
    Ok(master_info(&crate::read_header(path)?, None))
}

/// What a header, and an XISF image's metadata, record of an integration.
pub fn master_info(header: &FitsHeader, xisf: Option<&XisfMetadata>) -> MasterInfo {
    let records = Records::new(header, xisf);
    let mut info = MasterInfo {
        filter: records.filter(),
        integrator: records.integrator(),
        drizzle_scale: records.drizzle_scale(),
        ..MasterInfo::default()
    };
    info.frames = records.frames(&mut info.notes);
    records.exposures(&mut info);
    info
}

struct Records<'a> {
    header: &'a FitsHeader,
    xisf: Option<&'a XisfMetadata>,
    /// `HISTORY` and `COMMENT` text. XISF keeps it in keywords' comments.
    lines: Vec<&'a str>,
}

impl<'a> Records<'a> {
    fn new(header: &'a FitsHeader, xisf: Option<&'a XisfMetadata>) -> Self {
        let keywords = xisf
            .into_iter()
            .flat_map(|metadata| &metadata.image_elements)
            .filter(|element| {
                element.local_name() == "FITSKeyword"
                    && matches!(element.attribute("name"), Some("HISTORY" | "COMMENT"))
            })
            .filter_map(|element| element.attribute("comment"));
        let lines = header.history.iter().chain(&header.comments);
        Self {
            header,
            xisf,
            lines: lines.map(String::as_str).chain(keywords).collect(),
        }
    }

    fn card(&self, key: &str) -> Option<&HeaderValue> {
        let mut cards = self.header.cards.iter();
        cards.find(|(name, _)| name == key).map(|(_, value)| value)
    }

    /// A card's positive number, which some programs write as a string.
    fn number(&self, key: &str) -> Option<Recorded<f64>> {
        let value = self.card(key)?;
        let number = value
            .as_f64()
            .or_else(|| value.as_str()?.trim().parse().ok())?;
        (number.is_finite() && number > 0.0).then(|| Recorded::new(number, key))
    }

    fn text(&self, key: &str) -> Option<Recorded<String>> {
        let text = self.card(key)?.as_str()?.trim();
        (!text.is_empty()).then(|| Recorded::new(text.to_string(), key))
    }

    /// The last line `name: value` that PixInsight's integration processes
    /// write, such as `ImageIntegration.numberOfImages: 38`.
    fn process_line(&self, name: &str) -> Option<Recorded<&'a str>> {
        ["ImageIntegration", "DrizzleIntegration", "FastIntegration"]
            .into_iter()
            .find_map(|process| {
                let key = format!("{process}.{name}");
                self.lines.iter().rev().find_map(|line| {
                    let value = line.trim().strip_prefix(&key)?.strip_prefix(':')?;
                    Some(Recorded::new(value.trim(), key.clone()))
                })
            })
    }

    fn property(&self, id: &str) -> Option<&'a XisfElement> {
        self.xisf?.property(id)
    }

    fn filter(&self) -> Option<Recorded<String>> {
        self.text("FILTER").or_else(|| {
            let id = "Instrument:Filter:Name";
            let element = self.property(id)?;
            let value = element.attribute("value").unwrap_or(&element.text).trim();
            (!value.is_empty()).then(|| Recorded::new(value.to_string(), id))
        })
    }

    fn integrator(&self) -> Option<Recorded<String>> {
        // The line a PixInsight process leaves, whatever saved the file since.
        let pixinsight = self.lines.iter().find_map(|line| {
            let program = line.trim().strip_prefix("Integration with ")?;
            let program = program.strip_prefix("PixInsight")?;
            Some(Recorded::new(
                format!("PixInsight{program}"),
                "Integration with PixInsight",
            ))
        });
        if pixinsight.is_some() {
            return pixinsight;
        }
        // Cards only Seiza writes together.
        if self.card("STACKCNT").is_some() && self.card("STACKREJ").is_some() {
            return Some(Recorded::new("Seiza".into(), "STACKREJ"));
        }
        // Only beside a frame count: on one frame these are the program that
        // took or saved it.
        ["PROGRAM", "SOFTWARE"]
            .into_iter()
            .find_map(|key| self.text(key))
            .filter(|_| COUNT_CARDS.iter().any(|key| self.number(key).is_some()))
    }

    fn drizzle_scale(&self) -> Option<Recorded<f64>> {
        self.number("DRIZSCL").or_else(|| {
            let line = self
                .process_line("scale")
                .filter(|line| line.source.starts_with("Drizzle"))?;
            let scale = line.value.parse::<f64>().ok()?;
            (scale.is_finite() && scale > 0.0).then(|| Recorded::new(scale, line.source))
        })
    }

    fn frames(&self, notes: &mut Vec<String>) -> Option<Recorded<u64>> {
        let whole = |found: Recorded<f64>| {
            (found.value >= 1.0 && found.value.fract() == 0.0 && found.value < 1.0e9)
                .then(|| Recorded::new(found.value as u64, found.source))
        };
        let line = self.process_line("numberOfImages").and_then(|line| {
            let count = line.value.parse::<u64>().ok().filter(|&count| count > 0)?;
            Some(Recorded::new(count, line.source))
        });
        let count = line.or_else(|| {
            COUNT_CARDS
                .into_iter()
                .find_map(|key| self.number(key).and_then(whole))
        });
        count.or_else(|| {
            let rows = self.process_table_rows()?;
            notes.push(format!(
                "the frame count is the {} of the process that made the image, which lists rejected frames too",
                rows.source
            ));
            Some(rows)
        })
    }

    /// The rows of the frame table of the last integration process a file
    /// carries. PixInsight keeps processes as XML text in a property.
    fn process_table_rows(&self) -> Option<Recorded<u64>> {
        let properties = self.xisf?.image_elements.iter();
        properties
            .filter(|element| element.local_name() == "Property")
            .filter_map(|element| {
                let text = element.attribute("value").unwrap_or(&element.text);
                last_frame_table(text)
            })
            .next_back()
    }

    /// The exposures. A card `EXPTIME` is one frame's length or the total,
    /// depending on the program.
    fn exposures(&self, info: &mut MasterInfo) {
        let frames = info.frames.as_ref().map(|frames| frames.value as f64);
        let card = ["EXPTIME", "EXPOSURE", "XPOSURE"]
            .into_iter()
            .find_map(|key| self.number(key));
        // Astro Pixel Processor counts in NUMFRAME and puts the total in
        // EXPTIME.
        let counts_total = self.number("NUMFRAME").is_some();
        let (sub, total_card) = if counts_total {
            (None, card)
        } else {
            (card, None)
        };
        let mut total = self.pixinsight_total();
        if let (Some(found), Some(sub), Some(frames)) = (&total, &sub, frames)
            && frames > 1.0
            && same(found.value, sub.value)
        {
            info.notes.push(format!(
                "{} holds one frame's exposure, so it is not used as the total",
                found.source
            ));
            total = None;
        }
        let total = total.or_else(|| self.number("LIVETIME")).or(total_card);
        let from_parts = || {
            let (sub, frames) = (sub.as_ref()?, frames?);
            Some(Recorded::new(
                sub.value * frames,
                format!("{} × frames", sub.source),
            ))
        };
        info.sub_exposure_seconds = sub.clone().or_else(|| {
            let (total, frames) = (total.as_ref()?, frames?);
            Some(Recorded::new(
                total.value / frames,
                format!("{} ÷ frames", total.source),
            ))
        });
        if let (Some(total), Some(parts)) = (&total, from_parts())
            && !same(total.value, parts.value)
        {
            info.notes.push(format!(
                "{} gives {} s and {} gives {} s",
                total.source, total.value, parts.source, parts.value
            ));
        }
        // Without a count, one frame's exposure says nothing of a total.
        info.total_exposure_seconds = total.or_else(from_parts);
    }

    fn pixinsight_total(&self) -> Option<Recorded<f64>> {
        let id = "PCL:TotalExposureTime";
        let values = self.xisf?.vector(id)?;
        let most = values
            .into_iter()
            .filter(|value| value.is_finite() && *value > 0.0)
            .reduce(f64::max)?;
        Some(Recorded::new(most, id))
    }
}

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= SAME_EXPOSURE * a.abs().max(b.abs())
}

/// The frame count of the last integration process in PixInsight's XML for
/// processes: `<instance class="ImageIntegration" ...><table id="images"
/// rows="38">`.
fn last_frame_table(xml: &str) -> Option<Recorded<u64>> {
    let mut found = None;
    for instance in xml.split("<instance ").skip(1) {
        let instance = instance.split("</instance>").next().unwrap_or(instance);
        let class = attribute(instance.split('>').next()?, "class");
        let Some(class @ ("ImageIntegration" | "DrizzleIntegration" | "FastIntegration")) = class
        else {
            continue;
        };
        let rows = instance.split("<table ").skip(1).find_map(|table| {
            let tag = table.split('>').next()?;
            let id = attribute(tag, "id")?;
            (id == "images").then(|| attribute(tag, "rows")?.parse::<u64>().ok())?
        });
        if let Some(rows) = rows.filter(|&rows| rows > 0) {
            found = Some(Recorded::new(rows, format!("{class} frame table")));
        }
    }
    found
}

/// An attribute's value in an XML start tag's text.
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let start = format!(" {name}=\"");
    // The first attribute follows the tag's name, which the caller split off.
    let tag_start = format!("{name}=\"");
    let value = match tag.find(&start) {
        Some(at) => &tag[at + start.len()..],
        None => tag.trim_start().strip_prefix(&tag_start)?,
    };
    value.split('"').next()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(cards: &[(&str, HeaderValue)], history: &[&str]) -> FitsHeader {
        FitsHeader {
            cards: cards
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect(),
            history: history.iter().map(|line| line.to_string()).collect(),
            comments: Vec::new(),
        }
    }

    fn number(value: f64) -> HeaderValue {
        HeaderValue::Float(value)
    }

    fn text(value: &str) -> HeaderValue {
        HeaderValue::String(value.into())
    }

    /// An XISF file with `elements` in its image.
    fn xisf(elements: &str) -> MasterInfo {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("master.xisf");
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><xisf version=\"1.0\" \
             xmlns=\"http://www.pixinsight.com/xisf\"><Image geometry=\"1:1:1\" \
             sampleFormat=\"Float32\" bounds=\"0:1\" colorSpace=\"Gray\" \
             location=\"inline:hex\">00000000{elements}</Image></xisf>"
        );
        let mut file = b"XISF0100".to_vec();
        file.extend_from_slice(&(xml.len() as u32).to_le_bytes());
        file.extend_from_slice(&[0; 4]);
        file.extend_from_slice(xml.as_bytes());
        std::fs::write(&path, file).unwrap();
        read_master_info(&path).unwrap()
    }

    fn history(line: &str) -> String {
        format!("<FITSKeyword name=\"HISTORY\" value=\"\" comment=\"{line}\"/>")
    }

    fn exposure(seconds: &str) -> String {
        format!("<FITSKeyword name=\"EXPTIME\" value=\"{seconds}\" comment=\"\"/>")
    }

    /// Little-endian f64s as PixInsight writes `PCL:TotalExposureTime`.
    fn total(values: &[f64]) -> String {
        let hex: String = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .map(|byte| format!("{byte:02x}"))
            .collect();
        format!(
            "<Property id=\"PCL:TotalExposureTime\" type=\"F64Vector\" length=\"{}\" \
             location=\"inline:hex\">{hex}</Property>",
            values.len()
        )
    }

    fn value<T: Clone>(recorded: &Option<Recorded<T>>) -> Option<(T, &str)> {
        recorded
            .as_ref()
            .map(|recorded| (recorded.value.clone(), recorded.source.as_str()))
    }

    /// The records of a real Weighted Batch Preprocessing master.
    #[test]
    fn a_pixinsight_master_counts_in_its_history() {
        let info = xisf(&format!(
            "<FITSKeyword name=\"COMMENT\" value=\"\" comment=\"Integration with PixInsight 1.8.9-2\"/>\
             <FITSKeyword name=\"FILTER\" value=\"'L-Extreme'\" comment=\"\"/>{}{}{}",
            exposure("300.00"),
            history("Integration with ImageIntegration process"),
            history("ImageIntegration.numberOfImages: 38"),
        ));
        assert_eq!(
            value(&info.frames),
            Some((38, "ImageIntegration.numberOfImages"))
        );
        assert_eq!(value(&info.sub_exposure_seconds), Some((300.0, "EXPTIME")));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((11_400.0, "EXPTIME × frames"))
        );
        assert_eq!(value(&info.filter), Some(("L-Extreme".into(), "FILTER")));
        assert_eq!(
            value(&info.integrator),
            Some(("PixInsight 1.8.9-2".into(), "Integration with PixInsight"))
        );
        assert_eq!(info.drizzle_scale, None);
        assert!(info.notes.is_empty(), "{:?}", info.notes);
    }

    #[test]
    fn pixinsights_total_stands_unless_it_is_one_frames() {
        let count = history("ImageIntegration.numberOfImages: 10");
        // Frames of two lengths, so the count times EXPTIME is not the total.
        let info = xisf(&format!("{}{count}{}", exposure("300"), total(&[2_400.0])));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((2_400.0, "PCL:TotalExposureTime"))
        );
        assert_eq!(info.notes.len(), 1, "{:?}", info.notes);
        assert!(info.notes[0].contains("3000"), "{:?}", info.notes);
        // FastIntegration writes one frame's.
        let count = history("FastIntegration.numberOfImages: 10");
        let info = xisf(&format!("{}{count}{}", exposure("300"), total(&[300.0])));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((3_000.0, "EXPTIME × frames"))
        );
        assert!(info.notes[0].contains("one frame"), "{:?}", info.notes);
        // A colour image's deepest channel, and no count.
        let info = xisf(&total(&[600.0, 900.0, 300.0]));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((900.0, "PCL:TotalExposureTime"))
        );
        assert_eq!((info.frames, info.sub_exposure_seconds), (None, None));
        // One that cannot be decoded.
        let info = xisf(
            "<Property id=\"PCL:TotalExposureTime\" type=\"F64Vector\" length=\"9\" \
             location=\"inline:hex\">00</Property>",
        );
        assert_eq!(info, MasterInfo::default());
    }

    #[test]
    fn a_process_table_counts_when_nothing_else_does() {
        let process = |rows: u32| {
            format!(
                "&lt;instance class=&quot;SubframeSelector&quot; version=&quot;1&quot;&gt;\
                 &lt;table id=&quot;images&quot; rows=&quot;99&quot;&gt;&lt;/table&gt;&lt;/instance&gt;\
                 &lt;instance class=&quot;DrizzleIntegration&quot; version=&quot;1&quot;&gt;\
                 &lt;table id=&quot;images&quot; rows=&quot;{rows}&quot;&gt;&lt;/table&gt;&lt;/instance&gt;\
                 &lt;instance class=&quot;Blink&quot; version=&quot;1&quot;&gt;\
                 &lt;table id=&quot;images&quot; rows=&quot;7&quot;&gt;&lt;/table&gt;&lt;/instance&gt;"
            )
        };
        let property = format!(
            "<Property id=\"PixInsight:ProcessingHistory\" type=\"String\">{}</Property>",
            process(41)
        );
        let drizzle = history("DrizzleIntegration.scale: 2.00");
        let info = xisf(&format!("{property}{drizzle}"));
        assert_eq!(
            value(&info.frames),
            Some((41, "DrizzleIntegration frame table"))
        );
        assert_eq!(
            value(&info.drizzle_scale),
            Some((2.0, "DrizzleIntegration.scale"))
        );
        assert!(info.notes[0].contains("rejected"), "{:?}", info.notes);
        let counted = history("ImageIntegration.numberOfImages: 38");
        let info = xisf(&format!("{property}{counted}"));
        assert_eq!(info.frames.unwrap().value, 38);
        assert!(info.notes.is_empty(), "{:?}", info.notes);
    }

    #[test]
    fn fits_masters_count_and_time_each_their_own_way() {
        // Astro Pixel Processor: EXPTIME is the total.
        let info = master_info(
            &cards(
                &[("NUMFRAME", number(67.0)), ("EXPTIME", number(20_100.0))],
                &[],
            ),
            None,
        );
        assert_eq!(value(&info.frames), Some((67, "NUMFRAME")));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((20_100.0, "EXPTIME"))
        );
        assert_eq!(
            value(&info.sub_exposure_seconds),
            Some((300.0, "EXPTIME ÷ frames"))
        );
        // Siril.
        let info = master_info(
            &cards(
                &[
                    ("STACKCNT", HeaderValue::Integer(20)),
                    ("EXPTIME", number(180.0)),
                    ("LIVETIME", number(3_600.0)),
                    ("PROGRAM", text("Siril 1.2.0")),
                ],
                &[],
            ),
            None,
        );
        assert_eq!(value(&info.frames), Some((20, "STACKCNT")));
        assert_eq!(
            value(&info.total_exposure_seconds),
            Some((3_600.0, "LIVETIME"))
        );
        assert_eq!(value(&info.sub_exposure_seconds), Some((180.0, "EXPTIME")));
        assert_eq!(
            value(&info.integrator),
            Some(("Siril 1.2.0".into(), "PROGRAM"))
        );
        assert!(info.notes.is_empty(), "{:?}", info.notes);
        // Seiza, which records no exposure.
        let info = master_info(
            &cards(
                &[
                    ("STACKCNT", HeaderValue::Integer(349)),
                    ("STACKREJ", HeaderValue::Integer(0)),
                    ("DRIZSCL", HeaderValue::Integer(2)),
                ],
                &[],
            ),
            None,
        );
        assert_eq!(value(&info.frames), Some((349, "STACKCNT")));
        assert_eq!(value(&info.integrator), Some(("Seiza".into(), "STACKREJ")));
        assert_eq!(value(&info.drizzle_scale), Some((2.0, "DRIZSCL")));
        assert_eq!(
            (info.total_exposure_seconds, info.sub_exposure_seconds),
            (None, None)
        );
        // PixInsight's lines in FITS.
        let info = master_info(
            &cards(
                &[("EXPTIME", number(60.0))],
                &["ImageIntegration.numberOfImages: 12"],
            ),
            None,
        );
        assert_eq!(info.total_exposure_seconds.unwrap().value, 720.0);
    }

    #[test]
    fn a_single_frame_claims_no_integration() {
        let info = master_info(
            &cards(
                &[
                    ("EXPTIME", number(300.0)),
                    ("FILTER", text("Ha")),
                    ("PROGRAM", text("N.I.N.A.")),
                    ("STACKCNT", number(0.0)),
                    ("DRIZSCL", text("none")),
                ],
                &["ImageIntegration.numberOfImages: many"],
            ),
            None,
        );
        assert_eq!(value(&info.sub_exposure_seconds), Some((300.0, "EXPTIME")));
        assert_eq!(info.frames, None);
        assert_eq!(info.total_exposure_seconds, None);
        assert_eq!(info.drizzle_scale, None);
        assert_eq!(info.integrator, None);
        let none = master_info(&FitsHeader::default(), None);
        assert_eq!(none, MasterInfo::default());
    }
}
