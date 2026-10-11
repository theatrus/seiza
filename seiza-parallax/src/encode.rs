//! Writing frames out as a video or as numbered images.

use image::RgbImage;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

/// Why frames could not be written.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{action} {}: {source}", path.display())]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ffmpeg could not start: {0}")]
    FfmpegMissing(std::io::Error),
    #[error("ffmpeg failed: {0}")]
    Ffmpeg(String),
    /// The output cannot be written: not a video's name, or no folder.
    #[error("{0}")]
    Output(String),
    #[error("a frame is {got:?}, not the video's {expected:?}")]
    FrameSize {
        got: (u32, u32),
        expected: (u32, u32),
    },
    #[error("frames for H.264 need even sides; got {0}x{1}")]
    OddSize(u32, u32),
    #[error("could not write PNG frame {}: {source}", path.display())]
    Png {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    /// A sink of the caller's own refused a frame.
    #[error("the frame sink stopped: {0}")]
    Sink(String),
    #[error("OpenH264 encoding failed: {0}")]
    #[cfg(feature = "openh264")]
    OpenH264(String),
    #[error("MP4 muxing failed: {0}")]
    #[cfg(feature = "openh264")]
    Mp4(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Where rendered frames go.
pub trait FrameSink {
    /// Add the next frame.
    fn push(&mut self, frame: &RgbImage) -> Result<()>;
    /// Finish the output.
    fn finish(self: Box<Self>) -> Result<()>;
}

/// A sink that hands each frame to a function: the way to feed frames to
/// an encoder of the caller's own, such as a platform's video encoder. An
/// error from the function stops the video.
pub struct FrameFn<F> {
    push: F,
}

impl<F: FnMut(&RgbImage) -> std::result::Result<(), String>> FrameFn<F> {
    pub fn new(push: F) -> Self {
        Self { push }
    }
}

impl<F: FnMut(&RgbImage) -> std::result::Result<(), String>> FrameSink for FrameFn<F> {
    fn push(&mut self, frame: &RgbImage) -> Result<()> {
        (self.push)(frame).map_err(Error::Sink)
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

/// Common frame sizes by name, landscape; add `-portrait` to a name for
/// the same size on end.
pub const FRAME_SIZES: [(&str, (usize, usize)); 5] = [
    ("720p", (1280, 720)),
    ("1080p", (1920, 1080)),
    ("1440p", (2560, 1440)),
    ("4k", (3840, 2160)),
    ("2160p", (3840, 2160)),
];

/// A frame size: a name from [`FRAME_SIZES`] such as `1080p` or `4k`, the
/// same with `-portrait` for its tall form (`1080p-portrait` is 1080 wide
/// and 1920 high), or `WIDTHxHEIGHT`. Sides must be even, as H.264 needs,
/// and at least 16.
pub fn parse_frame_size(text: &str) -> std::result::Result<(usize, usize), String> {
    let lower = text.trim().to_ascii_lowercase();
    let (name, portrait) = match lower.strip_suffix("-portrait") {
        Some(name) => (name, true),
        None => (lower.as_str(), false),
    };
    if let Some(&(_, (width, height))) = FRAME_SIZES.iter().find(|(known, _)| *known == name) {
        return Ok(if portrait {
            (height, width)
        } else {
            (width, height)
        });
    }
    let names: Vec<&str> = FRAME_SIZES.iter().map(|(name, _)| *name).collect();
    let (width, height) = lower.split_once('x').ok_or_else(|| {
        format!(
            "expected WIDTHxHEIGHT or one of {} (with -portrait for tall); got {text}",
            names.join(", ")
        )
    })?;
    let number = |part: &str| {
        part.trim()
            .parse::<usize>()
            .map_err(|error| format!("{part}: {error}"))
    };
    let size = (number(width)?, number(height)?);
    if size.0 < 16 || size.1 < 16 || size.0 % 2 != 0 || size.1 % 2 != 0 {
        return Err(format!("{text}: sides must be even and at least 16"));
    }
    Ok(size)
}

/// Video settings shared by the encoders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Target bitrate for encoders that take one, bits per second.
    pub bitrate: u32,
}

fn check_size(frame: &RgbImage, settings: &VideoSettings) -> Result<()> {
    let got = frame.dimensions();
    let expected = (settings.width, settings.height);
    if got != expected {
        return Err(Error::FrameSize { got, expected });
    }
    Ok(())
}

/// An MP4, MOV or Matroska video through an `ffmpeg` executable, fed raw
/// RGB on its standard input. Dropped before [`FrameSink::finish`] (the
/// video stopped or failed), it stops ffmpeg and removes the unfinished
/// file, which would otherwise play as a shorter, finished-looking video.
pub struct FfmpegSink {
    child: Child,
    stdin: Option<ChildStdin>,
    settings: VideoSettings,
    output: PathBuf,
    finished: bool,
}

/// ffmpeg's muxer for `output`'s extension: MP4 (`.mp4`, `.m4v`), QuickTime
/// (`.mov`) or Matroska (`.mkv`). Named outright, so a slip such as `-o
/// image.tif` cannot overwrite an image with a video frame, and a name
/// without an extension gets a plain answer.
fn container(output: &Path) -> Result<&'static str> {
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("mp4" | "m4v") => Ok("mp4"),
        Some("mov") => Ok("mov"),
        Some("mkv") => Ok("matroska"),
        _ => Err(Error::Output(format!(
            "{} is not a video file name: end it in .mp4, .m4v, .mov or .mkv",
            output.display()
        ))),
    }
}

/// The video codec ffmpeg writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Codec {
    /// H.264, which every player takes: libx264, else libopenh264.
    #[default]
    H264,
    /// HEVC (H.265) with libx265: about a fifth smaller for the same look,
    /// and slower to encode, tagged so Apple's players open it.
    Hevc,
}

impl Codec {
    /// The codec named `h264` (or `avc`, `x264`) or `hevc` (or `h265`,
    /// `x265`).
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "h264" | "h.264" | "avc" | "x264" => Ok(Self::H264),
            "hevc" | "h265" | "h.265" | "x265" => Ok(Self::Hevc),
            other => Err(format!("the codec is h264 or hevc; got {other}")),
        }
    }
}

impl FfmpegSink {
    /// Whether `ffmpeg` (or `executable`) can write `output` in `codec`:
    /// it runs and has the encoder, the name is a video's, and its folder
    /// is there. Worth asking before the slow work a video needs.
    pub fn check(executable: &Path, output: &Path, codec: Codec) -> Result<()> {
        container(output)?;
        let folder = output
            .parent()
            .filter(|folder| !folder.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if !folder.is_dir() {
            return Err(Error::Output(format!(
                "{} has no folder {} to go in",
                output.display(),
                folder.display()
            )));
        }
        Self::encoder(executable, codec, 0).map(|_| ())
    }

    /// The arguments choosing `codec`'s encoder in the `executable`'s
    /// ffmpeg, at `bitrate` where it takes one.
    fn encoder(executable: &Path, codec: Codec, bitrate: u32) -> Result<Vec<String>> {
        let encoders = Command::new(executable)
            .args(["-hide_banner", "-encoders"])
            .stderr(Stdio::null())
            .output()
            .map_err(Error::FfmpegMissing)?;
        let encoders = String::from_utf8_lossy(&encoders.stdout);
        let has = |name: &str| encoders.split_whitespace().any(|word| word == name);
        let bitrate = bitrate.to_string();
        let codec: Vec<&str> = if codec == Codec::Hevc {
            if !has("libx265") {
                return Err(Error::Ffmpeg(
                    "this ffmpeg has no libx265 HEVC encoder; use H.264".into(),
                ));
            }
            vec![
                "-c:v",
                "libx265",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-tag:v",
                "hvc1",
                "-x265-params",
                "log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
            ]
        } else if has("libx264") {
            vec!["-c:v", "libx264", "-preset", "slow", "-crf", "16"]
        } else if has("libopenh264") {
            vec![
                "-c:v",
                "libopenh264",
                "-b:v",
                &bitrate,
                "-allow_skip_frames",
                "0",
            ]
        } else {
            return Err(Error::Ffmpeg(
                "this ffmpeg has neither the libx264 nor the libopenh264 H.264 encoder".into(),
            ));
        };
        Ok(codec.into_iter().map(String::from).collect())
    }

    /// Start `ffmpeg` (or the program `executable` names) writing `output`
    /// in `codec`, in the container its extension names (see
    /// [`Self::check`]). H.264 uses libx264 when the build has it, else
    /// Cisco's libopenh264, which builds without the x264 encoder (Fedora's
    /// ffmpeg-free among them) carry; HEVC uses libx265. Colours are
    /// converted and tagged as BT.709, which players assume for HD video.
    pub fn start(
        executable: &Path,
        output: &Path,
        settings: VideoSettings,
        codec: Codec,
    ) -> Result<Self> {
        if !settings.width.is_multiple_of(2) || !settings.height.is_multiple_of(2) {
            return Err(Error::OddSize(settings.width, settings.height));
        }
        let format = container(output)?;
        let codec = Self::encoder(executable, codec, settings.bitrate)?;
        // `file:` keeps a name with a colon or a leading dash a file name.
        let mut target = std::ffi::OsString::from("file:");
        target.push(output);
        let mut child = Command::new(executable)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{}x{}", settings.width, settings.height)])
            .args(["-r", &settings.fps.to_string(), "-i", "-"])
            .args(&codec)
            .args([
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv,setparams=colorspace=bt709:\
                 color_primaries=bt709:color_trc=bt709:range=tv",
            ])
            .args(["-colorspace", "bt709", "-color_primaries", "bt709"])
            .args(["-color_trc", "bt709", "-color_range", "tv"])
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .args(["-f", format])
            .arg(target)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(Error::FfmpegMissing)?;
        let stdin = child.stdin.take();
        Ok(Self {
            child,
            stdin,
            settings,
            output: output.to_owned(),
            finished: false,
        })
    }

    /// Whether `executable` runs as ffmpeg.
    pub fn available(executable: &Path) -> bool {
        Command::new(executable)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }
}

impl FrameSink for FfmpegSink {
    fn push(&mut self, frame: &RgbImage) -> Result<()> {
        check_size(frame, &self.settings)?;
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(Error::Ffmpeg("ffmpeg has already stopped".into()));
        };
        if let Err(error) = stdin.write_all(frame.as_raw()) {
            // ffmpeg quit; its own message says why.
            drop(self.stdin.take());
            let mut message = String::new();
            if let Some(mut stderr) = self.child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut stderr, &mut message);
            }
            let _ = self.child.wait();
            return Err(Error::Ffmpeg(format!(
                "stopped taking frames ({error}): {}",
                message.trim()
            )));
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        drop(self.stdin.take());
        let mut message = String::new();
        if let Some(mut stderr) = self.child.stderr.take() {
            let _ = std::io::Read::read_to_string(&mut stderr, &mut message);
        }
        let status = self.child.wait().map_err(|source| Error::Io {
            action: "waiting for",
            path: PathBuf::from("ffmpeg"),
            source,
        })?;
        if status.success() {
            self.finished = true;
            Ok(())
        } else {
            Err(Error::Ffmpeg(message.trim().to_owned()))
        }
    }
}

impl Drop for FfmpegSink {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Stopped or failed: no ffmpeg left behind, and no file that looks
        // finished.
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.output);
    }
}

/// Numbered PNG files, `frame-00000.png` onward, in a directory.
pub struct PngSequence {
    directory: PathBuf,
    next: usize,
    settings: VideoSettings,
}

impl PngSequence {
    /// Frames into `directory`, made if need be. Frames an earlier, longer
    /// video left there (`frame-NNNNN.png`) are removed first, so a
    /// `frame-%05d.png` pattern reads this video alone.
    pub fn new(directory: &Path, settings: VideoSettings) -> Result<Self> {
        std::fs::create_dir_all(directory).map_err(|source| Error::Io {
            action: "creating",
            path: directory.to_owned(),
            source,
        })?;
        let stale = |name: &str| {
            name.strip_prefix("frame-")
                .and_then(|rest| rest.strip_suffix(".png"))
                .is_some_and(|number| {
                    number.len() == 5 && number.bytes().all(|b| b.is_ascii_digit())
                })
        };
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            if entry.file_name().to_str().is_some_and(stale) {
                std::fs::remove_file(entry.path()).map_err(|source| Error::Io {
                    action: "removing",
                    path: entry.path(),
                    source,
                })?;
            }
        }
        Ok(Self {
            directory: directory.to_owned(),
            next: 0,
            settings,
        })
    }
}

impl FrameSink for PngSequence {
    fn push(&mut self, frame: &RgbImage) -> Result<()> {
        check_size(frame, &self.settings)?;
        let path = self.directory.join(format!("frame-{:05}.png", self.next));
        frame
            .save(&path)
            .map_err(|source| Error::Png { path, source })?;
        self.next += 1;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "openh264")]
pub use self::openh264_sink::OpenH264Sink;

#[cfg(feature = "openh264")]
mod openh264_sink {
    use super::*;
    use openh264::OpenH264API;
    use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
    use openh264::formats::{RgbSliceU8, YUVBuffer};

    /// H.264 MP4 encoded in process with Cisco's OpenH264, for hosts without
    /// ffmpeg. Its colours are converted with the BT.601 matrix and left
    /// untagged, so players assuming BT.709 shift them a little; ffmpeg
    /// converts and tags BT.709.
    pub struct OpenH264Sink {
        encoder: Encoder,
        writer: mp4::Mp4Writer<std::io::BufWriter<std::fs::File>>,
        settings: VideoSettings,
        track_added: bool,
        frame: u64,
        output: PathBuf,
    }

    impl OpenH264Sink {
        pub fn start(output: &Path, settings: VideoSettings) -> Result<Self> {
            if !settings.width.is_multiple_of(2) || !settings.height.is_multiple_of(2) {
                return Err(Error::OddSize(settings.width, settings.height));
            }
            // Skipping frames to hold the bitrate would drop them from the
            // video, shortening it.
            let config = EncoderConfig::new()
                .bitrate(BitRate::from_bps(settings.bitrate))
                .max_frame_rate(FrameRate::from_hz(settings.fps as f32))
                .skip_frames(false);
            let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
                .map_err(|error| Error::OpenH264(error.to_string()))?;
            let file = std::fs::File::create(output).map_err(|source| Error::Io {
                action: "creating",
                path: output.to_owned(),
                source,
            })?;
            let writer = mp4::Mp4Writer::write_start(
                std::io::BufWriter::new(file),
                &mp4::Mp4Config {
                    major_brand: str::parse("isom").expect("valid brand"),
                    minor_version: 512,
                    compatible_brands: ["isom", "iso2", "avc1", "mp41"]
                        .into_iter()
                        .map(|brand| str::parse(brand).expect("valid brand"))
                        .collect(),
                    timescale: settings.fps * 1000,
                },
            )
            .map_err(|error| Error::Mp4(error.to_string()))?;
            Ok(Self {
                encoder,
                writer,
                settings,
                track_added: false,
                frame: 0,
                output: output.to_owned(),
            })
        }
    }

    /// Split an Annex B stream into its NAL units, without start codes.
    fn nal_units(stream: &[u8]) -> Vec<&[u8]> {
        let mut units = Vec::new();
        let mut start = None;
        let mut index = 0;
        while index + 3 <= stream.len() {
            let three = stream[index..index + 3] == [0, 0, 1];
            let four = index + 4 <= stream.len() && stream[index..index + 4] == [0, 0, 0, 1];
            if three || four {
                if let Some(begin) = start {
                    units.push(&stream[begin..index]);
                }
                index += if four { 4 } else { 3 };
                start = Some(index);
            } else {
                index += 1;
            }
        }
        if let Some(begin) = start {
            units.push(&stream[begin..]);
        }
        units
    }

    impl FrameSink for OpenH264Sink {
        fn push(&mut self, frame: &RgbImage) -> Result<()> {
            check_size(frame, &self.settings)?;
            let (width, height) = (self.settings.width as usize, self.settings.height as usize);
            let rgb = RgbSliceU8::new(frame.as_raw(), (width, height));
            let yuv = YUVBuffer::from_rgb_source(rgb);
            let bitstream = self
                .encoder
                .encode(&yuv)
                .map_err(|error| Error::OpenH264(error.to_string()))?;
            let stream = bitstream.to_vec();
            let units = nal_units(&stream);
            let mut sps = None;
            let mut pps = None;
            let mut sample = Vec::new();
            let mut is_sync = false;
            for unit in units {
                let Some(&header) = unit.first() else {
                    continue;
                };
                match header & 0x1f {
                    7 => sps = Some(unit.to_vec()),
                    8 => pps = Some(unit.to_vec()),
                    kind => {
                        if kind == 5 {
                            is_sync = true;
                        }
                        sample.extend_from_slice(&(unit.len() as u32).to_be_bytes());
                        sample.extend_from_slice(unit);
                    }
                }
            }
            if !self.track_added {
                let (Some(sps), Some(pps)) = (sps, pps) else {
                    return Err(Error::OpenH264(
                        "the first frame carried no SPS and PPS".into(),
                    ));
                };
                self.writer
                    .add_track(&mp4::TrackConfig {
                        track_type: mp4::TrackType::Video,
                        timescale: self.settings.fps * 1000,
                        language: "und".into(),
                        media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
                            width: self.settings.width as u16,
                            height: self.settings.height as u16,
                            seq_param_set: sps,
                            pic_param_set: pps,
                        }),
                    })
                    .map_err(|error| Error::Mp4(error.to_string()))?;
                self.track_added = true;
            }
            if sample.is_empty() {
                // Time moves on all the same.
                self.frame += 1;
                return Ok(());
            }
            self.writer
                .write_sample(
                    1,
                    &mp4::Mp4Sample {
                        start_time: self.frame * 1000,
                        duration: 1000,
                        rendering_offset: 0,
                        is_sync,
                        bytes: mp4::Bytes::from(sample),
                    },
                )
                .map_err(|error| Error::Mp4(error.to_string()))?;
            self.frame += 1;
            Ok(())
        }

        fn finish(mut self: Box<Self>) -> Result<()> {
            self.writer
                .write_end()
                .map_err(|error| Error::Mp4(format!("{}: {error}", self.output.display())))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn nal_units_split_on_three_and_four_byte_start_codes() {
            let stream = [
                0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 4, 5,
            ];
            let units = nal_units(&stream);
            assert_eq!(
                units,
                vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4, 5][..]]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sizes_parse_by_name_and_by_sides() {
        assert_eq!(parse_frame_size("1080p"), Ok((1920, 1080)));
        assert_eq!(parse_frame_size("4K"), Ok((3840, 2160)));
        assert_eq!(parse_frame_size("720p-portrait"), Ok((720, 1280)));
        assert_eq!(parse_frame_size("1920x1080"), Ok((1920, 1080)));
        assert_eq!(parse_frame_size("1080X1350"), Ok((1080, 1350)));
        for bad in ["1921x1080", "8x8", "8k", "1080p-sideways", "big"] {
            assert!(parse_frame_size(bad).is_err(), "{bad}");
        }
    }

    fn small() -> VideoSettings {
        VideoSettings {
            width: 64,
            height: 48,
            fps: 10,
            bitrate: 1_000_000,
        }
    }

    #[test]
    fn a_video_name_must_be_a_videos_in_a_folder_that_is_there() {
        for name in ["frame.tif", "video", "notes.txt"] {
            assert!(container(Path::new(name)).is_err(), "{name}");
        }
        assert_eq!(container(Path::new("a/B.MP4")).unwrap(), "mp4");
        assert_eq!(container(Path::new("b.mkv")).unwrap(), "matroska");
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("no-such-folder").join("out.mp4");
        assert!(FfmpegSink::check(Path::new("ffmpeg"), &missing, Codec::H264).is_err());
    }

    #[test]
    fn ffmpeg_writes_a_video_and_cleans_up_one_it_did_not_finish() {
        if !FfmpegSink::available(Path::new("ffmpeg"))
            || FfmpegSink::check(Path::new("ffmpeg"), Path::new("probe.mp4"), Codec::H264).is_err()
        {
            eprintln!("no ffmpeg with an H.264 encoder; skipped");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        // A name ffmpeg would read as a protocol, or an option, without
        // the `file:` prefix.
        let finished = directory.path().join("-M42:Orion.mp4");
        let mut sink = Box::new(
            FfmpegSink::start(Path::new("ffmpeg"), &finished, small(), Codec::H264).unwrap(),
        );
        for _ in 0..5 {
            sink.push(&RgbImage::from_pixel(64, 48, image::Rgb([200, 40, 40])))
                .unwrap();
        }
        sink.finish().unwrap();
        assert!(std::fs::metadata(&finished).unwrap().len() > 0);
        // Tagged BT.709 throughout, in both codecs, where ffprobe can say.
        for codec in [Codec::H264, Codec::Hevc] {
            let path = directory.path().join(format!("{codec:?}.mp4"));
            let Ok(mut sink) = FfmpegSink::start(Path::new("ffmpeg"), &path, small(), codec) else {
                continue;
            };
            sink.push(&RgbImage::from_pixel(64, 48, image::Rgb([40, 200, 40])))
                .unwrap();
            Box::new(sink).finish().unwrap();
            let Ok(probe) = Command::new("ffprobe")
                .args(["-v", "error", "-select_streams", "v", "-show_entries"])
                .arg("stream=color_space,color_primaries,color_transfer")
                .args(["-of", "csv=p=0"])
                .arg(&path)
                .output()
            else {
                continue;
            };
            let tags = String::from_utf8_lossy(&probe.stdout);
            assert_eq!(tags.trim(), "bt709,bt709,bt709", "{codec:?}");
        }
        // Stopped early: no partial file, and ffmpeg waited for.
        let stopped = directory.path().join("stopped.mp4");
        let mut sink =
            FfmpegSink::start(Path::new("ffmpeg"), &stopped, small(), Codec::H264).unwrap();
        sink.push(&RgbImage::new(64, 48)).unwrap();
        let id = sink.child.id();
        drop(sink);
        assert!(!stopped.exists());
        assert!(!Path::new(&format!("/proc/{id}")).exists() || cfg!(not(target_os = "linux")));
    }

    #[test]
    fn png_sequences_number_their_frames_and_check_sizes() {
        let directory = tempfile::tempdir().unwrap();
        let settings = VideoSettings {
            width: 4,
            height: 2,
            fps: 30,
            bitrate: 1_000_000,
        };
        let mut sink: Box<dyn FrameSink> =
            Box::new(PngSequence::new(directory.path(), settings).unwrap());
        sink.push(&RgbImage::new(4, 2)).unwrap();
        sink.push(&RgbImage::new(4, 2)).unwrap();
        assert!(matches!(
            sink.push(&RgbImage::new(2, 2)),
            Err(Error::FrameSize { .. })
        ));
        sink.finish().unwrap();
        assert!(directory.path().join("frame-00001.png").exists());
        // A shorter video into the same folder leaves none of the longer
        // one's frames, and touches nothing else there.
        std::fs::write(directory.path().join("notes.txt"), "keep").unwrap();
        let mut sink: Box<dyn FrameSink> =
            Box::new(PngSequence::new(directory.path(), settings).unwrap());
        sink.push(&RgbImage::new(4, 2)).unwrap();
        sink.finish().unwrap();
        assert!(directory.path().join("frame-00000.png").exists());
        assert!(!directory.path().join("frame-00001.png").exists());
        assert!(directory.path().join("notes.txt").exists());
    }

    #[test]
    fn ffmpeg_refuses_odd_frame_sizes() {
        let settings = VideoSettings {
            width: 3,
            height: 2,
            fps: 30,
            bitrate: 1_000_000,
        };
        assert!(matches!(
            FfmpegSink::start(
                Path::new("ffmpeg"),
                Path::new("out.mp4"),
                settings,
                Codec::Hevc
            ),
            Err(Error::OddSize(3, 2))
        ));
    }

    #[test]
    fn codecs_parse_by_their_common_names() {
        for (text, codec) in [
            ("h264", Codec::H264),
            ("x264", Codec::H264),
            ("HEVC", Codec::Hevc),
            ("h265", Codec::Hevc),
        ] {
            assert_eq!(Codec::parse(text), Ok(codec), "{text}");
        }
        assert!(Codec::parse("vp9").is_err());
    }
}
