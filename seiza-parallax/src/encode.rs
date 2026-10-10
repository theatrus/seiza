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

/// H.264 MP4 through an `ffmpeg` executable, fed raw RGB on its standard
/// input.
pub struct FfmpegSink {
    child: Child,
    stdin: Option<ChildStdin>,
    settings: VideoSettings,
}

impl FfmpegSink {
    /// Start `ffmpeg` (or the program `executable` names) writing `output`
    /// as H.264: with libx264 when the build has it, else with Cisco's
    /// libopenh264, which builds without the x264 encoder (Fedora's
    /// ffmpeg-free among them) carry.
    pub fn start(executable: &Path, output: &Path, settings: VideoSettings) -> Result<Self> {
        if !settings.width.is_multiple_of(2) || !settings.height.is_multiple_of(2) {
            return Err(Error::OddSize(settings.width, settings.height));
        }
        let encoders = Command::new(executable)
            .args(["-hide_banner", "-encoders"])
            .stderr(Stdio::null())
            .output()
            .map_err(Error::FfmpegMissing)?;
        let encoders = String::from_utf8_lossy(&encoders.stdout);
        let has = |name: &str| encoders.split_whitespace().any(|word| word == name);
        let bitrate = settings.bitrate.to_string();
        let codec: Vec<&str> = if has("libx264") {
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
        let mut child = Command::new(executable)
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{}x{}", settings.width, settings.height)])
            .args(["-r", &settings.fps.to_string(), "-i", "-"])
            .args(&codec)
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(output)
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
        let stdin = self.stdin.as_mut().expect("open until finished");
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
            Ok(())
        } else {
            Err(Error::Ffmpeg(message.trim().to_owned()))
        }
    }
}

/// Numbered PNG files, `frame-00000.png` onward, in a directory.
pub struct PngSequence {
    directory: PathBuf,
    next: usize,
    settings: VideoSettings,
}

impl PngSequence {
    pub fn new(directory: &Path, settings: VideoSettings) -> Result<Self> {
        std::fs::create_dir_all(directory).map_err(|source| Error::Io {
            action: "creating",
            path: directory.to_owned(),
            source,
        })?;
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
    /// ffmpeg.
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
            let config = EncoderConfig::new()
                .bitrate(BitRate::from_bps(settings.bitrate))
                .max_frame_rate(FrameRate::from_hz(settings.fps as f32));
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
        #[test]
        fn frame_sizes_parse_by_name_and_by_sides() {
            assert_eq!(super::parse_frame_size("1080p"), Ok((1920, 1080)));
            assert_eq!(super::parse_frame_size("4K"), Ok((3840, 2160)));
            assert_eq!(super::parse_frame_size("720p-portrait"), Ok((720, 1280)));
            assert_eq!(super::parse_frame_size("1920x1080"), Ok((1920, 1080)));
            assert_eq!(super::parse_frame_size("1080X1350"), Ok((1080, 1350)));
            for bad in ["1921x1080", "8x8", "8k", "1080p-sideways", "big"] {
                assert!(super::parse_frame_size(bad).is_err(), "{bad}");
            }
        }

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
            FfmpegSink::start(Path::new("ffmpeg"), Path::new("out.mp4"), settings),
            Err(Error::OddSize(3, 2))
        ));
    }
}
