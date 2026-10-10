//! Labels over a parallax video: the catalogued objects in the field, each
//! moving with the layer that shows it and fading as it leaves the view,
//! the caller's own labels, and a watermark.
//!
//! The marks follow Seiza's image overlays (`@seiza/astro-overlay`): the same
//! catalog colours, labels and prominence ranking, a catalogued extent or
//! outline for each object, ticks either side of a named star, and a
//! "Field within" caption once the camera is inside an object.

use crate::pipeline::Error;
use crate::{Scene, Shot, View};
use ab_glyph::{Font, PxScale, ScaleFont};
use image::{Rgb, RgbImage};
use seiza::Wcs;
use seiza::objects::{GeometryData, ObjectCatalog, ObjectKind, SkyObject};
use seiza_draw::{Fonts, Mask, draw_text, measure};

/// The watermark `--watermark` writes when given no text.
pub const DEFAULT_WATERMARK: &str = "Rendered with seiza.fyi";

/// The share of the ranked objects in view that are labelled, as in the
/// image overlays.
pub const DEFAULT_DENSITY: f64 = 0.6;

/// The most prominent objects in view are labelled however low the density.
const MINIMUM_RANKED: usize = 4;

/// Seconds a mark takes to fade in or out.
const FADE_SECONDS: f64 = 0.5;

const STAR_GOLD: Rgb<u8> = Rgb([0xff, 0xd4, 0x79]);
const ENCOMPASSING: Rgb<u8> = Rgb([0xae, 0xe8, 0xff]);
const WATERMARK: Rgb<u8> = Rgb([0xec, 0xf0, 0xf4]);
const HALO: Rgb<u8> = Rgb([0, 0, 0]);

/// What marks an object's place, in image pixels.
#[derive(Clone, Debug, PartialEq)]
enum Marker {
    /// A catalogued extent; a circle of the major axis when its orientation
    /// is unknown.
    Ellipse {
        semi_major: f64,
        semi_minor: f64,
        angle_deg: Option<f64>,
    },
    /// Catalogued outlines, each a polyline, closed ones ending where they
    /// start.
    Outlines(Vec<Vec<(f64, f64)>>),
    /// A named star: ticks either side of it.
    Star,
    /// A circle of this radius, or at zero only the label, centred on the
    /// point.
    Circle(f64),
}

/// One labelled place in the image.
#[derive(Clone, Debug)]
pub(crate) struct Mark {
    label: String,
    color: Rgb<u8>,
    /// Image pixels.
    x: f64,
    y: f64,
    /// The depth of the layer that shows it, so it moves with what it
    /// marks.
    distance_pc: f64,
    marker: Marker,
    /// Its catalogued extent's semi-axes, image pixels, for the "Field
    /// within" caption; zero for a point.
    extent: (f64, f64),
    /// The catalogued object, ranked by prominence. The caller's own labels
    /// have none and always show.
    object: Option<SkyObject>,
    /// A star fades as the camera passes it.
    star: bool,
}

/// A label of the caller's own on the nebula's plane, at image pixel
/// `(x, y)`, circling `radius` pixels about it, or with a radius of zero
/// only the text, centred on the point.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomLabel {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub text: String,
}

/// A [`CustomLabel`] written `X,Y:TEXT` or `X,Y,RADIUS:TEXT`.
pub fn parse_label(text: &str) -> std::result::Result<CustomLabel, String> {
    let (place, label) = text
        .split_once(':')
        .ok_or_else(|| format!("expected X,Y[,RADIUS]:TEXT; got {text}"))?;
    let numbers = place
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|error| format!("{part}: {error}"))
                .and_then(|value| {
                    value
                        .is_finite()
                        .then_some(value)
                        .ok_or_else(|| format!("{part}: not a number"))
                })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let (x, y, radius) = match numbers[..] {
        [x, y] => (x, y, 0.0),
        [x, y, radius] if radius >= 0.0 => (x, y, radius),
        _ => return Err(format!("expected X,Y[,RADIUS]:TEXT; got {text}")),
    };
    let label = label.trim();
    if label.is_empty() {
        return Err(format!("{text}: the label has no text"));
    }
    Ok(CustomLabel {
        x,
        y,
        radius,
        text: label.to_string(),
    })
}

/// A colour written `#RRGGBB`.
pub fn parse_color(text: &str) -> std::result::Result<Rgb<u8>, String> {
    let hex = text.trim().trim_start_matches('#');
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("expected a colour as #RRGGBB; got {text}"));
    }
    let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).unwrap();
    Ok(Rgb([channel(0), channel(2), channel(4)]))
}

/// The caller's labels, on the nebula's plane.
pub(crate) fn custom_marks(labels: &[CustomLabel], color: Rgb<u8>, scene: &Scene) -> Vec<Mark> {
    labels
        .iter()
        .map(|label| Mark {
            label: label.text.clone(),
            color,
            x: label.x,
            y: label.y,
            distance_pc: scene.background_distance_pc,
            marker: Marker::Circle(label.radius),
            extent: (0.0, 0.0),
            object: None,
            star: false,
        })
        .collect()
}

/// The catalogued objects in a `dimensions` image, each at the depth of the
/// layer that shows it: a named star at the distance of its sprite, a
/// galaxy lifted onto the far field where `lifted` put it, and everything
/// else on the nebula's plane.
pub(crate) fn catalog_marks(
    catalog: &ObjectCatalog,
    wcs: &Wcs,
    dimensions: (u32, u32),
    scene: &Scene,
    lifted: &[(String, (f64, f64), f64)],
) -> Result<Vec<Mark>, Error> {
    let placed = catalog
        .objects_in_footprint(wcs, dimensions)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let mut marks = Vec::new();
    for placed in placed {
        let object = placed.object;
        if object.kind == ObjectKind::Transient {
            continue;
        }
        let star = matches!(object.kind, ObjectKind::Star | ObjectKind::DoubleStar);
        let (mut x, mut y) = (placed.x, placed.y);
        let mut distance_pc = scene.background_distance_pc;
        if star {
            // The sprite drawn for it, if one is near enough.
            let reach = 4.0_f64.max(placed.semi_major_px);
            match scene
                .sprites
                .iter()
                .map(|sprite| (sprite, (sprite.x - x).hypot(sprite.y - y)))
                .filter(|&(_, apart)| apart <= reach)
                .min_by(|a, b| a.1.total_cmp(&b.1))
            {
                Some((sprite, _)) => (x, y, distance_pc) = (sprite.x, sprite.y, sprite.distance_pc),
                None => distance_pc = scene.leftover_distance_pc,
            }
        } else if let Some((_, at, pc)) = lifted.iter().find(|(name, _, _)| *name == object.name) {
            (x, y, distance_pc) = (at.0, at.1, *pc);
        }
        let (dx, dy) = (x - placed.x, y - placed.y);
        let outlines = if star || object.metadata.id.is_empty() {
            Vec::new()
        } else {
            projected_outlines(catalog, &object.metadata.id, wcs, (dx, dy))
        };
        let marker = if star {
            Marker::Star
        } else if !outlines.is_empty() {
            Marker::Outlines(outlines)
        } else {
            Marker::Ellipse {
                semi_major: placed.semi_major_px,
                semi_minor: placed.semi_minor_px,
                angle_deg: placed.angle_deg,
            }
        };
        let extent = if star {
            (0.0, 0.0)
        } else {
            match placed.angle_deg {
                Some(_) => (placed.semi_major_px, placed.semi_minor_px),
                None => (placed.semi_major_px, placed.semi_major_px),
            }
        };
        marks.push(Mark {
            label: label_for(&object),
            color: color_for(&object),
            x,
            y,
            distance_pc,
            marker,
            extent,
            object: Some(object),
            star,
        });
    }
    Ok(marks)
}

/// An object's catalogued outlines in image pixels, moved by `shift`.
fn projected_outlines(
    catalog: &ObjectCatalog,
    id: &str,
    wcs: &Wcs,
    shift: (f64, f64),
) -> Vec<Vec<(f64, f64)>> {
    let Ok(geometries) = catalog.geometries(id) else {
        return Vec::new();
    };
    geometries
        .into_iter()
        .filter_map(|geometry| match geometry.data {
            GeometryData::OutlineSet { contours, .. } => Some(contours),
            _ => None,
        })
        .flatten()
        .filter_map(|contour| {
            let mut points = contour
                .vertices
                .iter()
                .map(|&(ra, dec)| {
                    wcs.world_to_pixel(ra, dec)
                        .map(|(x, y)| (x + shift.0, y + shift.1))
                })
                .collect::<Option<Vec<_>>>()?;
            if contour.closed {
                points.push(*points.first()?);
            }
            (points.len() >= 2).then_some(points)
        })
        .collect()
}

/// `name · common name`, or whichever there is.
fn label_for(object: &SkyObject) -> String {
    if !object.common_name.is_empty() && object.common_name != object.name {
        format!("{} · {}", object.name, object.common_name)
    } else if object.common_name.is_empty() {
        object.name.clone()
    } else {
        object.common_name.clone()
    }
}

/// The image overlays' suggested catalog colour, by the object's primary
/// designation; gold for a named star.
fn color_for(object: &SkyObject) -> Rgb<u8> {
    if matches!(object.kind, ObjectKind::Star | ObjectKind::DoubleStar) {
        return STAR_GOLD;
    }
    let name = object.name.trim();
    let hex = if word(name, "PGC") {
        0xa1aed8
    } else if word(name, "UGC") {
        0x79aff5
    } else if word(name, "LBN") {
        0xa2d96f
    } else if word(name, "Ced") || word(name, "Cederblad") {
        0x70d7d0
    } else if word(name, "LDN") || numbered(name, "B") {
        0xb4a3f0
    } else if word(name, "SNR") {
        0xf18782
    } else if sharpless(name) || word(name, "vdB") {
        0xee9a78
    } else if numbered(name, "M") {
        0xf2ca72
    } else if numbered(name, "NGC") {
        0x55cfff
    } else if numbered(name, "IC") {
        0x72dfb9
    } else {
        0xc1d1d3
    };
    Rgb([(hex >> 16) as u8, (hex >> 8) as u8, hex as u8])
}

/// The rest of `name` after `prefix`, ignoring case.
fn after<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let head = name.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &name[prefix.len()..])
}

/// Whether `name` is `prefix` followed by a space or nothing.
fn word(name: &str, prefix: &str) -> bool {
    after(name, prefix).is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
}

/// Whether `name` is `prefix` followed by a number, spaced or not.
fn numbered(name: &str, prefix: &str) -> bool {
    after(name, prefix)
        .is_some_and(|rest| rest.trim_start().starts_with(|c: char| c.is_ascii_digit()))
}

/// Whether `name` is a Sharpless designation, `Sh2-` or `Sh 2 `.
fn sharpless(name: &str) -> bool {
    after(name, "Sh").is_some_and(|rest| {
        let rest = rest.trim_start();
        rest.starts_with("2-") || rest.starts_with("2 ")
    })
}

/// The marks over a shot, and how much of each every frame shows.
pub(crate) struct Overlay {
    marks: Vec<Mark>,
    fonts: Fonts<'static>,
    watermark: Option<String>,
    density: f64,
    /// The image's scale, for the field a frame shows.
    arcsec_per_px: f64,
    /// Label size and line width, output pixels.
    font_size: f64,
    line: f64,
    /// Each label's width, output pixels.
    widths: Vec<f64>,
    /// Per frame, per mark: how much of its marker and label shows, and of
    /// its "Field within" caption.
    shown: Vec<Vec<f32>>,
    within: Vec<Vec<f32>>,
    /// Each mark's line among the captions, the first at the bottom.
    slots: Vec<Option<usize>>,
}

impl Overlay {
    pub(crate) fn new(
        marks: Vec<Mark>,
        watermark: Option<String>,
        density: f64,
        arcsec_per_px: f64,
        (width, height): (usize, usize),
    ) -> Result<Self, Error> {
        let fonts = Fonts::load().map_err(|error| Error::Fonts(error.to_string()))?;
        let short = width.min(height) as f64;
        let font_size = (short / 42.0).max(14.0);
        let widths = marks
            .iter()
            .map(|mark| measure(&fonts.regular, font_size, 0.0, &mark.label).0)
            .collect();
        Ok(Self {
            marks,
            fonts,
            watermark,
            density: density.clamp(0.0, 1.0),
            arcsec_per_px,
            font_size,
            line: (short / 720.0).max(1.2),
            widths,
            shown: Vec::new(),
            within: Vec::new(),
            slots: Vec::new(),
        })
    }

    /// Work out how much of each mark every frame of `shot` shows, at
    /// `fps` frames a second: a mark fades out as it nears the frame's edge,
    /// as the camera flies inside it (its name going to the "Field within"
    /// caption), as a star the camera passes fades, or when its label
    /// would cover a more prominent one or more prominent objects fill the
    /// density's share.
    pub(crate) fn plan(&mut self, shot: &Shot, scene: &Scene, fps: f64) {
        let mut shown = Vec::with_capacity(shot.frames);
        let mut within = Vec::with_capacity(shot.frames);
        for frame in 0..shot.frames {
            let view = shot.view(scene, frame);
            let (show, inside) = self.wanted(shot, &view);
            shown.push(show);
            within.push(inside);
        }
        let radius = (FADE_SECONDS * fps / 2.0).round() as usize;
        self.shown = smoothed(&shown, radius);
        self.within = smoothed(&within, radius);
        // Captions keep their lines, in the order they first appear.
        let mut first: Vec<(usize, usize)> = (0..self.marks.len())
            .filter_map(|mark| {
                self.within
                    .iter()
                    .position(|frame| frame[mark] > 0.01)
                    .map(|at| (at, mark))
            })
            .collect();
        first.sort();
        self.slots = vec![None; self.marks.len()];
        for (slot, &(_, mark)) in first.iter().enumerate() {
            self.slots[mark] = Some(slot);
        }
    }

    /// How much of each mark `view` should show before fading in time, and
    /// of each caption.
    fn wanted(&self, shot: &Shot, view: &View) -> (Vec<f32>, Vec<f32>) {
        let (width, height) = (shot.width as f64, shot.height as f64);
        let half_diagonal = width.hypot(height) / 2.0;
        let margin = 0.05 * width.min(height);
        let count = self.marks.len();
        let (mut show, mut within) = (vec![0.0_f32; count], vec![0.0_f32; count]);
        let mut prominence = vec![f64::INFINITY; count];
        for (index, mark) in self.marks.iter().enumerate() {
            let Some((u, v, scale)) = view.project(mark.x, mark.y, mark.distance_pc) else {
                continue;
            };
            let fade = if mark.star {
                shot.star_fade(view, scale)
            } else {
                1.0
            };
            let edge = u.min(width - 1.0 - u).min(v).min(height - 1.0 - v);
            let inside = (edge / margin).clamp(0.0, 1.0);
            // How nearly the object's extent takes in the whole frame.
            let filling = smoothstep(0.7, 1.0, mark.extent.1 * scale / half_diagonal);
            show[index] = (inside * fade * (1.0 - filling)) as f32;
            within[index] = (filling * fade) as f32;
            if let Some(object) = &mark.object {
                let field_deg = half_diagonal / scale * self.arcsec_per_px / 3600.0;
                prominence[index] = seiza::objects::predicted_prominence(object, true, field_deg);
            }
        }
        // The density's share of the ranked objects in view, the most
        // prominent first; the caller's own labels always.
        let mut ranked: Vec<usize> = (0..count).filter(|&index| show[index] > 0.0).collect();
        ranked.sort_by(|&a, &b| prominence[b].total_cmp(&prominence[a]));
        let rankable = ranked
            .iter()
            .filter(|&&index| self.marks[index].object.is_some())
            .count();
        let floor = rankable.min(MINIMUM_RANKED);
        let budget =
            floor.max((floor as f64 + (rankable - floor) as f64 * self.density).round() as usize);
        let mut placed: Vec<(f64, f64, f64)> = Vec::new();
        let mut taken = 0;
        for index in ranked {
            if self.marks[index].object.is_some() {
                if taken == budget {
                    show[index] = 0.0;
                    continue;
                }
                taken += 1;
            }
            // A label covering a more prominent one gives way.
            let (x, y) = self.label_place(index, view, (width, height));
            let half = self.widths[index] / 2.0;
            if placed.iter().any(|&(px, py, phalf)| {
                (py - y).abs() < self.font_size * 1.3 && (px - x).abs() < phalf + half
            }) {
                show[index] = 0.0;
                continue;
            }
            placed.push((x, y, half));
        }
        (show, within)
    }

    /// Where mark `index`'s label goes in `view`: its centre and baseline,
    /// above its marker and inside the frame.
    fn label_place(&self, index: usize, view: &View, (width, height): (f64, f64)) -> (f64, f64) {
        let mark = &self.marks[index];
        let size = self.font_size;
        let Some((u, v, scale)) = view.project(mark.x, mark.y, mark.distance_pc) else {
            return (f64::NAN, f64::NAN);
        };
        let reach = match &mark.marker {
            Marker::Circle(radius) if *radius <= 0.0 => {
                // The label alone, centred on the point.
                return self.inside((u, v + 0.35 * size), index, (width, height));
            }
            Marker::Circle(radius) => radius * scale,
            Marker::Star => size,
            Marker::Ellipse { .. } | Marker::Outlines(_) => {
                let (a, b) = (mark.extent.0 * scale, mark.extent.1 * scale);
                let angle = match &mark.marker {
                    Marker::Ellipse {
                        angle_deg: Some(angle),
                        ..
                    } => angle.to_radians() - view.turn(),
                    _ => 0.0,
                };
                let (sin, cos) = angle.sin_cos();
                (a.max(size) * sin).hypot(b.max(size) * cos)
            }
        };
        self.inside((u, v - reach - 0.5 * size), index, (width, height))
    }

    /// `(x, baseline)` kept inside the frame for mark `index`'s label.
    fn inside(&self, (x, y): (f64, f64), index: usize, (width, height): (f64, f64)) -> (f64, f64) {
        let size = self.font_size;
        let half = self.widths[index] / 2.0;
        let pad = size * 0.25;
        let x = if half + pad >= width / 2.0 {
            width / 2.0
        } else {
            x.clamp(half + pad, width - half - pad)
        };
        (x, y.clamp(size * 1.1, height - size * 0.35))
    }

    /// Draw frame `frame`'s marks, as `view` sees them, over `canvas`.
    pub(crate) fn draw(&self, frame: usize, view: &View, canvas: &mut RgbImage) {
        let (width, height) = (canvas.width() as f64, canvas.height() as f64);
        let size = self.font_size;
        if let Some(shown) = self.shown.get(frame) {
            for (index, mark) in self.marks.iter().enumerate() {
                let alpha = shown[index];
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let Some((u, v, scale)) = view.project(mark.x, mark.y, mark.distance_pc) else {
                    continue;
                };
                // Image pixel centres sit at whole numbers; the mask's
                // pixel i spans i to i + 1.
                let centre = (u + 0.5, v + 0.5);
                let shapes = match &mark.marker {
                    Marker::Ellipse {
                        semi_major,
                        semi_minor,
                        angle_deg,
                    } => {
                        let a = (semi_major * scale).max(size);
                        let b = match angle_deg {
                            Some(_) => (semi_minor * scale).max(size),
                            None => a,
                        };
                        let angle = angle_deg.unwrap_or(0.0) - view.turn().to_degrees();
                        vec![Shape::Ellipse(centre, a, b, angle)]
                    }
                    Marker::Outlines(outlines) => outlines
                        .iter()
                        .map(|outline| {
                            Shape::Line(
                                outline
                                    .iter()
                                    .filter_map(|&(x, y)| view.project(x, y, mark.distance_pc))
                                    .map(|(x, y, _)| (x + 0.5, y + 0.5))
                                    .collect(),
                            )
                        })
                        .collect(),
                    Marker::Star => {
                        let (x, y) = centre;
                        vec![
                            Shape::Line(vec![(x - size, y), (x - size / 3.0, y)]),
                            Shape::Line(vec![(x + size / 3.0, y), (x + size, y)]),
                        ]
                    }
                    Marker::Circle(radius) if *radius > 0.0 => {
                        vec![Shape::Ellipse(centre, radius * scale, radius * scale, 0.0)]
                    }
                    Marker::Circle(_) => Vec::new(),
                };
                let (x, baseline) = self.label_place(index, view, (width, height));
                let left = x - self.widths[index] / 2.0;
                let mut bounds = Bounds::around(&[
                    (left, baseline - size),
                    (left + self.widths[index], baseline + size * 0.3),
                ]);
                for shape in &shapes {
                    bounds = bounds.with(shape.bounds());
                }
                let Some(mut mask) = self.mask(bounds, canvas) else {
                    continue;
                };
                for shape in &shapes {
                    match shape {
                        &Shape::Ellipse(centre, a, b, angle) => {
                            mask.ellipse(centre, a, b, angle, self.line)
                        }
                        Shape::Line(points) => mask.polyline(points, self.line),
                    }
                }
                self.text(&mut mask, (left, baseline), size, &mark.label);
                self.composite(&mask, canvas, mark.color, alpha);
            }
        }
        // The objects the camera is inside, a line each.
        if let Some(within) = self.within.get(frame) {
            for (index, mark) in self.marks.iter().enumerate() {
                let (alpha, Some(slot)) = (within[index], self.slots[index]) else {
                    continue;
                };
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let text = format!("Field within: {}", mark.label);
                let baseline = height - size - slot as f64 * size * 1.4;
                self.caption(canvas, (size, baseline), size, &text, ENCOMPASSING, alpha);
            }
        }
        if let Some(watermark) = &self.watermark {
            let small = size * 0.8;
            let (text_width, _) = measure(&self.fonts.regular, small, 0.0, watermark);
            let place = (width - size - text_width, height - size);
            self.caption(canvas, place, small, watermark, WATERMARK, 0.8);
        }
    }

    /// One line of `text` at `size` from `(x, baseline)`, in `color` at
    /// `alpha`.
    fn caption(
        &self,
        canvas: &mut RgbImage,
        (x, baseline): (f64, f64),
        size: f64,
        text: &str,
        color: Rgb<u8>,
        alpha: f32,
    ) {
        let (text_width, _) = measure(&self.fonts.regular, size, 0.0, text);
        let bounds = Bounds::around(&[
            (x, baseline - size),
            (x + text_width, baseline + size * 0.3),
        ]);
        if let Some(mut mask) = self.mask(bounds, canvas) {
            self.text(&mut mask, (x, baseline), size, text);
            self.composite(&mask, canvas, color, alpha);
        }
    }

    /// A mask over `bounds` with room for strokes and the halo, cut to the
    /// canvas, or `None` when nothing of it lands there.
    fn mask(&self, bounds: Bounds, canvas: &RgbImage) -> Option<Mask> {
        let pad = self.halo() + self.line + 2.0;
        let left = ((bounds.low.0 - pad).floor() as i64).max(0);
        let top = ((bounds.low.1 - pad).floor() as i64).max(0);
        let right = ((bounds.high.0 + pad).ceil() as i64).min(canvas.width() as i64);
        let bottom = ((bounds.high.1 + pad).ceil() as i64).min(canvas.height() as i64);
        (left < right && top < bottom).then(|| {
            Mask::at(
                (left, top),
                (right - left) as usize,
                (bottom - top) as usize,
            )
        })
    }

    /// Write `text` at `size` with its left end at `(x, baseline)`.
    fn text(&self, mask: &mut Mask, (x, baseline): (f64, f64), size: f64, text: &str) {
        let ascent = self
            .fonts
            .regular
            .as_scaled(PxScale::from(size as f32))
            .ascent() as f64;
        draw_text(
            mask,
            &self.fonts.regular,
            size,
            0.0,
            (x, baseline - ascent),
            text,
        );
    }

    fn halo(&self) -> f64 {
        (self.font_size * 0.09).max(1.5)
    }

    /// Lay `mask` over `canvas` in `color` at `alpha`, on a dark halo.
    fn composite(&self, mask: &Mask, canvas: &mut RgbImage, color: Rgb<u8>, alpha: f32) {
        mask.dilated(self.halo())
            .composite(canvas, HALO, 0.75 * alpha);
        mask.composite(canvas, color, alpha);
    }
}

/// A marker's lines, output pixels.
enum Shape {
    /// Centre, semi-axes and angle, degrees clockwise.
    Ellipse((f64, f64), f64, f64, f64),
    Line(Vec<(f64, f64)>),
}

impl Shape {
    fn bounds(&self) -> Bounds {
        match self {
            &Shape::Ellipse((x, y), a, b, _) => {
                let reach = a.max(b);
                Bounds::around(&[(x - reach, y - reach), (x + reach, y + reach)])
            }
            Shape::Line(points) => Bounds::around(points),
        }
    }
}

/// A box, output pixels.
#[derive(Clone, Copy, Debug)]
struct Bounds {
    low: (f64, f64),
    high: (f64, f64),
}

impl Bounds {
    fn around(points: &[(f64, f64)]) -> Self {
        let mut bounds = Self {
            low: (f64::INFINITY, f64::INFINITY),
            high: (f64::NEG_INFINITY, f64::NEG_INFINITY),
        };
        for &(x, y) in points {
            bounds.low = (bounds.low.0.min(x), bounds.low.1.min(y));
            bounds.high = (bounds.high.0.max(x), bounds.high.1.max(y));
        }
        bounds
    }

    fn with(self, other: Bounds) -> Self {
        Self {
            low: (self.low.0.min(other.low.0), self.low.1.min(other.low.1)),
            high: (self.high.0.max(other.high.0), self.high.1.max(other.high.1)),
        }
    }
}

/// `values` per frame, each column averaged over `radius` frames either
/// side, so a mark fades in and out instead of popping.
fn smoothed(values: &[Vec<f32>], radius: usize) -> Vec<Vec<f32>> {
    let frames = values.len();
    let count = values.first().map_or(0, Vec::len);
    let mut out = vec![vec![0.0_f32; count]; frames];
    for column in 0..count {
        for (frame, row) in out.iter_mut().enumerate() {
            // Frames past either end hold the end's value.
            let mut sum = 0.0;
            for at in frame as isize - radius as isize..=(frame + radius) as isize {
                sum += values[at.clamp(0, frames as isize - 1) as usize][column];
            }
            row[column] = sum / (2 * radius + 1) as f32;
        }
    }
    out
}

fn smoothstep(low: f64, high: f64, value: f64) -> f64 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CutOptions, Easing, LightImage};

    fn object(name: &str, common: &str, kind: ObjectKind, major: f32) -> SkyObject {
        SkyObject {
            kind,
            ra: 0.0,
            dec: 0.0,
            mag: Some(8.0),
            major_arcmin: Some(major),
            minor_arcmin: None,
            position_angle_deg: None,
            name: name.into(),
            common_name: common.into(),
            metadata: Default::default(),
        }
    }

    fn scene() -> Scene {
        let image = LightImage::new(400, 300);
        Scene::new(
            &image,
            &image,
            &[],
            400.0,
            2000.0,
            400.0,
            &CutOptions::default(),
        )
    }

    fn mark(name: &str, x: f64, y: f64, radius: f64) -> Mark {
        Mark {
            label: name.into(),
            color: Rgb([255, 255, 255]),
            x,
            y,
            distance_pc: 400.0,
            marker: Marker::Ellipse {
                semi_major: radius,
                semi_minor: radius,
                angle_deg: None,
            },
            extent: (radius, radius),
            object: Some(object(name, "", ObjectKind::Nebula, radius as f32)),
            star: false,
        }
    }

    #[test]
    fn labels_and_colours_follow_the_image_overlays() {
        let iris = object("NGC 7023", "Iris Nebula", ObjectKind::Nebula, 10.0);
        assert_eq!(label_for(&iris), "NGC 7023 · Iris Nebula");
        assert_eq!(color_for(&iris), Rgb([0x55, 0xcf, 0xff]));
        let same = object("M 31", "M 31", ObjectKind::Galaxy, 10.0);
        assert_eq!(label_for(&same), "M 31");
        for (name, hex) in [
            ("M31", 0xf2ca72),
            ("Sh2-171", 0xee9a78),
            ("vdB 139", 0xee9a78),
            ("Ced 214", 0x70d7d0),
            ("LDN 1172", 0xb4a3f0),
            ("B33", 0xb4a3f0),
            ("Barnard's Loop", 0xc1d1d3),
            ("PGC 12345", 0xa1aed8),
            ("IC 1396", 0x72dfb9),
            ("Mel 20", 0xc1d1d3),
        ] {
            let found = color_for(&object(name, "", ObjectKind::Nebula, 1.0));
            assert_eq!(
                found,
                Rgb([(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]),
                "{name}"
            );
        }
        let star = object("HD 200775", "", ObjectKind::Star, 0.0);
        assert_eq!(color_for(&star), STAR_GOLD);
    }

    #[test]
    fn custom_labels_parse() {
        assert_eq!(
            parse_label("10,20:Teddy Bear").unwrap(),
            CustomLabel {
                x: 10.0,
                y: 20.0,
                radius: 0.0,
                text: "Teddy Bear".into()
            }
        );
        assert_eq!(
            parse_label("10.5, 20, 40: Pillars: west").unwrap().radius,
            40.0
        );
        assert_eq!(
            parse_label("1,2,3: a: b").unwrap().text,
            "a: b",
            "the text runs past the first colon"
        );
        for bad in ["10,20", "10:x", "1,2,-3:x", "1,2: ", "a,2:x"] {
            assert!(parse_label(bad).is_err(), "{bad}");
        }
        assert_eq!(parse_color("#ffd479").unwrap(), STAR_GOLD);
        assert!(parse_color("ffd47").is_err());
    }

    #[test]
    fn smoothing_fades_a_step_over_the_window() {
        let values: Vec<Vec<f32>> = (0..10)
            .map(|frame| vec![if frame < 5 { 1.0 } else { 0.0 }])
            .collect();
        let smooth = smoothed(&values, 2);
        assert_eq!(smooth[0][0], 1.0);
        assert_eq!(smooth[9][0], 0.0);
        assert!((smooth[4][0] - 0.6).abs() < 1e-6 && (smooth[5][0] - 0.4).abs() < 1e-6);
        for pair in smooth.windows(2) {
            assert!(pair[1][0] <= pair[0][0]);
        }
    }

    #[test]
    fn marks_fade_out_as_they_leave_the_view_and_as_the_camera_flies_inside() {
        let scene = scene();
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.8,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 40,
            easing: Easing::Linear,
            ..Shot::default()
        };
        // One small object near the image's edge, which the camera flying
        // in leaves behind, and one around the focus point that grows past
        // the frame.
        let marks = vec![
            mark("edge", 60.0, 150.0, 3.0),
            mark("core", 199.5, 149.5, 80.0),
        ];
        let mut overlay = Overlay::new(marks, None, 1.0, 1.0, (200, 150)).unwrap();
        overlay.plan(&shot, &scene, 10.0);
        let (first, last) = (&overlay.shown[0], &overlay.shown[39]);
        assert!(first[0] > 0.9 && first[1] > 0.9, "{first:?}");
        assert!(last[0] < 0.01, "{last:?}");
        assert!(last[1] < 0.01, "{last:?}");
        assert!(overlay.within[39][1] > 0.9 && overlay.within[0][1] < 0.01);
        assert_eq!(overlay.slots, vec![None, Some(0)]);
        // And it draws.
        let mut canvas = RgbImage::new(200, 150);
        overlay.draw(0, &shot.view(&scene, 0), &mut canvas);
        assert!(canvas.pixels().any(|pixel| pixel[0] > 128));
    }

    #[test]
    fn a_label_covering_a_more_prominent_one_gives_way_and_density_limits_the_rest() {
        let scene = scene();
        let shot = Shot {
            focus: (199.5, 149.5),
            dolly: 0.0,
            truck: (0.0, 0.0),
            width: 200,
            height: 150,
            frames: 2,
            ..Shot::default()
        };
        let view = shot.view(&scene, 0);
        // Two objects at nearly the same place: the larger is the more
        // prominent, and the smaller's label would cover its label.
        let mut marks = vec![
            mark("big", 200.0, 150.0, 8.0),
            mark("small", 202.0, 150.0, 2.0),
        ];
        let overlay = Overlay::new(marks.clone(), None, 1.0, 10.0, (200, 150)).unwrap();
        let (show, _) = overlay.wanted(&shot, &view);
        assert!(show[0] > 0.0 && show[1] == 0.0, "{show:?}");
        // Spread out, with density 0 only the four most prominent show,
        // and the caller's own labels besides.
        marks = (0..6)
            .map(|index| {
                mark(
                    &format!("o{index}"),
                    120.0 + 30.0 * index as f64,
                    60.0 + 30.0 * (index % 2) as f64 * 3.0,
                    10.0 - index as f64,
                )
            })
            .collect();
        let mut custom = mark("mine", 200.0, 260.0, 0.0);
        custom.object = None;
        marks.push(custom);
        let overlay = Overlay::new(marks, None, 0.0, 10.0, (200, 150)).unwrap();
        let (show, _) = overlay.wanted(&shot, &view);
        let shown: Vec<bool> = show.iter().map(|&value| value > 0.0).collect();
        assert_eq!(
            shown,
            [true, true, true, true, false, false, true],
            "{show:?}"
        );
    }
}
