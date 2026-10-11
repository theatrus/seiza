//! Anti-aliased lines, ellipses and text for marking images: the labels
//! Seiza draws on sky maps and over parallax videos.
//!
//! Shapes and text go into a [`Mask`] of coverage, one per colour, which is
//! then laid over an [`RgbImage`] once, so overlapping strokes never
//! double-blend; [`Mask::dilated`] makes a soft dark halo to lay under it.
//! Text uses the Inter typeface, embedded (SIL Open Font License 1.1; see
//! `fonts/LICENSE-Inter.txt`, which must ship with anything that embeds
//! this crate).

use ab_glyph::{Font, FontRef, InvalidFont, PxScale, ScaleFont, point};
use image::{Rgb, RgbImage};
use std::ops::RangeInclusive;

const REGULAR_TTF: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
const SEMIBOLD_TTF: &[u8] = include_bytes!("../fonts/Inter-SemiBold.ttf");

/// Coverage mask for one colour layer over a `width` x `height` area whose
/// top-left pixel is `origin` in the target image, composited once so
/// overlapping strokes never double-blend.
pub struct Mask {
    origin: (i64, i64),
    width: usize,
    height: usize,
    coverage: Vec<f32>,
}

impl Mask {
    pub fn new(width: u32, height: u32) -> Self {
        Self::at((0, 0), width as usize, height as usize)
    }

    pub fn at(origin: (i64, i64), width: usize, height: usize) -> Self {
        Self {
            origin,
            width,
            height,
            coverage: vec![0.0; width * height],
        }
    }

    /// The columns and rows of the mask, in target pixels, that a box from
    /// `(x0, y0)` to `(x1, y1)` touches.
    pub fn span(
        &self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
    ) -> (RangeInclusive<i64>, RangeInclusive<i64>) {
        let (ox, oy) = self.origin;
        (
            (x0.floor() as i64).max(ox)..=(x1.ceil() as i64).min(ox + self.width as i64 - 1),
            (y0.floor() as i64).max(oy)..=(y1.ceil() as i64).min(oy + self.height as i64 - 1),
        )
    }

    pub fn set(&mut self, x: i64, y: i64, value: f32) {
        let (x, y) = (x - self.origin.0, y - self.origin.1);
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let cell = &mut self.coverage[y as usize * self.width + x as usize];
        *cell = cell.max(value.clamp(0.0, 1.0));
    }

    /// An anti-aliased line of `width` pixels.
    pub fn stroke(&mut self, p: (f64, f64), q: (f64, f64), width: f64) {
        let half = width / 2.0;
        let reach = half + 1.0;
        let (columns, rows) = self.span(
            p.0.min(q.0) - reach,
            p.1.min(q.1) - reach,
            p.0.max(q.0) + reach,
            p.1.max(q.1) + reach,
        );
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let length_sq = dx * dx + dy * dy;
        for y in rows {
            for x in columns.clone() {
                let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                let t = if length_sq > 0.0 {
                    (((cx - p.0) * dx + (cy - p.1) * dy) / length_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let distance = (cx - (p.0 + t * dx)).hypot(cy - (p.1 + t * dy));
                let value = (half + 0.5 - distance) as f32;
                if value > 0.0 {
                    self.set(x, y, value);
                }
            }
        }
    }

    pub fn polyline(&mut self, points: &[(f64, f64)], width: f64) {
        for pair in points.windows(2) {
            self.stroke(pair[0], pair[1], width);
        }
    }

    /// A ring of `radius` and line `width`; a radius of zero fills a disc.
    pub fn ring(&mut self, center: (f64, f64), radius: f64, width: f64) {
        let reach = radius + width / 2.0 + 1.0;
        let (columns, rows) = self.span(
            center.0 - reach,
            center.1 - reach,
            center.0 + reach,
            center.1 + reach,
        );
        for y in rows {
            for x in columns.clone() {
                let distance = (x as f64 + 0.5 - center.0).hypot(y as f64 + 0.5 - center.1);
                let value = (width / 2.0 + 0.5 - (distance - radius).abs()) as f32;
                if value > 0.0 {
                    self.set(x, y, value);
                }
            }
        }
    }

    pub fn disc(&mut self, center: (f64, f64), radius: f64) {
        self.ring(center, radius / 2.0, radius);
    }

    pub fn ellipse(
        &mut self,
        center: (f64, f64),
        semi_major: f64,
        semi_minor: f64,
        angle_deg: f64,
        width: f64,
    ) {
        let segments = ((semi_major * 0.5) as usize).clamp(48, 720);
        let points =
            ellipse_points(center, semi_major, semi_minor, angle_deg, segments).collect::<Vec<_>>();
        self.polyline(&points, width);
    }

    /// Fade coverage to nothing over `fade` pixels above `floor(x)`.
    /// Returns whether anything drawn was dimmed.
    pub fn fade_below(&mut self, fade: f64, floor: impl Fn(f64) -> f64) -> bool {
        let mut dimmed = false;
        for x in 0..self.width {
            let limit = floor((self.origin.0 + x as i64) as f64 + 0.5) - self.origin.1 as f64;
            let start = ((limit - fade).max(0.0) as usize).min(self.height);
            for y in start..self.height {
                let factor = ((limit - (y as f64 + 0.5)) / fade).clamp(0.0, 1.0) as f32;
                let cell = &mut self.coverage[y * self.width + x];
                if *cell > 0.0 && factor < 1.0 {
                    dimmed = true;
                    *cell *= factor;
                }
            }
        }
        dimmed
    }

    /// Take the greater coverage of this mask and `other`, cell by cell;
    /// both must cover the same area.
    pub fn absorb(&mut self, other: &Mask) {
        for (cell, &value) in self.coverage.iter_mut().zip(&other.coverage) {
            *cell = cell.max(value);
        }
    }

    /// Spread the mask by `radius` pixels with a soft edge, for a halo.
    pub fn dilated(&self, radius: f64) -> Mask {
        let r = radius.ceil() as i64;
        let mut out = Mask::at(self.origin, self.width, self.height);
        let offsets = (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
            .filter_map(|(dx, dy)| {
                let weight = (radius + 0.5 - (dx as f64).hypot(dy as f64)).clamp(0.0, 1.0) as f32;
                (weight > 0.0).then_some((dx, dy, weight))
            })
            .collect::<Vec<_>>();
        for y in 0..self.height as i64 {
            for x in 0..self.width as i64 {
                let value = self.coverage[y as usize * self.width + x as usize];
                if value <= 0.0 {
                    continue;
                }
                for &(dx, dy, weight) in &offsets {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
                        continue;
                    }
                    let cell = &mut out.coverage[ny as usize * self.width + nx as usize];
                    *cell = cell.max(value * weight);
                }
            }
        }
        out
    }

    pub fn composite(&self, canvas: &mut RgbImage, color: Rgb<u8>, alpha: f32) {
        let (canvas_width, canvas_height) = canvas.dimensions();
        for (index, &value) in self.coverage.iter().enumerate() {
            if value <= 0.0 {
                continue;
            }
            let x = self.origin.0 + (index % self.width) as i64;
            let y = self.origin.1 + (index / self.width) as i64;
            if x < 0 || y < 0 || x >= canvas_width as i64 || y >= canvas_height as i64 {
                continue;
            }
            let a = value * alpha;
            let pixel = canvas.get_pixel_mut(x as u32, y as u32);
            for channel in 0..3 {
                let blended = pixel[channel] as f32 * (1.0 - a) + color[channel] as f32 * a;
                pixel[channel] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

pub struct Fonts<'a> {
    pub regular: FontRef<'a>,
    pub semibold: FontRef<'a>,
}

impl Fonts<'static> {
    /// The embedded Inter Regular and SemiBold.
    pub fn load() -> Result<Self, InvalidFont> {
        Ok(Self {
            regular: FontRef::try_from_slice(REGULAR_TTF)?,
            semibold: FontRef::try_from_slice(SEMIBOLD_TTF)?,
        })
    }
}

impl Fonts<'_> {
    /// The characters of `text`, each once, that these fonts have no glyph
    /// for, such as CJK or emoji: they would be drawn as blank space.
    pub fn missing(&self, text: &str) -> Vec<char> {
        let mut missing: Vec<char> = Vec::new();
        for c in text.chars() {
            let drawn = c.is_whitespace() || c.is_control() || self.regular.glyph_id(c).0 != 0;
            if !drawn && !missing.contains(&c) {
                missing.push(c);
            }
        }
        missing
    }
}

/// Width and line height of `text` at `size` pixels, with `tracking` extra
/// pixels between letters.
pub fn measure(font: &FontRef<'_>, size: f64, tracking: f64, text: &str) -> (f64, f64) {
    let scaled = font.as_scaled(PxScale::from(size as f32));
    let mut width = 0.0f64;
    let mut previous = None;
    for character in text.chars() {
        let id = scaled.glyph_id(character);
        if let Some(previous) = previous {
            width += scaled.kern(previous, id) as f64 + tracking;
        }
        width += scaled.h_advance(id) as f64;
        previous = Some(id);
    }
    (width, (scaled.ascent() - scaled.descent()) as f64)
}

/// Break `text` at spaces into lines no wider than `max_width`. A word
/// wider than that gets a line of its own.
pub fn wrap(font: &FontRef<'_>, size: f64, text: &str, max_width: f64) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ') {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if line.is_empty() || measure(font, size, 0.0, &candidate).0 <= max_width {
            line = candidate;
        } else {
            lines.push(std::mem::replace(&mut line, word.to_string()));
        }
    }
    lines.push(line);
    lines
}

/// Draw `text` with its top-left corner at `(x, y)` into a mask.
pub fn draw_text(
    mask: &mut Mask,
    font: &FontRef<'_>,
    size: f64,
    tracking: f64,
    (x, y): (f64, f64),
    text: &str,
) {
    let scale = PxScale::from(size as f32);
    let scaled = font.as_scaled(scale);
    let baseline = y as f32 + scaled.ascent();
    let mut caret = x as f32;
    let mut previous = None;
    for character in text.chars() {
        let id = scaled.glyph_id(character);
        if let Some(previous) = previous {
            caret += scaled.kern(previous, id) + tracking as f32;
        }
        let glyph = id.with_scale_and_position(scale, point(caret, baseline));
        caret += scaled.h_advance(id);
        previous = Some(id);
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, coverage| {
                mask.set(
                    bounds.min.x as i64 + gx as i64,
                    bounds.min.y as i64 + gy as i64,
                    coverage,
                );
            });
        }
    }
}

/// `segments + 1` points around an ellipse whose major axis lies
/// `angle_deg` from +x, the last repeating the first.
pub fn ellipse_points(
    center: (f64, f64),
    semi_major: f64,
    semi_minor: f64,
    angle_deg: f64,
    segments: usize,
) -> impl Iterator<Item = (f64, f64)> {
    let (sin_r, cos_r) = angle_deg.to_radians().sin_cos();
    (0..=segments).map(move |i| {
        let t = i as f64 / segments as f64 * std::f64::consts::TAU;
        let (lx, ly) = (semi_major * t.cos(), semi_minor * t.sin());
        (
            center.0 + lx * cos_r - ly * sin_r,
            center.1 + lx * sin_r + ly * cos_r,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_glyphs_are_named_once() {
        let fonts = Fonts::load().unwrap();
        assert!(fonts.missing("NGC 7822 · Sh2-170 α Cyg").is_empty());
        assert_eq!(fonts.missing("漢字 漢"), ['漢', '字']);
        // The embedded subset holds Latin and Greek, not Cyrillic.
        assert_eq!(fonts.missing("Ок").len(), 2);
    }

    #[test]
    fn text_measures_and_draws() {
        let fonts = Fonts::load().unwrap();
        let (w, h) = measure(&fonts.semibold, 20.0, 0.0, "Polaris");
        assert!(w > 50.0 && w < 90.0, "{w}");
        assert!(h > 18.0 && h < 30.0, "{h}");
        let (tracked, _) = measure(&fonts.semibold, 20.0, 2.0, "Polaris");
        assert!((tracked - w - 12.0).abs() < 1e-3);
        let mut mask = Mask::new(120, 40);
        draw_text(&mut mask, &fonts.semibold, 20.0, 0.0, (2.0, 2.0), "Polaris");
        assert!(mask.coverage.iter().filter(|&&c| c > 0.5).count() > 100);
        // A mask placed elsewhere takes the same text at its own origin.
        let mut moved = Mask::at((500, 300), 120, 40);
        draw_text(
            &mut moved,
            &fonts.semibold,
            20.0,
            0.0,
            (502.0, 302.0),
            "Polaris",
        );
        let ink = |mask: &Mask| mask.coverage.iter().sum::<f32>();
        assert!((ink(&moved) - ink(&mask)).abs() < 0.01 * ink(&mask));
    }
}
