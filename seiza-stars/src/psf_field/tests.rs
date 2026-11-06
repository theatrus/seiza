use super::*;

/// A xorshift generator with Gaussian samples.
struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// The true PSF of the synthetic fields: an elliptical Moffat whose FWHM
/// grows from left to right and whose `e1` grows from top to bottom, plus
/// a faint wide halo. With a `tilt`, the focal plane is tilted instead.
#[derive(Clone, Copy)]
struct TruePsf {
    fwhm_left: f64,
    fwhm_right: f64,
    e1_top: f64,
    e1_bottom: f64,
    beta: f64,
    halo_fraction: f64,
    halo_sigma: f64,
    tilt: Option<Tilt>,
}

/// A sensor tilted against the focal plane: defocus grows along a line
/// across the frame, through best focus at `(zero_x, zero_y)`. The FWHM is
/// `hypot(best, slope · distance)`, and astigmatism elongates stars along
/// the defocus direction on one side of focus and across it on the other.
#[derive(Clone, Copy)]
struct Tilt {
    best: f64,
    zero_x: f64,
    zero_y: f64,
    /// FWHM of defocus per pixel along the direction.
    slope: f64,
    /// The direction defocus grows in, from `+x` towards `+y`.
    direction: f64,
    /// `χ` per pixel of defocus FWHM.
    astigmatism: f64,
}

impl TruePsf {
    fn plain() -> Self {
        Self {
            fwhm_left: 2.6,
            fwhm_right: 3.8,
            e1_top: -0.08,
            e1_bottom: 0.12,
            beta: 3.0,
            halo_fraction: 0.0,
            halo_sigma: 6.0,
            tilt: None,
        }
    }

    /// `(a, b, θ)` at a point: the Moffat's α along its axis at `θ` and
    /// across it.
    fn at(&self, x: f64, y: f64, width: usize, height: usize) -> (f64, f64, f64) {
        let (fwhm, chi, theta) = match self.tilt {
            None => {
                let tx = x / (width - 1) as f64;
                let ty = y / (height - 1) as f64;
                let fwhm = self.fwhm_left + (self.fwhm_right - self.fwhm_left) * tx;
                let e1 = self.e1_top + (self.e1_bottom - self.e1_top) * ty;
                (fwhm, e1, 0.0)
            }
            Some(tilt) => {
                let (cos, sin) = (tilt.direction.cos(), tilt.direction.sin());
                let defocus = tilt.slope * ((x - tilt.zero_x) * cos + (y - tilt.zero_y) * sin);
                let chi = (tilt.astigmatism * defocus).clamp(-0.4, 0.4);
                (tilt.best.hypot(defocus), chi, tilt.direction)
            }
        };
        // χ = (a² − b²)/(a² + b²), with (a + b)/2 · factor = FWHM; a
        // negative χ puts the major axis across θ.
        let ratio = ((1.0 - chi.abs()) / (1.0 + chi.abs())).sqrt();
        let factor = 2.0 * (2.0_f64.powf(1.0 / self.beta) - 1.0).sqrt();
        let mean_alpha = fwhm / factor;
        let major = 2.0 * mean_alpha / (1.0 + ratio);
        let minor = major * ratio;
        if chi >= 0.0 {
            (major, minor, theta)
        } else {
            (minor, major, theta)
        }
    }

    /// The PSF's value per unit flux at offset `(dx, dy)`, for axes
    /// `(a, b)` at angle `θ`.
    fn value(&self, dx: f64, dy: f64, (ax, ay, theta): (f64, f64, f64)) -> f64 {
        let norm = (self.beta - 1.0) / (std::f64::consts::PI * ax * ay);
        let (cos, sin) = (theta.cos(), theta.sin());
        let (xp, yp) = (dx * cos + dy * sin, -dx * sin + dy * cos);
        let core = norm * (1.0 + xp * xp / (ax * ax) + yp * yp / (ay * ay)).powf(-self.beta);
        let s2 = self.halo_sigma * self.halo_sigma;
        let halo = (-(dx * dx + dy * dy) / (2.0 * s2)).exp() / (2.0 * std::f64::consts::PI * s2);
        (1.0 - self.halo_fraction) * core + self.halo_fraction * halo
    }

    /// The pixel-integrated stamp centred on its middle pixel, at image
    /// position `(x, y)`.
    fn stamp(&self, x: f64, y: f64, width: usize, height: usize, size: usize) -> Vec<f64> {
        let axes = self.at(x, y, width, height);
        let centre = (size / 2) as f64;
        let mut values: Vec<f64> = (0..size * size)
            .map(|index| {
                let (u, v) = (
                    (index % size) as f64 - centre,
                    (index / size) as f64 - centre,
                );
                integrate(|dx, dy| self.value(dx, dy, axes), u, v)
            })
            .collect();
        let total: f64 = values.iter().sum();
        values.iter_mut().for_each(|value| *value /= total);
        values
    }
}

/// The mean of `f` over the pixel centred on `(u, v)`, from 4 × 4 samples.
fn integrate(f: impl Fn(f64, f64) -> f64, u: f64, v: f64) -> f64 {
    let mut sum = 0.0;
    for j in 0..4 {
        for i in 0..4 {
            sum += f(u - 0.375 + 0.25 * i as f64, v - 0.375 + 0.25 * j as f64);
        }
    }
    sum / 16.0
}

struct Field {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<f32>,
    stars: Vec<(f64, f64, f64)>,
}

/// Stars on a jittered grid `spacing` px apart, fluxes spread over two
/// decades, `saturated` very bright stars clipped at 60000, Gaussian noise
/// of σ 2 on a background of 100. `psfs` gives each channel's PSF.
fn field(
    width: usize,
    height: usize,
    spacing: usize,
    saturated: usize,
    psfs: &[TruePsf],
    seed: u64,
) -> Field {
    field_with(width, height, spacing, saturated, 4.0e6, psfs, seed)
}

/// [`field`], with the saturated stars at `saturated_flux`.
fn field_with(
    width: usize,
    height: usize,
    spacing: usize,
    saturated: usize,
    saturated_flux: f64,
    psfs: &[TruePsf],
    seed: u64,
) -> Field {
    let channels = psfs.len();
    let mut random = Random(seed);
    let mut stars = Vec::new();
    let margin = 30;
    let mut y = margin;
    while y + margin < height {
        let mut x = margin;
        while x + margin < width {
            let jx = x as f64 + (random.uniform() - 0.5) * spacing as f64 * 0.5;
            let jy = y as f64 + (random.uniform() - 0.5) * spacing as f64 * 0.5;
            let flux = 3_000.0 * 10_f64.powf(1.8 * random.uniform());
            stars.push((jx, jy, flux));
            x += spacing;
        }
        y += spacing;
    }
    let count = stars.len();
    for index in 0..saturated.min(count) {
        // Every seventh star becomes a saturated one.
        let star = &mut stars[(index * 7 + 3) % count];
        star.2 = saturated_flux;
    }
    let mut data = vec![0.0_f64; width * height * channels];
    for (channel, psf) in psfs.iter().enumerate() {
        for &(sx, sy, flux) in &stars {
            let axes = psf.at(sx, sy, width, height);
            let reach = 40_isize;
            let (cx, cy) = (sx.round() as isize, sy.round() as isize);
            for py in (cy - reach).max(0)..(cy + reach).min(height as isize) {
                for px in (cx - reach).max(0)..(cx + reach).min(width as isize) {
                    let (dx, dy) = (px as f64 - sx, py as f64 - sy);
                    let value = if dx.abs() < 8.0 && dy.abs() < 8.0 {
                        integrate(|u, v| psf.value(u, v, axes), dx, dy)
                    } else {
                        psf.value(dx, dy, axes)
                    };
                    data[(py as usize * width + px as usize) * channels + channel] += flux * value;
                }
            }
        }
    }
    let data = data
        .into_iter()
        .map(|value| ((value + 100.0 + 2.0 * random.normal()).min(60_000.0)) as f32)
        .collect();
    Field {
        width,
        height,
        channels,
        data,
        stars,
    }
}

fn measure(field: &Field, options: &PsfFieldOptions) -> Result<PsfField, PsfFieldError> {
    PsfField::measure(
        &field.data,
        field.width,
        field.height,
        field.channels,
        options,
    )
}

#[test]
fn recovers_a_varying_field() {
    let psf = TruePsf::plain();
    let field = field(720, 600, 36, 0, &[psf], 11);
    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    assert!(measured.star_count() >= 100, "{}", measured.star_count());
    assert_eq!(measured.stamp_size(), DEFAULT_STAMP_SIZE);
    assert!(!measured.is_global());
    assert!(measured.warnings().is_empty());
    let fit = measured.channel_fits()[0];
    assert!(fit.relative_residual < 0.1, "{fit:?}");
    for &(x, y) in &[
        (60.0, 60.0),
        (660.0, 60.0),
        (360.0, 300.0),
        (60.0, 540.0),
        (660.0, 540.0),
    ] {
        let truth = PsfStamp {
            size: 33,
            channels: 1,
            data: psf
                .stamp(x, y, field.width, field.height, 33)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        }
        .shape(0)
        .unwrap();
        let shape = measured.shape_at(0, x, y).unwrap();
        assert!(
            (shape.fwhm - truth.fwhm).abs() < 0.05 * truth.fwhm,
            "FWHM at ({x}, {y}): {shape:?} vs {truth:?}"
        );
        assert!(
            (shape.e1 - truth.e1).abs() < 0.03 && (shape.e2 - truth.e2).abs() < 0.03,
            "ellipticity at ({x}, {y}): {shape:?} vs {truth:?}"
        );
        assert!(
            (shape.beta - truth.beta).abs() < 0.25 * truth.beta,
            "β at ({x}, {y}): {shape:?} vs {truth:?}"
        );
    }
}

#[test]
fn stamps_sum_to_one_and_are_centred() {
    let field = field(480, 420, 36, 0, &[TruePsf::plain()], 5);
    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    let stamp = measured.stamp_at(240.0, 210.0);
    assert_eq!(stamp.data.len(), 33 * 33);
    let total: f64 = stamp.data.iter().map(|&v| f64::from(v)).sum();
    assert!((total - 1.0).abs() < 1e-5, "{total}");
    let (mut mx, mut my) = (0.0, 0.0);
    for (index, &value) in stamp.data.iter().enumerate() {
        mx += f64::from(value) * ((index % 33) as f64 - 16.0);
        my += f64::from(value) * ((index / 33) as f64 - 16.0);
    }
    assert!(mx.abs() < 0.05 && my.abs() < 0.05, "centroid ({mx}, {my})");
    // Far outside the image, the edge's PSF.
    assert_eq!(
        measured.stamp_at(-500.0, 210.0),
        measured.stamp_at(0.0, 210.0)
    );
}

#[test]
fn refuses_too_few_stars_unless_a_global_psf_is_allowed() {
    let field = field(300, 240, 60, 0, &[TruePsf::plain()], 3);
    let error = measure(&field, &PsfFieldOptions::default()).unwrap_err();
    let PsfFieldError::TooFewStars { usable, required } = error else {
        panic!("{error:?}");
    };
    assert_eq!(required, 30);
    assert!((GLOBAL_MINIMUM_STARS..30).contains(&usable), "{usable}");

    let options = PsfFieldOptions {
        sparse: SparseFieldPolicy::Global,
        ..PsfFieldOptions::default()
    };
    let measured = measure(&field, &options).unwrap();
    assert!(measured.is_global());
    assert_eq!(measured.component_count(), 0);
    assert_eq!(
        measured.warnings(),
        &[PsfFieldWarning::GlobalPsf {
            usable: measured.star_count(),
            required: 30
        }]
    );
    // One PSF everywhere.
    assert_eq!(
        measured.stamp_at(10.0, 10.0),
        measured.stamp_at(290.0, 230.0)
    );
}

#[test]
fn catalogue_neighbours_disqualify_stars() {
    let field = field(480, 420, 36, 0, &[TruePsf::plain()], 9);
    let plain = measure(&field, &PsfFieldOptions::default()).unwrap();
    // The catalogue lists every star, and beside half of them a faint
    // companion two magnitudes down that detection would miss.
    let mut catalogue: Vec<CatalogueStar> = field
        .stars
        .iter()
        .map(|&(x, y, flux)| CatalogueStar {
            x,
            y,
            magnitude: -2.5 * flux.log10(),
        })
        .collect();
    let companions: Vec<CatalogueStar> = field
        .stars
        .iter()
        .step_by(2)
        .map(|&(x, y, flux)| CatalogueStar {
            x: x + 6.0,
            y: y + 4.0,
            magnitude: -2.5 * flux.log10() + 2.0,
        })
        .collect();
    catalogue.extend(&companions);
    let measured = PsfField::measure_with_catalogue(
        &field.data,
        field.width,
        field.height,
        1,
        &catalogue,
        &PsfFieldOptions::default(),
    )
    .unwrap();
    assert!(measured.star_count() < plain.star_count());
    for star in measured.stars() {
        assert!(
            companions
                .iter()
                .all(
                    |companion| (companion.x - 6.0 - star.x).hypot(companion.y - 4.0 - star.y)
                        > 2.0
                ),
            "star at ({}, {}) has a companion",
            star.x,
            star.y
        );
    }
}

#[test]
fn fits_each_rgb_channel() {
    let red = TruePsf::plain();
    let blue = TruePsf {
        fwhm_left: 3.6,
        fwhm_right: 4.8,
        ..red
    };
    let field = field(480, 420, 36, 0, &[red, red, blue], 21);
    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    assert_eq!(measured.channels(), 3);
    let stamp = measured.stamp_at(240.0, 210.0);
    assert_eq!(stamp.channels, 3);
    let (r, b) = (stamp.shape(0).unwrap(), stamp.shape(2).unwrap());
    assert!(b.fwhm > r.fwhm + 0.7, "red {r:?}, blue {b:?}");
}

#[test]
fn saturated_stars_give_the_wings() {
    // A wide halo, as filter reflections give: too faint in any core star
    // to see, and still bright at the stamp's edge, where a core star's
    // background ring takes it for sky.
    let psf = TruePsf {
        halo_fraction: 0.05,
        halo_sigma: 8.0,
        ..TruePsf::plain()
    };
    let field = field(720, 600, 36, 12, &[psf], 17);
    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    let (x, y) = (360.0, 300.0);
    let stamp = measured.stamp_at(x, y);
    let size = measured.stamp_size();
    let truth = psf.stamp(x, y, field.width, field.height, size);
    // Between 9 and 15 px the halo is most of the light. The saturated
    // stars' wings, with the sky measured far outside them, restore it.
    let (mut measured_ring, mut true_ring) = (0.0, 0.0);
    for ((value, truth), r) in stamp.data.iter().zip(&truth).zip(radii(size)) {
        if (9.0..=15.0).contains(&r) {
            measured_ring += f64::from(*value);
            true_ring += truth;
        }
    }
    assert!(
        (measured_ring - true_ring).abs() < 0.08 * true_ring,
        "ring {measured_ring} against {true_ring}"
    );
    let fit = measured.channel_fits()[0];
    assert!(fit.wing_stars >= WING_MINIMUM_STARS, "{fit:?}");
    assert!(fit.wing_overlap.is_some());
    assert_eq!(
        measured
            .stars()
            .iter()
            .filter(|star| star.role == PsfStarRole::Wing)
            .count(),
        fit.wing_stars
    );
}

#[test]
fn rejects_bad_input() {
    let options = PsfFieldOptions::default();
    assert!(matches!(
        PsfField::measure(&[0.0; 9], 5, 2, 1, &options),
        Err(PsfFieldError::InvalidImage(_))
    ));
    assert!(matches!(
        PsfField::measure(&[0.0; 20], 5, 2, 2, &options),
        Err(PsfFieldError::InvalidImage(_))
    ));
    let even = PsfFieldOptions {
        stamp_size: Some(32),
        ..PsfFieldOptions::default()
    };
    assert!(matches!(
        PsfField::measure(&[0.0; 10], 5, 2, 1, &even),
        Err(PsfFieldError::InvalidOptions(_))
    ));
    assert!(matches!(
        PsfField::measure(&[0.0; 10_000], 100, 100, 1, &options),
        Err(PsfFieldError::TooFewStars { usable: 0, .. })
    ));
}

#[test]
fn words_round_trip() {
    for policy in [SparseFieldPolicy::Refuse, SparseFieldPolicy::Global] {
        assert_eq!(policy.to_string().parse::<SparseFieldPolicy>(), Ok(policy));
    }
    for role in [PsfStarRole::Core, PsfStarRole::Wing] {
        assert_eq!(role.to_string().parse::<PsfStarRole>(), Ok(role));
    }
    assert!("sometimes".parse::<SparseFieldPolicy>().is_err());
}

/// A tilted sensor: best focus runs along a line through the lower left,
/// and stars soften and stretch towards the upper right, elongated along the
/// defocus on one side of focus and across it on the other. The
/// coefficient field must follow it, not average it away.
#[test]
fn follows_a_tilted_focal_plane() {
    let psf = TruePsf {
        tilt: Some(Tilt {
            best: 2.4,
            zero_x: 300.0,
            zero_y: 380.0,
            slope: 0.0046,
            direction: (-0.4_f64).atan2(1.0),
            astigmatism: 0.08,
        }),
        ..TruePsf::plain()
    };
    let field = field(900, 600, 36, 0, &[psf], 23);
    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    // Best focus crosses the frame: no quadratic follows the V it makes.
    assert_eq!(measured.channel_fits()[0].degree, 3);
    let mut worst_fwhm: f64 = 0.0;
    let mut worst_ellipticity: f64 = 0.0;
    for &(x, y) in &[
        (60.0, 540.0),
        (300.0, 380.0),
        (450.0, 300.0),
        (840.0, 60.0),
        (60.0, 60.0),
        (840.0, 540.0),
    ] {
        let truth = PsfStamp {
            size: 33,
            channels: 1,
            data: psf
                .stamp(x, y, field.width, field.height, 33)
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        }
        .shape(0)
        .unwrap();
        let shape = measured.shape_at(0, x, y).unwrap();
        worst_fwhm = worst_fwhm.max((shape.fwhm - truth.fwhm).abs() / truth.fwhm);
        worst_ellipticity = worst_ellipticity
            .max((shape.e1 - truth.e1).abs())
            .max((shape.e2 - truth.e2).abs());
    }
    assert!(worst_fwhm < 0.07, "FWHM off by {worst_fwhm}");
    assert!(
        worst_ellipticity < 0.02,
        "ellipticity off by {worst_ellipticity}"
    );
}

/// Stars so bright that their cores bloom into flat discs 11 px across:
/// they never give the core, and still give the wings, matched outside
/// their discs.
#[test]
fn bloomed_stars_give_wings_not_cores() {
    let psf = TruePsf {
        halo_fraction: 0.03,
        halo_sigma: 5.0,
        ..TruePsf::plain()
    };
    let field = field_with(720, 600, 36, 12, 4.0e7, &[psf], 29);
    let bloomed: Vec<(f64, f64)> = field
        .stars
        .iter()
        .filter(|star| star.2 >= 4.0e7)
        .map(|star| (star.0, star.1))
        .collect();
    assert_eq!(bloomed.len(), 12);
    // The discs really are flat and wide.
    let (bx, by) = bloomed[0];
    let disc = (-8..=8)
        .filter(|dx| {
            field.data[by.round() as usize * field.width + (bx.round() as isize + dx) as usize]
                >= 60_000.0
        })
        .count();
    assert!(disc >= 9, "disc {disc} px across");

    let measured = measure(&field, &PsfFieldOptions::default()).unwrap();
    for star in measured.stars() {
        let near_bloom = bloomed
            .iter()
            .any(|(x, y)| (x - star.x).hypot(y - star.y) < 3.0);
        assert_eq!(
            near_bloom,
            star.role == PsfStarRole::Wing,
            "star at ({}, {}) as {}",
            star.x,
            star.y,
            star.role
        );
    }
    let fit = measured.channel_fits()[0];
    assert!(fit.wing_stars >= WING_MINIMUM_STARS, "{fit:?}");
    let (inner, _) = fit.wing_overlap.unwrap();
    assert!(inner > 5.0, "{fit:?}");

    let (x, y) = (360.0, 300.0);
    let size = measured.stamp_size();
    let stamp = measured.stamp_at(x, y);
    let truth = psf.stamp(x, y, field.width, field.height, size);
    let (mut measured_ring, mut true_ring) = (0.0, 0.0);
    for ((value, truth), r) in stamp.data.iter().zip(&truth).zip(radii(size)) {
        if (9.0..=15.0).contains(&r) {
            measured_ring += f64::from(*value);
            true_ring += truth;
        }
    }
    assert!(
        (measured_ring - true_ring).abs() < 0.08 * true_ring,
        "ring {measured_ring} against {true_ring}"
    );
}

#[test]
fn a_varying_field_needs_ten_stars_and_no_components_is_one_psf() {
    let field = field(240, 200, 60, 0, &[TruePsf::plain()], 3);
    let global = PsfFieldOptions {
        sparse: SparseFieldPolicy::Global,
        ..PsfFieldOptions::default()
    };
    let usable = measure(&field, &global).unwrap().star_count();
    assert!((5..10).contains(&usable), "{usable}");
    // Fewer than ten stars cannot carry a component: the field refuses, as
    // it does for too few stars, rather than return one PSF as a varying
    // field.
    let few = PsfFieldOptions {
        minimum_stars: 5,
        ..PsfFieldOptions::default()
    };
    assert_eq!(
        measure(&field, &few).unwrap_err(),
        PsfFieldError::TooFewStars {
            usable,
            required: 10
        }
    );
    let accepted = measure(
        &field,
        &PsfFieldOptions {
            sparse: SparseFieldPolicy::Global,
            ..few
        },
    )
    .unwrap();
    assert!(accepted.is_global());
    assert_eq!(
        accepted.warnings(),
        &[PsfFieldWarning::GlobalPsf {
            usable,
            required: 10
        }]
    );
    // No components asked for: one PSF, and no warning.
    let plenty = field_with(480, 420, 36, 0, 4.0e6, &[TruePsf::plain()], 5);
    let uniform = measure(
        &plenty,
        &PsfFieldOptions {
            components: 0,
            ..PsfFieldOptions::default()
        },
    )
    .unwrap();
    assert!(uniform.is_global());
    assert!(uniform.warnings().is_empty());
    assert_eq!(uniform.component_count(), 0);
    assert_eq!(uniform.channel_fits()[0].degree, 0);
}

/// The fit a channel reports is measured on the model the field returns,
/// after the wings are added, not on the core model before them.
#[test]
fn the_channel_fit_describes_the_model_with_its_wings() {
    // One PSF across the field, so a constant core model fits its core.
    let halo = TruePsf {
        fwhm_left: 3.2,
        fwhm_right: 3.2,
        e1_top: 0.0,
        e1_bottom: 0.0,
        halo_fraction: 0.05,
        halo_sigma: 8.0,
        ..TruePsf::plain()
    };
    let scene = field_with(720, 600, 36, 12, 4.0e6, &[halo], 17);
    let (width, height, size) = (scene.width, scene.height, 33);
    let plane = Plane {
        data: &scene.data,
        width,
        height,
        channels: 1,
        channel: 0,
    };
    let wing_stars: Vec<Candidate> = scene
        .stars
        .iter()
        .filter(|star| star.2 >= 4.0e6)
        .map(|&(x, y, flux)| Candidate {
            x,
            y,
            flux,
            snr: 1e4,
            fwhm: 3.2,
            saturated: true,
            brightness: 1e4,
        })
        .collect();
    let aperture = 8.0;
    let normalised = |values: Vec<f64>| {
        let flux = aperture_sum(&values, size, aperture);
        values.into_iter().map(|v| v / flux).collect::<Vec<_>>()
    };
    let positions: Vec<(f64, f64)> = (0..5)
        .flat_map(|j| (0..5).map(move |i| (100.0 + 120.0 * i as f64, 80.0 + 110.0 * j as f64)))
        .collect();
    let radius = radii(size);
    // Stars that hold the halo, and a core model that has lost it beyond
    // its aperture, as core stars too faint to show it give.
    let stamp_of = |(x, y): (f64, f64), reach: f64| StarStamp {
        u: field_coordinate(x, width),
        v: field_coordinate(y, height),
        values: normalised(halo.stamp(x, y, width, height, size))
            .into_iter()
            .zip(&radius)
            .map(|(v, &r)| if r <= reach { v } else { 0.0 })
            .collect(),
        noise: 1e-5,
        valid: None,
    };
    let core: Vec<StarStamp> = positions.iter().map(|&p| stamp_of(p, 9.0)).collect();
    let stars: Vec<StarStamp> = positions
        .iter()
        .map(|&p| stamp_of(p, f64::INFINITY))
        .collect();
    let core_refs: Vec<&StarStamp> = core.iter().collect();
    let star_refs: Vec<&StarStamp> = stars.iter().collect();
    let mut model = model::fit(&core_refs, size, 0, Some(0));
    let core_mask: Vec<bool> = radius.iter().map(|&r| r <= 4.8).collect();
    let median = |values: &[f64]| median_f64(&mut values.to_vec());
    let before = model::quality(&model, &star_refs, &core_mask);
    let (fit, used) = finish_channel(
        &mut model,
        &star_refs,
        &core_mask,
        &WingSource {
            plane: &plane,
            saturation: 48_000.0,
            stars: &wing_stars,
            radius: &radius,
            fwhm0: 3.2,
        },
        |x| field_coordinate(x, width),
        |y| field_coordinate(y, height),
    );
    assert!(used.len() >= WING_MINIMUM_STARS, "{fit:?}");
    let after = model::quality(&model, &star_refs, &core_mask);
    assert_eq!(fit.reduced_chi_squared, median(&after.chi_squared));
    assert_eq!(fit.relative_residual, median(&after.relative));
    // The wings changed the model enough that measuring before them would
    // report a different fit.
    let (old, new) = (median(&before.chi_squared), median(&after.chi_squared));
    assert!((old - new).abs() > 0.1 * old, "{old} against {new}");
}
