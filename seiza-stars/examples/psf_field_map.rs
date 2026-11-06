//! Measure a linear image's PSF field and print its FWHM, ellipticity and
//! Moffat β on a grid across the image.
//!
//! The first argument is a FITS image, mono or planar RGB; the optional
//! second and third are the grid's columns and rows (5 and 4). Each line of
//! output is one grid point and channel: `x y channel fwhm e1 e2 beta`.
//!
//! ```sh
//! cargo run --release -p seiza-stars --example psf_field_map -- stack.fits 6 4
//! ```

use std::path::Path;

use seiza_fits::FitsImage;
use seiza_stars::{PsfField, PsfFieldOptions, PsfStarRole, SparseFieldPolicy};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments
        .next()
        .expect("usage: psf_field_map IMAGE [COLUMNS ROWS]");
    let columns: usize = arguments.next().map_or(5, |v| v.parse().expect("columns"));
    let rows: usize = arguments.next().map_or(4, |v| v.parse().expect("rows"));
    let image = FitsImage::open(Path::new(&path)).expect("readable FITS image");
    let (width, height, planes) = (image.width, image.height, image.planes);
    let planar = image.into_physical_f32();
    // Interleave planar RGB.
    let pixels = width * height;
    let data: Vec<f32> = if planes == 1 {
        planar
    } else {
        (0..pixels * planes)
            .map(|index| planar[(index % planes) * pixels + index / planes])
            .collect()
    };
    let options = PsfFieldOptions {
        sparse: SparseFieldPolicy::Global,
        ..PsfFieldOptions::default()
    };
    let started = std::time::Instant::now();
    let field = match PsfField::measure(&data, width, height, planes, &options) {
        Ok(field) => field,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    let wings = field
        .stars()
        .iter()
        .filter(|star| star.role == PsfStarRole::Wing)
        .count();
    println!(
        "# {width}x{height}x{planes}: {} core stars, {wings} wing stars, stamp {} px, {} components, {:.1} s",
        field.star_count(),
        field.stamp_size(),
        field.component_count(),
        started.elapsed().as_secs_f64()
    );
    for (channel, fit) in field.channel_fits().iter().enumerate() {
        println!(
            "# channel {channel}: degree {}, reduced chi2 {:.2}, relative residual {:.4}, {} wing stars, overlap {:?}",
            fit.degree,
            fit.reduced_chi_squared,
            fit.relative_residual,
            fit.wing_stars,
            fit.wing_overlap
        );
    }
    for warning in field.warnings() {
        println!("# warning: {warning}");
    }
    for row in 0..rows {
        for column in 0..columns {
            let x = (column as f64 + 0.5) / columns as f64 * (width - 1) as f64;
            let y = (row as f64 + 0.5) / rows as f64 * (height - 1) as f64;
            for channel in 0..planes {
                match field.shape_at(channel, x, y) {
                    Some(shape) => println!(
                        "{x:.0} {y:.0} {channel} {:.3} {:+.4} {:+.4} {:.2}",
                        shape.fwhm, shape.e1, shape.e2, shape.beta
                    ),
                    None => println!("{x:.0} {y:.0} {channel} - - - -"),
                }
            }
        }
    }
}
