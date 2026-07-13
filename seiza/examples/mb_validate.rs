//! Print a named minor body's geocentric position at a JD.
use seiza::minor_bodies::MinorBodyCatalog;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let catalog = MinorBodyCatalog::open(std::path::Path::new(&args[1])).unwrap();
    let jd: f64 = args[3].parse().unwrap();
    for body in catalog.bodies() {
        if body.name.contains(args[2].as_str())
            && let Some((ra, dec, mag, delta)) = MinorBodyCatalog::position_at(body, jd)
        {
            println!(
                "{}: RA {:.5}  Dec {:+.5}  V~{:.1}  delta {:.3} AU",
                body.name, ra, dec, mag, delta
            );
        }
    }
}
