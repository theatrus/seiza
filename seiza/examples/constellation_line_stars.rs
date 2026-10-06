//! Regenerate `data/constellation-lines/line-stars.tsv`: the position and
//! magnitude of every Bright Star Catalogue star the embedded constellation
//! lines use, read from a seiza star-identifier sidecar (`hr:NNN` entries,
//! which come from the Bright Star Catalogue, CDS V/50, propagated to the
//! sidecar's epoch).
//!
//! ```text
//! cargo run -p seiza --example constellation_line_stars -- \
//!     /path/to/stars-lite-tycho2.ids.bin > seiza/data/constellation-lines/line-stars.tsv
//! ```

use std::collections::BTreeSet;
use std::path::PathBuf;

use seiza::constellations::parse_lines_csv;
use seiza::star_ids::{StarIdentifier, StarIdentifierCatalog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: constellation_line_stars <stars.ids.bin>")?,
    );
    let catalog = StarIdentifierCatalog::open(&path)?;
    let csv = include_str!("../data/constellation-lines/ConstellationLines.csv");
    let stars = parse_lines_csv(csv)?
        .into_iter()
        .flat_map(|line| line.stars)
        .collect::<BTreeSet<_>>();
    println!("# Positions of the Bright Star Catalogue stars used by ConstellationLines.csv.");
    println!(
        "# Source: Bright Star Catalogue, 5th Revised Ed. (Hoffleit & Warren 1991, CDS V/50),"
    );
    println!(
        "# J2000 positions propagated with BSC proper motions to epoch {:.1}, via the seiza",
        catalog.epoch()
    );
    println!(
        "# star-identifier sidecar. Regenerate with seiza/examples/constellation_line_stars.rs."
    );
    println!("# Columns: hr, ra_deg, dec_deg, vmag");
    for hr in stars {
        let matches = catalog.lookup(StarIdentifier::HarvardRevised(hr));
        let star = matches
            .first()
            .ok_or_else(|| format!("HR {hr} is not in {}", path.display()))?;
        println!("{hr}\t{:.6}\t{:.6}\t{:.2}", star.ra, star.dec, star.mag);
    }
    Ok(())
}
