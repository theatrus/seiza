//! Synthetic GZIP_2 fixtures, including partial edge tiles.
use std::io::Write;

fn unit(cards: &[String], data: &[u8]) -> Vec<u8> {
    let mut bytes = cards
        .iter()
        .map(String::as_str)
        .chain(["END"])
        .flat_map(|card| format!("{card:<80}").into_bytes())
        .collect::<Vec<_>>();
    bytes.resize(bytes.len().next_multiple_of(2880), b' ');
    bytes.extend_from_slice(data);
    bytes.resize(bytes.len().next_multiple_of(2880), 0);
    bytes
}

pub fn encode(image: &seiza_stacking::LinearImage) -> Vec<u8> {
    assert_eq!(image.channels, 1);
    let mut descriptors = Vec::new();
    let mut heap = Vec::new();
    for y in (0..image.height).step_by(31) {
        for x in (0..image.width).step_by(37) {
            let raw = (y..(y + 31).min(image.height))
                .flat_map(|row| {
                    image.data[row * image.width + x..row * image.width + (x + 37).min(image.width)]
                        .iter()
                        .flat_map(|sample| sample.to_be_bytes())
                })
                .collect::<Vec<_>>();
            let shuffled = (0..4)
                .flat_map(|byte| raw.iter().skip(byte).step_by(4).copied())
                .collect::<Vec<_>>();
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&shuffled).unwrap();
            let compressed = encoder.finish().unwrap();
            descriptors.extend_from_slice(&(compressed.len() as i32).to_be_bytes());
            descriptors.extend_from_slice(&(heap.len() as i32).to_be_bytes());
            heap.extend(compressed);
        }
    }
    let cards = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        "NAXIS1  =                    8".into(),
        format!("NAXIS2  = {:>20}", descriptors.len() / 8),
        format!("PCOUNT  = {:>20}", heap.len()),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        "TFORM1  = '1PB'".into(),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {:>20}", image.width),
        format!("ZNAXIS2 = {:>20}", image.height),
        "ZTILE1  =                   37".into(),
        "ZTILE2  =                   31".into(),
        "ZCMPTYPE= 'GZIP_2'".into(),
    ];
    let primary = [
        "SIMPLE  =                    T".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    0".into(),
    ];
    let mut bytes = unit(&primary, &[]);
    descriptors.extend(heap);
    bytes.extend(unit(&cards, &descriptors));
    bytes
}
