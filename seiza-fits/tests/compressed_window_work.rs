//! A one-pixel window must not allocate bookkeeping for every tile.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn header(cards: &[String]) -> Vec<u8> {
    let mut bytes = cards
        .iter()
        .map(String::as_str)
        .chain(["END"])
        .flat_map(|card| format!("{card:<80}").into_bytes())
        .collect::<Vec<_>>();
    bytes.resize(bytes.len().next_multiple_of(2880), b' ');
    bytes
}

#[test]
fn one_pixel_does_not_allocate_for_unrelated_tiles() {
    const TILES: usize = 100_000;
    let file = tempfile::Builder::new().suffix(".fits").tempfile().unwrap();
    let mut output = std::io::BufWriter::new(file.as_file());
    output
        .write_all(&header(&[
            "SIMPLE  =                    T".into(),
            "BITPIX  =                    8".into(),
            "NAXIS   =                    0".into(),
        ]))
        .unwrap();
    output
        .write_all(&header(&[
            "XTENSION= 'BINTABLE'".into(),
            "BITPIX  =                    8".into(),
            "NAXIS   =                    2".into(),
            "NAXIS1  =                    8".into(),
            format!("NAXIS2  = {TILES:>20}"),
            "PCOUNT  =                    2".into(),
            "GCOUNT  =                    1".into(),
            "TFIELDS =                    1".into(),
            "TTYPE1  = 'COMPRESSED_DATA'".into(),
            "TFORM1  = '1PB'".into(),
            "ZIMAGE  =                    T".into(),
            "ZBITPIX =                   16".into(),
            "ZNAXIS  =                    2".into(),
            format!("ZNAXIS1 = {TILES:>20}"),
            "ZNAXIS2 =                    1".into(),
            "ZTILE1  =                    1".into(),
            "ZTILE2  =                    1".into(),
            "ZCMPTYPE= 'NOCOMPRESS'".into(),
        ]))
        .unwrap();
    // Every tile refers to the same two-byte heap array, so the image is
    // valid as a whole, not only at the pixel the window reads.
    for _ in 0..TILES {
        output.write_all(&[0, 0, 0, 2, 0, 0, 0, 0]).unwrap();
    }
    output.write_all(&42_i16.to_be_bytes()).unwrap();
    output.flush().unwrap();
    drop(output);
    let length = file.as_file().metadata().unwrap().len();
    file.as_file()
        .set_len(length.next_multiple_of(2880))
        .unwrap();

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let window = seiza_fits::read_window(file.path(), TILES - 1, 0, 1, 1).unwrap();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    eprintln!("one-pixel window: {allocations} allocations");
    assert_eq!(window.values, [42.0]);
    assert!(
        allocations < 1024,
        "one pixel made {allocations} allocations"
    );
}
