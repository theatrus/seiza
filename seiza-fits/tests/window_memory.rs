//! A window read allocates for the window, not the image: a 64 × 64 read
//! from a 40000 × 40000 float image (6.4 GB of samples) stays within a few
//! hundred kilobytes of heap.
//!
//! The image is a sparse file, so this runs where creating one is cheap.
#![cfg(unix)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The system allocator, counting the bytes live at once.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
        PEAK.fetch_max(live, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn a_window_of_a_huge_image_allocates_for_the_window_alone() {
    const SIDE: usize = 40_000;
    let mut header = Vec::new();
    for card in [
        "SIMPLE  =                    T".to_string(),
        "BITPIX  =                  -32".to_string(),
        "NAXIS   =                    2".to_string(),
        format!("NAXIS1  = {SIDE:>20}"),
        format!("NAXIS2  = {SIDE:>20}"),
        "END".to_string(),
    ] {
        header.extend_from_slice(format!("{card:<80}").as_bytes());
    }
    header.resize(2880, b' ');
    let file = tempfile::Builder::new().suffix(".fits").tempfile().unwrap();
    std::fs::write(file.path(), &header).unwrap();
    // The data unit is a hole: zeros, with no blocks behind them.
    let data = (SIDE * SIDE * 4) as u64;
    std::fs::OpenOptions::new()
        .write(true)
        .open(file.path())
        .unwrap()
        .set_len(2880 + data.next_multiple_of(2880))
        .unwrap();

    let before = LIVE.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    let window = seiza_fits::read_window(file.path(), SIDE - 100, SIDE - 70, 64, 64).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - before;

    assert_eq!((window.width, window.height), (64, 64));
    assert!(window.values.iter().all(|&value| value == 0.0));
    assert!(peak < 512 * 1024, "a 64 × 64 window allocated {peak} bytes");
}
