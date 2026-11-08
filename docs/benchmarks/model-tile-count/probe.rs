use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Measured;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Measured {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(size, Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Measured = Measured;
fn main() {
    let tiling = seiza_model::Tiling { tile: 64, overlap: 36, margin: 8, multiple: 16 };
    let before_calls = CALLS.load(Relaxed);
    let before_bytes = BYTES.load(Relaxed);
    let count = std::hint::black_box(tiling).tiles(16_000_064, 64).unwrap();
    let calls = CALLS.load(Relaxed) - before_calls;
    let bytes = BYTES.load(Relaxed) - before_bytes;
    assert_eq!(count, 1_000_001);
    println!("tiles={count} allocation_calls={calls} allocated_bytes={bytes}");
}
