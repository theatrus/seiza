//! Deterministic allocation budgets for the borrowed f64 statistics. Count
//! only this test thread, so the test harness cannot affect the budget.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATED: Cell<Option<usize>> = const { Cell::new(None) };
}

struct Counting;

// SAFETY: allocation and deallocation are delegated unchanged to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let _ = ALLOCATED.try_with(|count| {
                if let Some(bytes) = count.get() {
                    count.set(Some(bytes.saturating_add(layout.size())));
                }
            });
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocation<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATED.with(|count| count.set(Some(0)));
    let result = operation();
    let bytes = ALLOCATED.with(|count| count.replace(None).unwrap());
    (result, bytes)
}

#[test]
fn statistics_need_at_most_one_sample_buffer() {
    let mut values: Vec<_> = (0..65536).map(|i| ((i * 7919) % 65536) as f64).collect();
    let sample_bytes = std::mem::size_of_val(values.as_slice());
    let (median, bytes) = allocation(|| seiza_stats::median_f64(&values));
    assert_eq!(median, Some(32767.5));
    assert!(bytes <= sample_bytes, "median allocated {bytes} bytes");

    let (sigma, bytes) = allocation(|| seiza_stats::robust_sigma_f64(&values, 32767.5));
    assert_eq!(sigma, Some(16384.0 * seiza_stats::NORMAL_MAD_SCALE));
    assert!(bytes <= sample_bytes, "sigma allocated {bytes} bytes");

    values.sort_unstable_by(f64::total_cmp);
    for _ in 0..2 {
        let (median, bytes) = allocation(|| seiza_stats::median_f64(&values));
        assert_eq!(median, Some(32767.5));
        assert_eq!(bytes, 0, "a monotonic median needs no scratch buffer");
        values.reverse();
    }
    let (empty, bytes) = allocation(|| {
        (
            seiza_stats::median_f64(&[]),
            seiza_stats::robust_sigma_f64(&[], 0.0),
        )
    });
    assert_eq!(empty, (None, None));
    assert_eq!(bytes, 0);
}
