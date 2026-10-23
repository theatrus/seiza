//! Frame-sized sample buffers kept between frames.
//!
//! A frame's decoded image, its clip mask and its resampled copy are each a
//! few hundred megabytes on a large sensor, and the system allocator maps
//! memory that large afresh and returns it when it is freed: the kernel then
//! faults in and zeroes every page of the next frame's buffer, and unmapping
//! one interrupts every core. On the 200-frame 61 MP Pleiades stack that was
//! 46.8 million page faults and 268 of 736 CPU-s in the kernel on an i5-1340P,
//! and 26-30 of 120 CPU-s on an M4 Max. A buffer a finished frame gives back
//! here is taken by the next frame instead, its pages still mapped.

use std::sync::{Mutex, MutexGuard};

/// Buffers smaller than this go back to the allocator, which serves them
/// from memory it already holds.
const MINIMUM_BYTES: usize = 16 << 20;

/// Bytes of idle buffers kept at most: enough for the frames a pipeline has
/// between preparing one and taking the next, about a dozen 61 MP frames.
const MAXIMUM_BYTES: usize = 3 << 30;

static POOL: Mutex<Pool> = Mutex::new(Pool {
    buffers: Vec::new(),
    capacity_bytes: MAXIMUM_BYTES,
});

fn pool() -> MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// An empty vector with room for `len` samples: one a finished frame gave
/// back, when one fits without wasting half of itself, or else a new one.
pub(crate) fn take(len: usize) -> Vec<f32> {
    if len.saturating_mul(4) < MINIMUM_BYTES {
        return Vec::with_capacity(len);
    }
    let found = pool().take(len);
    found.unwrap_or_else(|| Vec::with_capacity(len))
}

/// Keep `buffer` for a later [`take`], if it is large enough to be worth
/// keeping and the pool has room for it.
pub(crate) fn give(buffer: Vec<f32>) {
    let refused = pool().give(buffer);
    // Freed outside the lock: unmapping a large buffer takes a while.
    drop(refused);
}

/// Return every kept buffer to the allocator, when a run that prepared
/// frames ends.
pub(crate) fn clear() {
    let buffers = std::mem::take(&mut pool().buffers);
    drop(buffers);
}

/// Clears the pool when dropped, at the end of a run that prepared frames.
pub(crate) struct Scope;

impl Drop for Scope {
    fn drop(&mut self) {
        clear();
    }
}

struct Pool {
    buffers: Vec<Vec<f32>>,
    capacity_bytes: usize,
}

impl Pool {
    fn take(&mut self, len: usize) -> Option<Vec<f32>> {
        let index = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= len && buffer.capacity() / 2 <= len)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index)?;
        let mut buffer = self.buffers.swap_remove(index);
        buffer.clear();
        Some(buffer)
    }

    /// Keep `buffer`, or hand it back when it is too small to keep or there
    /// is no room for it.
    fn give(&mut self, buffer: Vec<f32>) -> Option<Vec<f32>> {
        let bytes = buffer.capacity().saturating_mul(4);
        let held: usize = self
            .buffers
            .iter()
            .map(|buffer| buffer.capacity() * 4)
            .sum();
        if bytes < MINIMUM_BYTES || held + bytes > self.capacity_bytes {
            return Some(buffer);
        }
        self.buffers.push(buffer);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_given_back_is_taken_again_only_where_it_fits() {
        let len = MINIMUM_BYTES / 4;
        let mut pool = Pool {
            buffers: Vec::new(),
            capacity_bytes: 3 * MINIMUM_BYTES,
        };
        let mut buffer = Vec::with_capacity(len);
        buffer.resize(len, 1.0);
        let address = buffer.as_ptr();
        assert!(pool.give(buffer).is_none());
        // Too large for what it would serve, and too small to serve.
        assert!(pool.take(len / 4).is_none());
        assert!(pool.take(len + 1).is_none());
        let again = pool.take(len - 1).unwrap();
        assert_eq!(again.as_ptr(), address);
        assert!(again.is_empty() && again.capacity() >= len);
        assert!(pool.take(len).is_none());
        // Small buffers, and buffers past the pool's bytes, are refused.
        assert!(pool.give(vec![0.0; 16]).is_some());
        assert!(pool.give(again).is_none());
        assert!(pool.give(Vec::with_capacity(2 * len)).is_none());
        assert!(pool.give(Vec::with_capacity(len)).is_some());
    }
}
