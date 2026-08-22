//! Runtime-gated diagnostic output for the detection pipeline.
//!
//! Ported with the detectors from PSF Guard, keeping the same shape: a host
//! wires its own verbose flag in through [`init_debug`], and the pipeline's
//! step-by-step commentary prints only when asked. Off means zero cost beyond
//! one relaxed atomic load per site.

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

/// Turn diagnostic output on or off for the whole process.
pub fn init_debug(verbose: bool) {
    DEBUG_ENABLED.store(verbose, Ordering::Relaxed);
}

/// Whether diagnostic output is currently enabled.
pub fn is_debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

/// Print one line of detection diagnostics when enabled.
#[macro_export]
macro_rules! debug_detection {
    ($($arg:tt)*) => {
        if $crate::debug::is_debug_enabled() {
            eprintln!("DETECT: {}", format!($($arg)*));
        }
    }
}
