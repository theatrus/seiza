use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cooperative cancellation for builds that run over many input frames.
///
/// A build that accepts a signal checks it between inputs and returns
/// [`crate::Error::Cancelled`] once the caller asks it to stop. Partial work is
/// dropped: nothing is written and no result is returned.
///
/// The check runs between frames, so cancellation takes effect within one
/// frame's work rather than instantly.
#[derive(Clone)]
pub struct CancelSignal(Arc<dyn Fn() -> bool + Send + Sync>);

impl CancelSignal {
    /// Build a signal from any predicate, for example a channel or a flag
    /// inside a caller's own job registry.
    pub fn new(is_cancelled: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(is_cancelled))
    }

    /// True once the caller has asked for the work to stop.
    pub fn is_cancelled(&self) -> bool {
        (self.0)()
    }
}

impl From<Arc<AtomicBool>> for CancelSignal {
    fn from(flag: Arc<AtomicBool>) -> Self {
        Self::new(move || flag.load(Ordering::Relaxed))
    }
}

impl fmt::Debug for CancelSignal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately does not call the predicate: a Debug print must not run
        // caller code that may take a lock.
        formatter.write_str("CancelSignal")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flag_reads_through_the_signal() {
        let flag = Arc::new(AtomicBool::new(false));
        let signal = CancelSignal::from(Arc::clone(&flag));
        assert!(!signal.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(signal.is_cancelled());
    }
}
