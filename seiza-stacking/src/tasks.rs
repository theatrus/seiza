//! Work a call starts ahead of its use: frames read and prepared while
//! earlier ones are integrated, bands read while the last one is, and the
//! drizzle a frame behind.
//!
//! That work runs on helper threads of the call's own, so its serial parts
//! overlap the caller's parallel work. Where its Rayon work goes depends on
//! the thread that made the call:
//!
//! - From a thread in no Rayon pool, a helper does the whole piece of work,
//!   and its Rayon work goes to the global pool, which is the caller's.
//! - From a pool thread, as under `pool.install`, a helper only reads the
//!   files the work will open, so the work finds them in the page cache,
//!   then hands the work to the caller's pool and waits for it. The pool
//!   bounds all of it, and the helper itself does nothing but I/O.
//!
//! In a pool, the calling thread never just blocks on a helper. The work it
//! waits for may be queued in its own pool, where with one thread nobody
//! else would run it, so it helps with the pool's work while it waits.
//!
//! Work handed to the pool must not hold a lock while it runs Rayon work if
//! other work may take the same lock. A pool thread waiting on Rayon work
//! runs other queued work meanwhile, and that can be the other work, on the
//! thread that holds the lock.

use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;
use std::sync::mpsc::{
    Receiver, RecvError, RecvTimeoutError, SendError, SyncSender, TryRecvError, TrySendError,
    sync_channel,
};
use std::thread::{Scope, ScopedJoinHandle};
use std::time::Duration;

/// How long a pool thread with nothing to help with waits before it looks
/// for pool work again.
const IDLE_WAIT: Duration = Duration::from_micros(200);

/// Where a call's helper threads send their work; see the module docs.
pub(crate) struct Ahead<'a, 'pool> {
    pool: Option<&'a rayon::Scope<'pool>>,
}

/// Run `body` with the [`Ahead`] for the calling thread.
pub(crate) fn ahead<'pool, R>(body: impl FnOnce(&Ahead<'_, 'pool>) -> R) -> R {
    if rayon::current_thread_index().is_some() {
        rayon::in_place_scope(|pool| body(&Ahead { pool: Some(pool) }))
    } else {
        body(&Ahead { pool: None })
    }
}

impl<'pool> Ahead<'_, 'pool> {
    /// Start `work` on a helper thread of `scope`. In a pool, the helper
    /// first reads `files()` through.
    pub(crate) fn spawn<'scope, T>(
        &'scope self,
        scope: &'scope Scope<'scope, '_>,
        files: impl FnOnce() -> Vec<PathBuf> + Send + 'scope,
        work: impl FnOnce() -> T + Send + 'pool,
    ) -> ScopedJoinHandle<'scope, T>
    where
        T: Send + 'pool,
        'pool: 'scope,
    {
        scope.spawn(move || {
            if self.pool.is_some() {
                read_through(&files());
            }
            self.run(work)
        })
    }

    /// Run `work` from a helper thread: on that thread, or in the caller's
    /// pool while the helper waits. Never call it from the calling thread,
    /// which in a pool must not block.
    pub(crate) fn run<T>(&self, work: impl FnOnce() -> T + Send + 'pool) -> T
    where
        T: Send + 'pool,
    {
        let Some(pool) = self.pool else {
            return work();
        };
        debug_assert!(rayon::current_thread_index().is_none());
        let (sender, outcome) = sync_channel(1);
        pool.spawn(move |_| {
            let _ = sender.send(catch_unwind(AssertUnwindSafe(work)));
        });
        outcome
            .recv()
            .expect("work in the pool always reports its outcome")
            .unwrap_or_else(|panic| resume_unwind(panic))
    }

    /// The helper's result, from the calling thread. A helper's panic is
    /// raised again here.
    pub(crate) fn join<T>(&self, helper: ScopedJoinHandle<'_, T>) -> T {
        if self.pool.is_some() {
            while !helper.is_finished() {
                if !help() {
                    std::thread::sleep(IDLE_WAIT);
                }
            }
        }
        helper.join().unwrap_or_else(|panic| resume_unwind(panic))
    }

    /// [`Receiver::recv`] on the calling thread.
    pub(crate) fn recv<T>(&self, receiver: &Receiver<T>) -> Result<T, RecvError> {
        if self.pool.is_none() {
            return receiver.recv();
        }
        loop {
            match receiver.try_recv() {
                Ok(value) => return Ok(value),
                Err(TryRecvError::Disconnected) => return Err(RecvError),
                Err(TryRecvError::Empty) => {}
            }
            if !help() {
                match receiver.recv_timeout(IDLE_WAIT) {
                    Ok(value) => return Ok(value),
                    Err(RecvTimeoutError::Disconnected) => return Err(RecvError),
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        }
    }

    /// [`SyncSender::send`] on the calling thread.
    pub(crate) fn send<T>(&self, sender: &SyncSender<T>, mut value: T) -> Result<(), SendError<T>> {
        if self.pool.is_none() {
            return sender.send(value);
        }
        loop {
            match sender.try_send(value) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(rejected)) => return Err(SendError(rejected)),
                Err(TrySendError::Full(rejected)) => value = rejected,
            }
            if !help() {
                std::thread::sleep(IDLE_WAIT);
            }
        }
    }
}

/// Run one piece of the pool's pending work on this pool thread, or report
/// that there was none.
fn help() -> bool {
    matches!(rayon::yield_now(), Some(rayon::Yield::Executed))
}

/// Read `files` and drop the bytes, so the work that opens them next finds
/// them in the page cache instead of waiting on storage in the pool. Any
/// error is left for that work to meet.
fn read_through(files: &[PathBuf]) {
    let mut buffer = vec![0_u8; 1 << 20];
    for file in files {
        let Ok(mut file) = std::fs::File::open(file) else {
            continue;
        };
        while matches!(file.read(&mut buffer), Ok(read) if read > 0) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    /// Where a helper's work ran: in `pool`, its Rayon work too, and on
    /// which thread.
    fn work_in(pool: &rayon::ThreadPool) -> (bool, bool, std::thread::ThreadId) {
        let nested = (0..64)
            .into_par_iter()
            .all(|_| pool.current_thread_index().is_some());
        (
            pool.current_thread_index().is_some(),
            nested,
            std::thread::current().id(),
        )
    }

    #[test]
    fn outside_a_pool_helpers_do_the_work_themselves() {
        let caller = std::thread::current().id();
        let unused = pool(1);
        let ran = ahead(|ahead| {
            std::thread::scope(|scope| {
                let helper = ahead.spawn(scope, Vec::new, || work_in(&unused));
                ahead.join(helper)
            })
        });
        assert!(!ran.0 && !ran.1);
        assert_ne!(ran.2, caller);
    }

    #[test]
    fn inside_a_pool_the_work_and_its_rayon_work_stay_there() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("frame.bin");
        std::fs::write(&file, vec![7_u8; 3 << 20]).unwrap();
        for threads in [1, 2] {
            let pool = pool(threads);
            let ran = pool.install(|| {
                ahead(|ahead| {
                    std::thread::scope(|scope| {
                        let helpers = (0..4)
                            .map(|_| {
                                let file = file.clone();
                                ahead.spawn(scope, move || vec![file], || work_in(&pool))
                            })
                            .collect::<Vec<_>>();
                        helpers
                            .into_iter()
                            .map(|helper| ahead.join(helper))
                            .collect::<Vec<_>>()
                    })
                })
            });
            assert!(ran.iter().all(|ran| ran.0 && ran.1), "{threads} thread(s)");
        }
    }

    /// The calling thread may be the pool's only one: waiting on a channel a
    /// helper feeds from work in the pool must run that work, not block.
    #[test]
    fn a_one_thread_pool_never_waits_on_itself() {
        let pool = pool(1);
        let total = pool.install(|| {
            ahead(|ahead| {
                std::thread::scope(|scope| {
                    let (requests, incoming) = sync_channel::<u64>(1);
                    let (results, outcomes) = sync_channel::<u64>(1);
                    let worker = scope.spawn(move || {
                        for value in incoming {
                            let doubled = ahead
                                .run(move || (0..value).into_par_iter().map(|_| 2).sum::<u64>());
                            if results.send(doubled).is_err() {
                                break;
                            }
                        }
                    });
                    let mut total = 0;
                    for value in 1..=6 {
                        ahead.send(&requests, value).unwrap();
                        total += ahead.recv(&outcomes).unwrap();
                    }
                    drop(requests);
                    ahead.join(worker);
                    total
                })
            })
        });
        assert_eq!(total, 42);
    }

    #[test]
    fn a_panic_in_a_helpers_work_reaches_the_caller() {
        for threads in [None, Some(1), Some(2)] {
            let run = || {
                ahead(|ahead| {
                    std::thread::scope(|scope| {
                        let helper =
                            ahead.spawn(scope, Vec::new, || -> usize { panic!("work failed") });
                        ahead.join(helper)
                    })
                })
            };
            let caught = std::panic::catch_unwind(AssertUnwindSafe(|| match threads {
                Some(threads) => pool(threads).install(run),
                None => run(),
            }))
            .unwrap_err();
            assert_eq!(
                caught.downcast_ref::<&str>(),
                Some(&"work failed"),
                "{threads:?}"
            );
        }
    }
}
