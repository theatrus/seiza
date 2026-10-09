//! Work a call starts ahead of its use: frames read and prepared while
//! earlier ones are integrated, bands read while the last one is, and the
//! drizzle a frame behind.
//!
//! Where that work runs depends on the thread that made the call:
//!
//! - From a thread in no Rayon pool, a helper thread of the call's own does
//!   each piece of work, so its serial parts overlap the caller's parallel
//!   work, and its Rayon work goes to the global pool, which is the
//!   caller's.
//! - From a pool thread, as under `pool.install`, each piece is a task in
//!   the caller's pool, so the pool bounds all of it. The calling thread
//!   spawns the task into its own queue, where idle threads take the oldest
//!   work first, so it starts at once rather than after the caller's own
//!   parallel work. A helper thread reads the files the task will open, so
//!   storage is read off the pool's threads and the task mostly finds them
//!   in the page cache.
//!
//! In a pool, the calling thread never just blocks. The work it waits for
//! may be queued in its own pool, where with one thread nobody else would
//! run it, so it runs a task that has not started itself, and otherwise
//! helps with the pool's work while it waits.
//!
//! Work in a pool must not hold a lock while it runs Rayon work if other
//! work may take the same lock. A pool thread waiting on Rayon work runs
//! other queued work meanwhile, and that can be the other work, on the
//! thread that holds the lock. Nor may it wait for other such work.

use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SendError, SyncSender, TryRecvError, TrySendError, sync_channel,
};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{Scope, ScopedJoinHandle};
use std::time::Duration;

/// How long a pool thread with nothing to help with waits before it looks
/// for pool work again.
const IDLE_WAIT: Duration = Duration::from_micros(200);

/// Where a call's work ahead runs; see the module docs.
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

/// Work [`Ahead::spawn`] started, which [`Ahead::join`] collects.
pub(crate) enum Started<'scope, 'pool, T> {
    Helper(ScopedJoinHandle<'scope, T>),
    Task(Task<'pool, T>),
}

impl<'pool> Ahead<'_, 'pool> {
    /// Start `work`: on a helper thread of `scope`, or as a task in the
    /// caller's pool while a helper reads `files()` through.
    pub(crate) fn spawn<'scope, T>(
        &self,
        scope: &'scope Scope<'scope, '_>,
        files: impl FnOnce() -> Vec<PathBuf>,
        work: impl FnOnce() -> T + Send + 'pool,
    ) -> Started<'scope, 'pool, T>
    where
        T: Send + 'pool,
        'pool: 'scope,
    {
        match self.pool {
            None => Started::Helper(scope.spawn(work)),
            Some(pool) => {
                let files = files();
                if !files.is_empty() {
                    scope.spawn(move || read_through(&files));
                }
                Started::Task(task(pool, work))
            }
        }
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

    /// Collect started work on the calling thread. A panic in it is raised
    /// again here.
    pub(crate) fn join<T>(&self, started: Started<'_, '_, T>) -> T {
        match started {
            Started::Task(task) => task.join(),
            Started::Helper(helper) => self.join_helper(helper),
        }
    }

    /// Collect a helper thread on the calling thread, helping the pool while
    /// it finishes.
    pub(crate) fn join_helper<T>(&self, helper: ScopedJoinHandle<'_, T>) -> T {
        if self.pool.is_some() {
            while !helper.is_finished() {
                if !help() {
                    std::thread::sleep(IDLE_WAIT);
                }
            }
        }
        helper.join().unwrap_or_else(|panic| resume_unwind(panic))
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

type Work<'pool, T> = Box<dyn FnOnce() -> T + Send + 'pool>;
type Slot<'pool, T> = Arc<Mutex<Option<Work<'pool, T>>>>;

/// Work spawned into the caller's pool by [`Ahead::spawn`].
///
/// Dropping a task that has not started means it never does: the caller
/// has stopped wanting it, after an error or a cancel. Dropping one that
/// has waits for it to finish, and raises its panic.
pub(crate) struct Task<'pool, T> {
    work: Slot<'pool, T>,
    /// Taken by [`Task::join`].
    outcome: Option<Receiver<std::thread::Result<T>>>,
}

fn task<'pool, T: Send + 'pool>(
    pool: &rayon::Scope<'pool>,
    work: impl FnOnce() -> T + Send + 'pool,
) -> Task<'pool, T> {
    let slot: Slot<'pool, T> = Arc::new(Mutex::new(Some(Box::new(work))));
    let (sender, outcome) = sync_channel(1);
    let queued = Arc::clone(&slot);
    pool.spawn(move |_| {
        let Some(work) = take(&queued) else {
            // Run by the thread that joined it, or dropped unstarted.
            return;
        };
        let result = catch_unwind(AssertUnwindSafe(work));
        // Nobody will join a dropped task, but its panic must not vanish:
        // raised here, the scope raises it again when it ends.
        if let Err(SendError(Err(panic))) = sender.send(result) {
            resume_unwind(panic);
        }
    });
    Task {
        work: slot,
        outcome: Some(outcome),
    }
}

fn take<'pool, T>(slot: &Slot<'pool, T>) -> Option<Work<'pool, T>> {
    slot.lock().unwrap_or_else(PoisonError::into_inner).take()
}

impl<T> Task<'_, T> {
    /// The task's result, running it on this thread if it has not started.
    fn join(mut self) -> T {
        let outcome = self.outcome.take().expect("a task is joined once");
        match take(&self.work) {
            Some(work) => work(),
            None => wait(&outcome).unwrap_or_else(|panic| resume_unwind(panic)),
        }
    }
}

impl<T> Drop for Task<'_, T> {
    fn drop(&mut self) {
        let unstarted = take(&self.work);
        if let Some(outcome) = self.outcome.take()
            && unstarted.is_none()
            && !std::thread::panicking()
            && let Err(panic) = wait(&outcome)
        {
            resume_unwind(panic);
        }
    }
}

/// The outcome of a task another thread has started, helping with the
/// pool's other work while it runs.
fn wait<T>(outcome: &Receiver<std::thread::Result<T>>) -> std::thread::Result<T> {
    let disconnected = "a started task always reports its outcome";
    loop {
        match outcome.try_recv() {
            Ok(result) => return result,
            Err(TryRecvError::Disconnected) => panic!("{disconnected}"),
            Err(TryRecvError::Empty) => {}
        }
        if !help() {
            match outcome.recv_timeout(IDLE_WAIT) {
                Ok(result) => return result,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => panic!("{disconnected}"),
            }
        }
    }
}

/// Run one piece of the pool's pending work on this pool thread, or report
/// that there was none.
fn help() -> bool {
    matches!(rayon::yield_now(), Some(rayon::Yield::Executed))
}

/// Read `files` and drop the bytes, so the task that opens them finds them
/// in the page cache instead of waiting on storage in the pool. Any error is
/// left for that task to meet.
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

    /// Where work ran: in `pool`, its Rayon work too, and on which thread.
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
                let started = ahead.spawn(scope, Vec::new, || work_in(&unused));
                ahead.join(started)
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
                        let started = (0..4)
                            .map(|_| {
                                let file = file.clone();
                                ahead.spawn(scope, move || vec![file], || work_in(&pool))
                            })
                            .collect::<Vec<_>>();
                        // Joined in reverse, so some are claimed unstarted.
                        started
                            .into_iter()
                            .rev()
                            .map(|started| ahead.join(started))
                            .collect::<Vec<_>>()
                    })
                })
            });
            assert!(ran.iter().all(|ran| ran.0 && ran.1), "{threads} thread(s)");
        }
    }

    /// A helper feeding work into a one-thread pool must not stall a caller
    /// that is the pool's only thread.
    #[test]
    fn a_one_thread_pool_never_waits_on_itself() {
        let pool = pool(1);
        let total = pool.install(|| {
            ahead(|ahead| {
                std::thread::scope(|scope| {
                    let (requests, incoming) = sync_channel::<u64>(1);
                    let worker = scope.spawn(move || {
                        incoming
                            .into_iter()
                            .map(|value| {
                                ahead
                                    .run(move || (0..value).into_par_iter().map(|_| 2).sum::<u64>())
                            })
                            .sum::<u64>()
                    });
                    for value in 1..=6 {
                        ahead.send(&requests, value).unwrap();
                    }
                    drop(requests);
                    ahead.join_helper(worker)
                })
            })
        });
        assert_eq!(total, 42);
    }

    #[test]
    fn a_dropped_task_that_has_not_started_never_runs() {
        let pool = pool(1);
        let runs = std::sync::atomic::AtomicUsize::new(0);
        pool.install(|| {
            ahead(|ahead| {
                std::thread::scope(|scope| {
                    let started = ahead.spawn(scope, Vec::new, || {
                        runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    });
                    drop(started);
                })
            })
        });
        assert_eq!(runs.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn a_panic_in_started_work_reaches_the_caller() {
        for threads in [None, Some(1), Some(2)] {
            let run = || {
                ahead(|ahead| {
                    std::thread::scope(|scope| {
                        let started =
                            ahead.spawn(scope, Vec::new, || -> usize { panic!("work failed") });
                        ahead.join(started)
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
