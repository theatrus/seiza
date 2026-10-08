//! Ctrl-C and SIGTERM handling for long runs that keep scratch files.
//!
//! The first signal asks the run to stop at its next frame: the stacker
//! returns, its scratch directory is removed as it drops, and the command
//! exits with status 130. A second signal removes this process's scratch
//! directories at once and exits. A directory left by a run that could not
//! clean up (killed outright, power loss) carries its owner's process ID, so
//! a later run in the same place removes it once that process is gone.

use seiza_stacking::CancelSignal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Name prefix of the stacker's reintegration scratch directories; the owning
/// process ID follows it.
const SCRATCH_PREFIX: &str = ".seiza-reintegrate-";

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static SCRATCH_PARENTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Exit status for a run stopped by a signal, as shells report SIGINT.
pub(crate) const INTERRUPTED_EXIT_CODE: i32 = 130;

/// Install the handler once. Failing to install it only loses the cleanup.
pub(crate) fn install() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let result = ctrlc::set_handler(|| {
            if INTERRUPTED.swap(true, Ordering::SeqCst) {
                remove_own_scratch();
                std::process::exit(INTERRUPTED_EXIT_CODE);
            }
            eprintln!(
                "\ninterrupted: stopping after the current frame and removing scratch files \
                 (interrupt again to quit at once)"
            );
        });
        if let Err(error) = result {
            eprintln!("warning: cannot handle interrupts ({error}); an interrupted run may leave scratch files");
        }
    });
}

/// True once a stop has been requested.
pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// A cancel signal for the stacking library that follows the interrupt.
pub(crate) fn cancel_signal() -> CancelSignal {
    CancelSignal::new(interrupted)
}

/// Note a directory where this process may create scratch directories, so a
/// second interrupt can remove them, and remove any left there by processes
/// that no longer exist.
pub(crate) fn watch_scratch_parent(parent: &Path) {
    remove_stale_scratch(parent);
    if let Ok(mut parents) = SCRATCH_PARENTS.lock() {
        parents.push(parent.to_path_buf());
    }
}

/// The owning process ID encoded in a scratch directory's name.
fn scratch_owner(name: &str) -> Option<u32> {
    name.strip_prefix(SCRATCH_PREFIX)?
        .split('-')
        .next()?
        .parse()
        .ok()
}

fn scratch_directories(parent: &Path) -> Vec<(PathBuf, u32)> {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| {
            let owner = scratch_owner(entry.file_name().to_str()?)?;
            Some((entry.path(), owner))
        })
        .collect()
}

fn remove_own_scratch() {
    let own = std::process::id();
    let parents = match SCRATCH_PARENTS.lock() {
        Ok(parents) => parents.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    for parent in parents {
        for (path, owner) in scratch_directories(&parent) {
            if owner == own {
                let _ = std::fs::remove_dir_all(path);
            }
        }
    }
}

fn remove_stale_scratch(parent: &Path) {
    let own = std::process::id();
    for (path, owner) in scratch_directories(parent) {
        if owner != own && process_gone(owner) {
            match std::fs::remove_dir_all(&path) {
                Ok(()) => eprintln!(
                    "removed scratch files left by an interrupted run: {}",
                    path.display()
                ),
                Err(error) => eprintln!(
                    "warning: cannot remove stale scratch files {}: {error}",
                    path.display()
                ),
            }
        }
    }
}

/// Whether no process with this ID exists. Errs towards "still running" when
/// the answer is unclear, so a live run's files are never removed.
#[cfg(unix)]
fn process_gone(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // Signal 0 checks existence and permission without sending anything.
    // SAFETY: kill with signal 0 has no effect on the target process.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return false;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(windows)]
fn process_gone(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_INVALID_PARAMETER, GetLastError, STILL_ACTIVE,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    // SAFETY: the handle is checked and closed; the calls only query state.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return GetLastError() == ERROR_INVALID_PARAMETER;
        }
        let mut code = 0u32;
        let exited = GetExitCodeProcess(handle, &mut code) != 0 && code != STILL_ACTIVE as u32;
        CloseHandle(handle);
        exited
    }
}

#[cfg(not(any(unix, windows)))]
fn process_gone(_pid: u32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_names_carry_their_owner() {
        assert_eq!(scratch_owner(".seiza-reintegrate-4242-Ab3xYz"), Some(4242));
        assert_eq!(scratch_owner(".seiza-reintegrate-Ab3xYz"), None);
        assert_eq!(scratch_owner("seiza-reintegrate-4242-Ab3xYz"), None);
    }

    #[test]
    fn stale_scratch_is_removed_and_live_scratch_kept() {
        let parent = tempfile::tempdir().unwrap();
        let own = parent
            .path()
            .join(format!("{SCRATCH_PREFIX}{}-own", std::process::id()));
        // A process ID far above any system's limit cannot be running.
        let gone = parent
            .path()
            .join(format!("{SCRATCH_PREFIX}2000000000-gone"));
        let unrelated = parent.path().join("keep-me");
        for directory in [&own, &gone, &unrelated] {
            std::fs::create_dir(directory).unwrap();
            std::fs::write(directory.join("frame-0.f32"), b"x").unwrap();
        }
        remove_stale_scratch(parent.path());
        assert!(own.exists(), "this process's scratch must survive");
        assert!(!gone.exists(), "a dead process's scratch is removed");
        assert!(unrelated.exists());
    }
}
