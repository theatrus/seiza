//! Blocking forms of the network loads, for callers with no async runtime of
//! their own, such as language bindings.

use crate::{
    CelesTrakLoad, CelesTrakSource, Error, OrbitalCatalogLoad, OrbitalCatalogSource, Result,
    UtcTimestamp,
};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// How often a blocking load looks at its cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(50);

impl CelesTrakSource {
    /// [`Self::load_active`] on a private single-thread runtime, blocking the
    /// calling thread. When `cancel` turns true the load stops at once and
    /// returns [`Error::Cancelled`]; the cache keeps no half-written snapshot.
    /// Do not call it from inside an async runtime.
    pub fn load_active_blocking(&self, cancel: Option<&AtomicBool>) -> Result<CelesTrakLoad> {
        block_on(self.load_active(), cancel)
    }
}

impl OrbitalCatalogSource {
    /// [`Self::load_at`] on a private single-thread runtime, blocking the
    /// calling thread. When `cancel` turns true the load stops at once and
    /// returns [`Error::Cancelled`]; the cache keeps no half-written snapshot.
    /// Do not call it from inside an async runtime.
    pub fn load_at_blocking(
        &self,
        time_utc: UtcTimestamp,
        cancel: Option<&AtomicBool>,
    ) -> Result<OrbitalCatalogLoad> {
        block_on(self.load_at(time_utc), cancel)
    }
}

fn block_on<T>(load: impl Future<Output = Result<T>>, cancel: Option<&AtomicBool>) -> Result<T> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(Error::Runtime)?;
    let Some(cancel) = cancel else {
        return runtime.block_on(load);
    };
    let outcome = runtime.block_on(async {
        tokio::select! {
            biased;
            () = cancelled(cancel) => None,
            result = load => Some(result),
        }
    });
    match outcome {
        Some(result) => result,
        None => {
            // The dropped load may have left a cache-lock wait running on a
            // blocking thread; let it finish without holding up the caller.
            runtime.shutdown_background();
            Err(Error::Cancelled)
        }
    }
}

async fn cancelled(flag: &AtomicBool) {
    while !flag.load(Ordering::Relaxed) {
        tokio::time::sleep(CANCEL_POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMM: &str = r#"[{
        "OBJECT_NAME":"ISS (ZARYA)", "OBJECT_ID":"1998-067A",
        "EPOCH":"2024-05-02T12:00:00.000000", "MEAN_MOTION":15.5,
        "ECCENTRICITY":0.0005, "INCLINATION":51.64, "RA_OF_ASC_NODE":160.0,
        "ARG_OF_PERICENTER":80.0, "MEAN_ANOMALY":280.0, "NORAD_CAT_ID":25544,
        "BSTAR":0.0001
    }]"#;

    #[test]
    fn a_fresh_cache_loads_without_a_runtime_of_the_callers() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("celestrak-active-manual.json"), OMM).unwrap();
        let source = CelesTrakSource::new(directory.path()).unwrap();
        let load = source.load_active_blocking(None).unwrap();
        assert_eq!(load.catalog.len(), 1);
        assert_eq!(load.state.as_str(), "fresh");
    }

    #[test]
    fn a_raised_flag_stops_the_load() {
        let directory = tempfile::tempdir().unwrap();
        // An endpoint that is never reached: the flag is up before the load starts.
        let source = CelesTrakSource::new(directory.path())
            .unwrap()
            .with_endpoint("http://127.0.0.1:9/never");
        let cancel = AtomicBool::new(true);
        let error = source.load_active_blocking(Some(&cancel)).unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error}");
    }
}
