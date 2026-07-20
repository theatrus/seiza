use seiza_satellites::{CacheState, SatCheckerSource, UtcTimestamp};

#[tokio::test]
#[ignore = "network: downloads an epoch-appropriate IAU SatChecker TLE set"]
async fn historical_satchecker_tles_are_parseable_and_cached() {
    let cache = tempfile::tempdir().unwrap();
    let source = SatCheckerSource::new(cache.path()).unwrap();
    let exposure_midpoint = UtcTimestamp::parse("2025-10-18T12:52:12.790Z").unwrap();

    let first = source.load_at(exposure_midpoint).await.unwrap();
    assert_eq!(first.state, CacheState::Downloaded);
    assert!(first.catalog.len() > 1_000);
    assert_eq!(first.snapshot.query_time, exposure_midpoint);
    assert!(first.cache_path.is_file());

    let second = source.load_at(exposure_midpoint).await.unwrap();
    assert_eq!(second.state, CacheState::Cached);
    assert_eq!(second.catalog.len(), first.catalog.len());
    assert_eq!(second.cache_path, first.cache_path);
}
