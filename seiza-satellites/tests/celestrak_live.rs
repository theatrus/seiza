use seiza_satellites::{CacheState, CelesTrakSource};

#[tokio::test]
#[ignore = "network: downloads the current CelesTrak active-satellite OMM set"]
async fn current_celestrak_omm_is_parseable_and_cached() {
    let cache = tempfile::tempdir().unwrap();
    let source = CelesTrakSource::new(cache.path()).unwrap();

    let first = source.load_active().await.unwrap();
    assert_eq!(first.state, CacheState::Downloaded);
    assert!(first.catalog.len() > 1_000);
    assert!(first.cache_path.is_file());

    let second = source.load_active().await.unwrap();
    assert_eq!(second.state, CacheState::Fresh);
    assert_eq!(second.catalog.len(), first.catalog.len());
    assert_eq!(second.cache_path, first.cache_path);
}
