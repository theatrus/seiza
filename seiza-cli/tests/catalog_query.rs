use seiza::objects::{ObjectCatalog, ObjectKind, ObjectMetadata, SkyObject};
use std::process::Command;

fn object(name: &str, ra: f64, dec: f64) -> SkyObject {
    SkyObject {
        kind: ObjectKind::Galaxy,
        ra,
        dec,
        mag: Some(8.0),
        major_arcmin: Some(30.0),
        minor_arcmin: Some(15.0),
        position_angle_deg: Some(35.0),
        name: name.to_string(),
        common_name: format!("{name} common"),
        metadata: ObjectMetadata {
            id: format!("test:{}", name.to_lowercase()),
            source: "test-catalog".to_string(),
            aliases: vec![format!("{name} alias")],
            parent_ids: Vec::new(),
            alternate_ids: vec![format!("other:{}", name.to_lowercase())],
            alternate_sources: vec!["other-catalog".to_string()],
        },
    }
}

#[test]
fn catalog_objects_supports_cone_and_polygon_json_queries() {
    let dir = std::env::temp_dir().join(format!("seiza-catalog-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog_path = dir.join("objects.bin");
    ObjectCatalog::new(vec![
        object("Near", 0.0, 0.0),
        object("Wrapped", 359.5, 0.0),
        object("Far", 20.0, 20.0),
    ])
    .write_to(&catalog_path)
    .unwrap();

    let cone = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "catalog",
            "objects",
            "--data",
            catalog_path.to_str().unwrap(),
            "--ra",
            "0",
            "--dec",
            "0",
            "--radius",
            "1",
            "--format",
            "json",
            "--limit",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        cone.status.success(),
        "{}",
        String::from_utf8_lossy(&cone.stderr)
    );
    let cone: serde_json::Value = serde_json::from_slice(&cone.stdout).unwrap();
    assert_eq!(cone["returned"], 2);
    assert_eq!(cone["objects"][0]["center_inside"], true);
    assert_eq!(cone["objects"][0]["source"], "test-catalog");
    assert_eq!(cone["objects"][0]["alternate_ids"][0], "other:near");

    let polygon = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "catalog",
            "objects",
            "--data",
            catalog_path.to_str().unwrap(),
            "--corner",
            "359,-1",
            "--corner",
            "1,-1",
            "--corner",
            "1,1",
            "--corner",
            "359,1",
            "--format",
            "json",
            "--limit",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        polygon.status.success(),
        "{}",
        String::from_utf8_lossy(&polygon.stderr)
    );
    let polygon: serde_json::Value = serde_json::from_slice(&polygon.stdout).unwrap();
    assert_eq!(polygon["returned"], 2);
    assert_eq!(polygon["region"]["type"], "polygon");

    let csv = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "catalog",
            "objects",
            "--data",
            catalog_path.to_str().unwrap(),
            "--ra",
            "0",
            "--dec",
            "0",
            "--radius",
            "1",
            "--format",
            "csv",
        ])
        .output()
        .unwrap();
    assert!(
        csv.status.success(),
        "{}",
        String::from_utf8_lossy(&csv.stderr)
    );
    let csv = String::from_utf8(csv.stdout).unwrap();
    assert!(csv.starts_with("kind,name,common_name,ra_deg,dec_deg"));
    assert!(csv.contains("galaxy,Near,Near common"));
    assert!(csv.contains("test:near,test-catalog,Near alias"));

    std::fs::remove_dir_all(&dir).ok();
}
