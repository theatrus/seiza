use seiza::objects::{ObjectCatalog, ObjectKind, ObjectMetadata, SkyObject};
use seiza::star_ids::{
    StarIdentifier, StarIdentifierCatalogBuilder, StarNameCatalog, StarNameKind,
};
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

#[test]
fn catalog_star_resolves_tyc_and_hip_identifiers() {
    let dir = std::env::temp_dir().join(format!("seiza-star-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog_path = dir.join("stars.ids.bin");
    let mut builder = StarIdentifierCatalogBuilder::new(2025.5, "test Tycho-2 source");
    builder
        .add(
            StarIdentifier::Tycho2 {
                region: 5949,
                number: 2777,
                component: 1,
            },
            101.28854,
            -16.71314,
            -1.088,
        )
        .unwrap();
    builder
        .add_name(
            StarNameCatalog::GeneralCatalogOfVariableStars,
            StarNameKind::VariableStar,
            "RR Lyr",
            "gcvs:RRLYR",
            "RRAB",
            291.3663,
            42.7844,
            Some(7.06),
        )
        .unwrap();
    builder
        .add(
            StarIdentifier::Hipparcos(32349),
            101.28854,
            -16.71314,
            -1.088,
        )
        .unwrap();
    builder.write_to(&catalog_path).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "catalog",
            "star",
            "--data",
            catalog_path.to_str().unwrap(),
            "HIP 32349",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["query"], "HIP 32349");
    assert_eq!(json["returned"], 1);
    assert_eq!(json["matches"][0]["catalog"], "Hipparcos");
    assert_eq!(json["matches"][0]["designation"], "HIP 32349");
    assert_eq!(json["matches"][0]["stable_id"], "hip:32349");

    let names = Command::new(env!("CARGO_BIN_EXE_seiza"))
        .args([
            "catalog",
            "star",
            "--data",
            catalog_path.to_str().unwrap(),
            "rr-l",
            "--prefix",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        names.status.success(),
        "{}",
        String::from_utf8_lossy(&names.stderr)
    );
    let names: serde_json::Value = serde_json::from_slice(&names.stdout).unwrap();
    assert_eq!(names["mode"], "prefix");
    assert_eq!(names["returned"], 1);
    assert_eq!(names["matches"][0]["designation"], "RR Lyr");
    assert_eq!(names["matches"][0]["detail"], "RRAB");

    std::fs::remove_dir_all(&dir).ok();
}
