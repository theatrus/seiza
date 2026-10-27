use psf_guard::calibration as c;
use psf_guard::commands::import::headers::FrameMeta;
use rusqlite::{Connection, params};
use serde_json::{Value, json};
fn frame(id: usize, kind: &str, at: i64, temp: f64) -> Value {
    json!({"id":id,"frame_uuid":format!("f{id}"),"rig_uuid":"r","kind":kind,"source_path":format!("/f{id}.fits"),"source_fingerprint":format!("fp{id}"),"captured_at":at,"camera":"Camera","telescope":"Scope","width":3000,"height":2000,"channels":1,"binning_x":1,"binning_y":1,"gain":100,"offset":20,"exposure_s":300.0,"camera_temp":temp,"filter":"Ha"})
}
fn names(frames: &[c::CalibrationFrame]) -> Vec<String> {
    frames.iter().map(|f| f.frame_uuid.clone()).collect()
}
fn main() {
    let at = 1_790_000_000i64;
    let day = 86400;
    let light = json!({"path":"/light.fits","rig_uuid":"r","camera":"Camera","telescope":"Scope","width":3000,"height":2000,"channels":1,"binning_x":1,"binning_y":1,"gain":100,"offset":20,"exposure_s":300.0,"camera_temp":-10.0,"filter":"Ha","timestamp":at});
    let base: Vec<_> = (0..2)
        .map(|i| frame(i, "dark", at + (i as i64 + 1) * day, -8.0))
        .chain((2..14).map(|i| frame(i, "dark", at + 7 * day + i as i64 * 60, -10.0)))
        .collect();
    let mut cases = Vec::new();
    for scenario in 0..16 {
        let conn = Connection::open_in_memory().unwrap();
        c::ensure_schema(&conn).unwrap();
        let mut frames = base.clone();
        let mut l = light.clone();
        let mut settings = Vec::new();
        let mut grouping = Value::Null;
        match scenario {
            1 => {
                for f in &mut frames {
                    f["kind"] = json!("bias");
                    f["camera_temp"] = json!(if f["id"].as_u64().unwrap() < 2 {
                        -4.1
                    } else {
                        -10.0
                    });
                }
            }
            2 => {
                c::configure_dark_choice(Some(c::DarkChoice::Nearest));
                settings.push(json!({"key":"dark_choice","value":"nearest"}));
            }
            3 => {
                frames.truncate(7);
            }
            4 => {
                for f in &mut frames {
                    f["captured_at"] = json!(at - 400 * day);
                }
            }
            5 => {
                for f in &mut frames {
                    f["captured_at"] = json!(at - 400 * day);
                    f["valid_direction"] = json!("forward");
                }
            }
            6 => {
                frames[0]["is_master"] = json!(true);
                frames[0]["gain"] = Value::Null;
                frames[0]["camera_temp"] = Value::Null;
            }
            7 => {
                for f in &mut frames {
                    f["readout_mode_name"] = json!(if f["id"].as_u64().unwrap() < 2 {
                        "High Gain Mode"
                    } else {
                        "Extend Fullwell 2CMS"
                    });
                }
                l["readout_mode_name"] = json!(" extend fullwell 2cms ");
            }
            8 => {
                for f in &mut frames {
                    f["gain"] = json!(200);
                }
            }
            9 => {
                for f in &mut frames {
                    f["kind"] = json!("flat");
                    f["rotation"] = json!(90.0);
                    f["camera_temp"] = json!(20.0);
                    f["captured_at"] = json!(at - 3600);
                    f["exposure_s"] = json!(2.0);
                }
                l["rotator_position"] = json!(90.0);
            }
            10 => {
                for f in &mut frames {
                    f["kind"] = json!("flat");
                    f["rotation"] = json!(90.0);
                    f["captured_at"] = json!(at - 61 * day);
                }
                l["rotator_position"] = json!(90.0);
            }
            11 => {
                for f in &mut frames {
                    f["valid_direction"] = json!("forward");
                }
            }
            _ => {}
        }

        if scenario >= 12 {
            grouping = serde_json::from_str(include_str!("../flat-groups/request.json")).unwrap();
            grouping["limits"] = json!([]);
            grouping["lights"] = json!([]);
            let meta: FrameMeta = serde_json::from_value(l.clone()).unwrap();
            let rig = c::rules::light_rig_uuid(&meta);
            l["rig_uuid"] = json!(rig);
            l["timestamp"] = json!(if scenario == 12 {
                2 * day
            } else if scenario == 13 {
                4 * day
            } else {
                7 * day
            });
            l["rotator_position"] = json!(90.0);
            if scenario == 15 {
                l["gain"] = json!(200);
            }
            frames.clear();
            for row in grouping["responses"].as_array_mut().unwrap() {
                row["rig_uuid"] = json!(rig);
                for (i, id) in row["frame_uuids"].as_array().unwrap().iter().enumerate() {
                    let mut f = frame(
                        frames.len(),
                        "flat",
                        row["captured_first"].as_i64().unwrap() + i as i64 * 60,
                        -10.0,
                    );
                    f["frame_uuid"] = id.clone();
                    f["rig_uuid"] = json!(rig);
                    f["rotation"] = row["rotation"].clone();
                    frames.push(f);
                }
            }
            conn.execute_batch(GROUP_SQL).unwrap();
            for row in grouping["responses"].as_array().unwrap() {
                conn.execute("INSERT INTO psf_guard_flat_response (response_key,session_key,rig_uuid,family_key,night,filter_name,rotation,frame_uuids,frames,captured_first,captured_last,level_adu,pedestal_adu,noise,width,height,bin,algorithm_version,measured_at) VALUES (?1,?2,?3,?4,'','Ha',?5,?6,3,?7,?8,30000,500,0.0003,100,80,10,1,0)",params![row["response_key"].as_str(),row["session_key"].as_str(),rig,row["family_key"].as_str(),row["rotation"].as_f64(),row["frame_uuids"].to_string(),row["captured_first"].as_i64(),row["captured_last"].as_i64()]).unwrap();
            }
            for row in grouping["comparisons"].as_array().unwrap() {
                conn.execute("INSERT INTO psf_guard_flat_comparison (earlier_key,later_key,algorithm_version,broad,dust,sigma,detail_json,measured_at) VALUES (?1,?2,?3,?4,?5,?6,?7,0)",params![row["earlier_key"].as_str(),row["later_key"].as_str(),c::flat_groups::metric::COMPARE_ALGORITHM_VERSION,row["broad"].as_f64(),row["dust"].as_f64(),row["sigma"].as_f64(),row.to_string()]).unwrap();
            }
        }
        for f in &frames {
            conn.execute("INSERT INTO psf_guard_calibration_frame (id,frame_uuid,rig_uuid,kind,source_path,source_fingerprint,captured_at,camera,telescope,width,height,channels,binning_x,binning_y,gain,offset,exposure_s,camera_temp,filter_name,is_master,rotation,valid_direction,readout_mode_name,added_at,updated_at) VALUES (?1,?2,?14,?3,?4,?5,?6,'Camera','Scope',3000,2000,1,1,1,?7,20,?8,?9,'Ha',?10,?11,?12,?13,0,0)",params![f["id"].as_i64(),f["frame_uuid"].as_str(),f["kind"].as_str(),f["source_path"].as_str(),f["source_fingerprint"].as_str(),f["captured_at"].as_i64(),f["gain"].as_i64(),f["exposure_s"].as_f64(),f["camera_temp"].as_f64(),f["is_master"].as_bool().unwrap_or(false),f["rotation"].as_f64(),f["valid_direction"].as_str(),f["readout_mode_name"].as_str(),f["rig_uuid"].as_str()]).unwrap();
        }
        let meta: FrameMeta = serde_json::from_value(l.clone()).unwrap();
        let selected = c::select_for_light(&conn, &meta).unwrap();
        let counts = c::master_input_counts(&selected);
        cases.push(json!({"name":format!("reference-{scenario}"),"request":{"frames":frames,"lights":[l.clone(),l.clone()],"global_settings":settings,"grouping":if grouping.is_null(){json!({})}else{grouping}},"expected":{"bias":names(&selected.bias),"dark":names(&selected.dark),"dark_flat":names(&selected.dark_flat),"flat":names(&selected.flat),"kept_out":names(&selected.kept_out),"master_counts":counts,"tolerances":selected.match_tolerances(),"flat_basis":selected.flat_basis,"flat_family":selected.flat_family}}));
        c::configure_dark_choice(None);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"psf_guard_revision":"79f5e31d53b8371e8fcd0b22a9b69692412cd8b4","cases":cases})
        )
        .unwrap()
    );
}

const GROUP_SQL: &str = r#"
    CREATE TABLE IF NOT EXISTS psf_guard_flat_response (
        response_key       TEXT PRIMARY KEY,
        session_key        TEXT NOT NULL,
        rig_uuid           TEXT NOT NULL,
        family_key         TEXT NOT NULL,
        night              TEXT NOT NULL,
        filter_name        TEXT,
        rotation           REAL,
        frame_uuids        TEXT NOT NULL,
        frames             INTEGER NOT NULL,
        captured_first     INTEGER,
        captured_last      INTEGER,
        level_adu          REAL,
        pedestal_adu       REAL,
        noise              REAL,
        width              INTEGER,
        height             INTEGER,
        bin                INTEGER,
        algorithm_version  INTEGER NOT NULL,
        error              TEXT,
        measured_at        INTEGER NOT NULL,
        unread             TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_psf_guard_flat_response_family
        ON psf_guard_flat_response(rig_uuid, family_key);
    CREATE TABLE IF NOT EXISTS psf_guard_flat_comparison (
        earlier_key        TEXT NOT NULL,
        later_key          TEXT NOT NULL,
        algorithm_version  INTEGER NOT NULL,
        broad              REAL NOT NULL,
        dust               REAL NOT NULL,
        sigma              REAL NOT NULL,
        detail_json        TEXT NOT NULL,
        measured_at        INTEGER NOT NULL,
        PRIMARY KEY (earlier_key, later_key, algorithm_version)
    );
    CREATE TABLE IF NOT EXISTS psf_guard_flat_light_evidence (
        session_key        TEXT NOT NULL,
        event_key          TEXT NOT NULL,
        algorithm_version  INTEGER NOT NULL,
        verdict            TEXT NOT NULL,
        detail_json        TEXT NOT NULL,
        measured_at        INTEGER NOT NULL,
        change_key         TEXT,
        PRIMARY KEY (session_key, event_key, algorithm_version)
    );
"#;
