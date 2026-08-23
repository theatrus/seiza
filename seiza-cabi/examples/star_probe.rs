//! Diagnostic probe for the measurement star detector: runs the path-based
//! JSON entry point with pipeline debug output enabled and prints a summary.
//!
//! Usage: star_probe <image path> [options JSON]

use std::ffi::{CStr, CString};
use std::ptr;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: star_probe <path> [options json]");
    let options = args.next().unwrap_or_else(|| {
        r#"{"psfType":"moffat4","detectionBinning":2,"sensitivity":30}"#.to_string()
    });

    seiza_stars::debug::init_debug(true);

    let c_path = CString::new(path).unwrap();
    let c_options = CString::new(options).unwrap();
    let mut error: *mut std::os::raw::c_char = ptr::null_mut();
    let start = std::time::Instant::now();
    let out = unsafe {
        seiza_cabi::seiza_stars_detect_path_json(c_path.as_ptr(), c_options.as_ptr(), &mut error)
    };
    let elapsed = start.elapsed();
    if out.is_null() {
        let message = if error.is_null() {
            "unknown error".to_string()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        eprintln!("ERROR: {message}");
        std::process::exit(1);
    }
    let json = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    unsafe { seiza_cabi::seiza_string_free(out) };

    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let stars = value["stars"].as_array().map_or(0, Vec::len);
    println!(
        "stars={} avgHfr={} noise={} bg={} elapsed={:.2}s",
        stars,
        value["averageHfr"],
        value["noiseSigma"],
        value["backgroundMean"],
        elapsed.as_secs_f64()
    );
    if let Some(cells) = value["cells"].as_array() {
        for cell in cells {
            println!(
                "cell r{} c{} stars={} medianHfr={}",
                cell["row"], cell["col"], cell["starCount"], cell["medianHfr"]
            );
        }
    }
}
