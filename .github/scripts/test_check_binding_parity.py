#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_binding_parity.py")
SPEC = importlib.util.spec_from_file_location("binding_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


C_SOURCE = """
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VideoRequest {
    /// Where the camera flies.
    focus: Option<[f64; 2]>,
    rotate_degrees: Option<f64>,
    framing: Option<String>,
    #[serde(skip)]
    internal: usize,
}

#[derive(Serialize)]
struct SummaryResponse {
    schema_version: u32,
    frames: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn seiza_parallax_prepare_json() {}
#[unsafe(no_mangle)]
pub extern "C" fn seiza_string_free() {}

#[cfg(test)]
mod tests {
    #[unsafe(no_mangle)]
    pub extern "C" fn seiza_only_in_tests() {}
}
"""

STUB = """
class ParallaxVideo:
    def __init__(self, starless, *, focus=None, rotate_deg=0.0, framing="fill") -> None: ...
    @property
    def summary(self) -> dict: ...
"""

PY_SOURCE = """
fn options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ParallaxOptions> {
    for (key, value) in kwargs.iter() {
        match key.as_str() {
            "focus" => {}
            "rotate_deg" => {}
            "framing" => {}
            "tour" => {
                for stop in value {
                    match stop.as_str() {
                        "hold" => {}
                        other => {}
                    }
                }
            }
            other => {}
        }
    }
}

fn summary(dict: &PyDict) {
    dict.set_item("frames", 1)?;
}
"""

CORE = """
/// What a video is filmed with.
pub struct VideoOptions {
    pub focus: Option<(f64, f64)>,
    pub rotate_deg: f64,
    pub framing: Framing,
    pub watermark_size: f64,
    hidden: usize,
}
"""

MANIFEST = """
[conventions]
suffixes = { "_degrees" = "_deg" }

[rust.VideoRequest]
core = "seiza-parallax::VideoOptions"
watermark_size = { none = "inside the watermark object" }

[functions]
seiza_parallax_prepare_json = "ParallaxVideo"
seiza_string_free = { none = "C memory management" }

[fields]
"*.schema_version" = { none = "versions the JSON" }

[keywords."ParallaxVideo.__init__"]
source = "seiza-py/src/parallax.rs"
function = "options"
ignore = ["starless", "tour"]
"""


class BindingParityTests(unittest.TestCase):
    def tree(self, *, c=C_SOURCE, stub=STUB, python=PY_SOURCE, manifest=MANIFEST, core=CORE):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        files = {
            "seiza-cabi/src/lib.rs": c,
            "seiza-cabi/src/parallax.rs": "",
            "seiza-py/seiza.pyi": textwrap.dedent(stub),
            "seiza-py/src/parallax.rs": python,
            ".github/binding-parity.toml": manifest,
            "seiza-parallax/src/pipeline.rs": core,
        }
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        return root

    def problems(self, **files):
        return MODULE.run(self.tree(**files), None).problems

    def test_matching_bindings_pass(self):
        self.assertEqual(self.problems(), [])

    def test_tests_and_skipped_fields_are_not_surface(self):
        root = self.tree()
        sources = [(root / path).read_text() for path in MODULE.C_SOURCES]
        self.assertEqual(MODULE.c_exports(sources), {"seiza_parallax_prepare_json", "seiza_string_free"})
        self.assertNotIn("internal", MODULE.json_types(sources)["VideoRequest"])

    def test_an_unmapped_export_fails(self):
        c = C_SOURCE.replace(
            "pub extern \"C\" fn seiza_string_free() {}",
            "pub extern \"C\" fn seiza_string_free() {}\n#[unsafe(no_mangle)]\npub extern \"C\" fn seiza_parallax_new_call() {}",
        )
        [problem] = self.problems(c=c)
        self.assertIn("seiza_parallax_new_call is exported but not in [functions]", problem)

    def test_a_mapping_to_a_name_python_lacks_fails(self):
        manifest = MANIFEST.replace('"ParallaxVideo"', '"ParallaxVideo.prepare"')
        [problem] = self.problems(manifest=manifest)
        self.assertIn("ParallaxVideo.prepare, which seiza.pyi does not declare", problem)

    def test_a_new_c_field_without_a_python_name_fails(self):
        c = C_SOURCE.replace("    framing: Option<String>,", "    framing: Option<String>,\n    crop: Option<[usize; 4]>,")
        problems = self.problems(c=c)
        self.assertEqual(len(problems), 1)
        self.assertIn("VideoRequest.crop has no Python name", problems[0])

    def test_conventions_rename_suffixes(self):
        self.assertEqual(MODULE.python_name("distance_parsecs", {"_parsecs": "_pc"}), "distance_pc")

    def test_an_exemption_python_has_outgrown_fails(self):
        manifest = MANIFEST.replace('[fields]\n', '[fields]\n"VideoRequest.framing" = { gap = "not in Python yet" }\n', 1)
        [problem] = self.problems(manifest=manifest)
        self.assertIn("Python has it now: remove the entry", problem)

    def test_gaps_pass_and_are_counted(self):
        c = C_SOURCE.replace("    framing: Option<String>,", "    framing: Option<String>,\n    crop: Option<[usize; 4]>,")
        manifest = MANIFEST.replace('[fields]\n', '[fields]\n"VideoRequest.crop" = { gap = "crop is C-only for now" }\n', 1)
        report = MODULE.run(self.tree(c=c, manifest=manifest), None)
        self.assertEqual(report.problems, [])
        self.assertEqual(report.gaps, ["VideoRequest.crop: crop is C-only for now"])

    def test_keywords_must_match_the_stub_both_ways(self):
        python = PY_SOURCE.replace('            "framing" => {}\n', "")
        [problem] = self.problems(python=python)
        self.assertIn("declares framing in seiza.pyi, but options() does not accept it", problem)
        stub = STUB.replace(', framing="fill"', "")
        problems = self.problems(stub=stub)
        self.assertTrue(any("options() accepts framing, but seiza.pyi does not declare it" in p for p in problems))

    def test_nested_match_arms_are_not_keywords(self):
        body = MODULE.rust_function(PY_SOURCE, "options")
        self.assertEqual(MODULE.outer_arms(body), {"focus", "rotate_deg", "framing", "tour"})

    def test_a_new_core_field_must_reach_the_c_abi(self):
        core = CORE.replace("    pub framing: Framing,", "    pub framing: Framing,\n    pub crop: Option<(usize, usize, usize, usize)>,")
        [problem] = self.problems(core=core)
        self.assertIn("seiza-parallax::VideoOptions.crop does not reach the C ABI's VideoRequest", problem)

    def test_core_entries_must_stay_true(self):
        # A C name the C type lacks.
        manifest = MANIFEST.replace('watermark_size = { none = "inside the watermark object" }', 'watermark_size = { c = "watermark" }')
        [problem] = self.problems(manifest=manifest)
        self.assertIn("VideoRequest has no watermark", problem)
        # An exemption the C type has outgrown.
        c = C_SOURCE.replace("    framing: Option<String>,", "    framing: Option<String>,\n    watermark_size: Option<f64>,")
        problems = self.problems(c=c)
        self.assertTrue(any("the C ABI lacks it, but VideoRequest has it now" in p for p in problems), problems)
        # A field the core type no longer has, and a core type that is gone.
        [problem] = self.problems(core=CORE.replace("    pub watermark_size: f64,\n", ""))
        self.assertIn("lists watermark_size, which seiza-parallax::VideoOptions no longer has", problem)
        [problem] = self.problems(core="")
        self.assertIn("0 definitions of pub struct VideoOptions", problem)

    def test_one_binding_alone_needs_a_reason(self):
        report = MODULE.Report()
        MODULE.check_both_sides(report, ["seiza-cabi/src/parallax.rs", "seiza-parallax/src/dust.rs"], "Fix dust\n")
        self.assertEqual(len(report.problems), 1)
        self.assertIn("touches the C ABI's source but not Python's", report.problems[0])
        report = MODULE.Report()
        MODULE.check_both_sides(
            report,
            ["seiza-cabi/src/lib.rs"],
            "Add a catalog setup call\n\nBinding-parity: the apps' catalog setup; Python has fetch_catalogs\n",
        )
        self.assertEqual(report.problems, [])
        for paths in (["seiza-cabi/src/lib.rs", "seiza-py/seiza.pyi"], ["seiza-parallax/src/dust.rs"], ["seiza-cabi/include/seiza_cabi.h"]):
            report = MODULE.Report()
            MODULE.check_both_sides(report, paths, "")
            self.assertEqual(report.problems, [], paths)


if __name__ == "__main__":
    unittest.main()
