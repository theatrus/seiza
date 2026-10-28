"""Create a browser regression fixture using an installed Seiza wheel.

python seiza-analysis/tests/browser_fixture.py /path/to/check.html
Open the resulting self-contained HTML. The first line must say PASS.
"""
from pathlib import Path
import sys
import seiza

report = {
    "schema_version": 1, "stacking_version": "fixture", "output_version": "fixture",
    "stage": "online", "width": 2, "height": 1, "channels": 1,
    "accepted_frames": 1001, "rejected_frames": 205,
    "support": [{"supported_pixels": 2, "total_pixels": 2, "support_minimum": 1,
                 "support_maximum": 1001, "support_mean": 501, "rejected_samples": 1000}],
    "support_map": {"width": 2, "height": 1, "channels": 1, "values": [1, 1001], "quantity": "accepted_observations"},
    "rejection_map": None, "final_samples": [{"finite_samples": 2, "integrated_samples": 2 if i == 0 else 1} for i in range(1001)], "warnings": [],
    "context": {"accepted_frames": 1001, "rejected_frames": 205, "reference_integrated": True,
                "reference_image": None, "options": {}, "calibrations": [], "planning_notes": [],
                "frames": [{"source": f"fixture-{i:04}.fits", "exposure_seconds": 60,
                            "calibration_index": 0, "registration_rms_pixels": None if i == 0 else 0.1 + i / 10000,
                            "weight": [], "noise": [], "mapping": None} for i in range(1001)]},
    "provenance": {"frames": [{"source": {"path": f"rejected-{i:04}.fits"},
                              "disposition": "rejected", "reason": "fixture rejection"} for i in range(205)]},
}
collection = {"schema_version": 1, "title": "Browser regression fixture",
              "stacks": [{"label": label, "report": report} for label in "SHO"]}
html = seiza.stack_collection_html(collection)
harness = Path(__file__).with_name("stack-report.browser.js").read_text()
Path(sys.argv[1]).write_text(html.replace("</body>", "<script>" + harness + "</script></body>"), encoding="utf-8")
