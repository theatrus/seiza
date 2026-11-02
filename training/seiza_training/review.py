"""Pass 6: one review page per built set, and the decisions reviewers make.

A page shows an input sub-frame and the set's stacks side by side, every
stack at one stretch taken from the deep stack, with the measurements and
checks beside them. A reviewer accepts or rejects the set and its starless
background with `decide`; the decision goes into the manifest with the
reviewer and the date. Nothing that needs review is trained on without one.
"""

from __future__ import annotations

import html
import warnings
from pathlib import Path
from typing import Any

import numpy as np


from .manifest import Manifest, loads
from .store import Store, read_image, robust_sigma, write_png

THUMBNAIL_WIDTH = 1024


def mono(image: np.ndarray) -> np.ndarray:
    """A (h, w) view of a plane stored as (h, w) or (c, h, w)."""
    return image if image.ndim == 2 else image.mean(axis=0)


def shrink(image: np.ndarray, width: int = THUMBNAIL_WIDTH) -> np.ndarray:
    """Block-mean `image` down to at most `width` columns."""
    factor = max(1, -(-image.shape[1] // width))
    height = image.shape[0] // factor * factor
    columns = image.shape[1] // factor * factor
    blocks = np.asarray(image[:height, :columns], dtype=np.float32)
    with warnings.catch_warnings():
        # A block wholly outside the stack's coverage is NaN, and shown black.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(
            blocks.reshape(height // factor, factor, columns // factor, factor), axis=(1, 3)
        )


class Stretch:
    """An asinh display stretch fixed by one image, for every image of a
    page: black just below the sky, white at the 99.95th percentile."""

    def __init__(self, image: np.ndarray, strength: float = 40.0) -> None:
        finite = image[np.isfinite(image)]
        sigma = robust_sigma(image) or float(np.std(finite))
        self.black = float(np.median(finite)) - 2.0 * sigma
        self.white = float(np.percentile(finite, 99.95))
        self.strength = strength

    def __call__(self, image: np.ndarray) -> np.ndarray:
        scaled = (image - self.black) / max(self.white - self.black, 1e-12)
        scaled = np.nan_to_num(np.clip(scaled, 0.0, None), nan=0.0)
        shown = np.arcsinh(self.strength * scaled) / np.arcsinh(self.strength)
        return (np.clip(shown, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def thumbnail(path: Path, image: np.ndarray, stretch: Stretch) -> str:
    write_png(path, stretch(image))
    return path.name


def table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f"<table>{body}</table>"


def figure(src: str, caption: str) -> str:
    return (
        f'<figure><a href="{src}"><img src="{src}" loading="lazy"></a>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


STYLE = """
body { font: 14px/1.4 -apple-system, system-ui, sans-serif; margin: 16px; color: #222; background: #fafafa; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
figure { margin: 0; background: #fff; border: 1px solid #ddd; padding: 6px; }
img { width: 100%; image-rendering: auto; background: #000; }
figcaption { font-size: 12px; color: #555; margin-top: 4px; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #ddd; padding: 3px 8px; text-align: left; font-weight: normal; }
th { color: #666; } code { background: #eee; padding: 1px 4px; }
.accept { color: #070; } .reject { color: #a00; }
"""


def decision(manifest: Manifest, subject: str, kind: str) -> str:
    row = manifest.review(subject, kind)
    if row is None:
        return "unreviewed"
    return f"{row['decision']} by {row['reviewer']} on {row['decided_at']}"


def page(manifest: Manifest, store: Store, set_row: Any) -> Path:
    """Write one set's review page and its thumbnails."""
    set_id = set_row["id"]
    directory = store.review / set_id
    directory.mkdir(parents=True, exist_ok=True)
    products = {row["id"]: row for row in manifest.rows("SELECT * FROM products WHERE set_id = ?", (set_id,))}

    def load(product_id: str) -> np.ndarray | None:
        row = products.get(product_id)
        if row is None:
            return None
        files = loads(row["files"], {})
        return mono(np.load(store.root / files["image"], mmap_mode="r"))

    deep = load(f"{set_id}/deep")
    if deep is None:
        raise RuntimeError("the set has no deep stack")
    small_deep = shrink(deep)
    stretch = Stretch(small_deep)
    figures = []
    build = loads(set_row["build_reason"], {}) if set_row["build_status"] == "built" else {}
    reference = build.get("reference")
    if reference:
        sub = shrink(mono(read_image(reference, set_row["width"], set_row["height"])))
        figures.append(figure(thumbnail(directory / "sub-frame.png", sub, Stretch(sub)),
                              "reference sub-frame, uncalibrated, its own stretch"))
    figures.append(figure(thumbnail(directory / "deep.png", small_deep, stretch), "deep stack"))
    for half in ("best", "worst"):
        image = load(f"{set_id}/seeing/{half}")
        if image is not None:
            stats = loads(products[f"{set_id}/seeing/{half}"]["stats"], {})
            figures.append(figure(
                thumbnail(directory / f"seeing-{half}.png", shrink(image), stretch),
                f"seeing {half} quarter, stack FWHM {stats.get('stack_fwhm_px', 'n/a')}",
            ))
    for half in ("a", "b"):
        image = load(f"{set_id}/noise/n08-d0/{half}")
        if image is not None:
            figures.append(figure(
                thumbnail(directory / f"noise-n08-{half}.png", shrink(image), stretch),
                f"noise pair n08-d0, half {half.upper()}",
            ))

    measurement = loads(set_row["measurement"], {})
    calibration = loads(set_row["calibration"], {})
    pair_rows = []
    for row in sorted(products.values(), key=lambda row: row["id"]):
        if row["kind"] != "noise-a":
            continue
        check = loads(row["recipe"], {}).get("check", {})
        pair_rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["id"].removeprefix(f"{set_id}/noise/").removesuffix("/a"),
                    row["status"], row["reason"] or "",
                    f"{check.get('fwhm_a_px', 0):.2f} / {check.get('fwhm_b_px', 0):.2f}",
                    f"{check.get('background_offset', 0):.3g}",
                    f"{check.get('difference_sigma') or 0:.3g}",
                    f"{check.get('predicted_difference_sigma', 0):.3g}",
                )
            ) + "</tr>"
        )
    pairs = (
        "<table><tr><th>pair</th><th>status</th><th>reason</th><th>FWHM A / B</th>"
        "<th>offset</th><th>σ of A−B</th><th>σ predicted</th></tr>" + "".join(pair_rows) + "</table>"
        if pair_rows else "<p>No noise pairs.</p>"
    )
    store_flag = f"--store {html.escape(str(store.root))}"
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(set_row['name'])}</title><style>{STYLE}</style></head><body>
<p><a href="../index.html">All sets</a></p>
<h1>{html.escape(set_row['name'])}</h1>
{table([
    ("set", set_id), ("split", f"{set_row['split']} (group {set_row['sky_group']})"),
    ("frames", f"{measurement.get('accepted')} accepted of {set_row['frames']}, {build.get('admitted')} admitted"),
    ("median FWHM", f"{measurement.get('median_fwhm_px', 0):.2f} px"),
    ("seeing quarters", f"{measurement.get('best_quartile_fwhm_px', 0):.2f} / {measurement.get('worst_quartile_fwhm_px', 0):.2f} px"),
    ("flags", ", ".join(f"{key} {value}" for key, value in measurement.get("flagged", {}).items()) or "none"),
    ("masters", ", ".join(f"{kind}: {Path(path).name}" for kind, path in calibration.get("masters", {}).items() if path)),
    ("solve", measurement.get("solve", {}).get("status")),
    ("set decision", decision(manifest, set_id, "set")),
    ("starless decision", decision(manifest, set_id, "starless")),
])}
<p>Decide: <code>python -m seiza_training {store_flag} decide {set_id} set accept --reviewer NAME</code>
(or <code>reject</code>, with <code>--note</code>).</p>
<h2>Stacks, one stretch</h2><div class="grid">{''.join(figures)}</div>
<h2>Starless background</h2>
<p>Not built yet: catalogue-guided star removal lands with the foundations.</p>
<h2>Noise pairs</h2>{pairs}
</body></html>"""
    path = directory / "index.html"
    path.write_text(body)
    return path


def index(manifest: Manifest, store: Store) -> Path:
    rows = []
    for set_row in manifest.rows("SELECT * FROM sets ORDER BY name"):
        built = set_row["build_status"] == "built"
        link = (
            f'<a href="{set_row["id"]}/index.html">{html.escape(set_row["name"])}</a>'
            if built and (store.review / set_row["id"] / "index.html").exists()
            else html.escape(set_row["name"])
        )
        rows.append(
            "<tr>" + "".join(
                f"<td>{cell}</td>"
                for cell in (
                    link,
                    html.escape(str(set_row["split"])),
                    set_row["frames"],
                    html.escape(str(set_row["calibration_status"])),
                    html.escape(str(set_row["measurement_status"])),
                    html.escape(str(set_row["build_status"])),
                    html.escape(decision(manifest, set_row["id"], "set")),
                )
            ) + "</tr>"
        )
    path = store.review / "index.html"
    path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corpus review</title><style>{STYLE}</style></head><body><h1>Corpus review</h1>
<table><tr><th>set</th><th>split</th><th>frames</th><th>calibration</th><th>measured</th>
<th>built</th><th>decision</th></tr>{''.join(rows)}</table></body></html>"""
    )
    return path


def review(manifest: Manifest, store: Store, *, only: set[str] | None = None) -> dict[str, Any]:
    pages = 0
    for set_row in manifest.rows("SELECT * FROM sets WHERE build_status = 'built' ORDER BY name"):
        if only is not None and set_row["id"] not in only:
            continue
        page(manifest, store, set_row)
        pages += 1
    return {"pages": pages, "index": str(index(manifest, store))}
