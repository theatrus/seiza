"""The manifest: one SQLite file recording every frame, set, product and
review decision the corpus passes make.

Each pass reads what earlier passes wrote and adds its own columns or rows,
so a pass can be rerun on its own. Nothing here is a training example; the
manifest says which files hold them and how they were made.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per image file the crawl read: lights, masters made elsewhere,
-- and calibration masters from the library.
CREATE TABLE IF NOT EXISTS frames (
    path TEXT PRIMARY KEY,
    size INTEGER,
    mtime REAL,
    telescope TEXT,
    header_telescope TEXT,
    origin TEXT NOT NULL,
    target TEXT,
    night TEXT,
    role TEXT,
    is_master INTEGER,
    raw_image_type TEXT,
    camera TEXT,
    filter TEXT,
    gain INTEGER,
    "offset" INTEGER,
    readout_mode INTEGER,
    binning_x INTEGER,
    binning_y INTEGER,
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    bayer_pattern TEXT,
    exposure_seconds REAL,
    camera_temp_c REAL,
    captured_at_unix INTEGER,
    focal_length_mm REAL,
    pixel_size_um REAL,
    rotation_deg REAL,
    ra_deg REAL,
    dec_deg REAL,
    object TEXT,
    airmass REAL,
    pier_side TEXT,
    saturation REAL,
    software TEXT,
    calibration_state TEXT,
    flags TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    set_id TEXT
);
CREATE INDEX IF NOT EXISTS frames_set ON frames(set_id);
CREATE INDEX IF NOT EXISTS frames_origin ON frames(origin, role);

-- One row per set: a target, filter, camera, gain, offset, exposure,
-- binning and night on one telescope.
CREATE TABLE IF NOT EXISTS sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    telescope TEXT,
    target TEXT,
    filter TEXT,
    camera TEXT,
    gain INTEGER,
    "offset" INTEGER,
    exposure_seconds REAL,
    binning INTEGER,
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    bayer_pattern TEXT,
    night TEXT,
    frames INTEGER,
    ra_deg REAL,
    dec_deg REAL,
    pixel_scale_arcsec REAL,
    field_deg REAL,
    flags TEXT NOT NULL DEFAULT '[]',
    calibration TEXT,
    calibration_status TEXT,
    measurement TEXT,
    measurement_status TEXT,
    sky_group TEXT,
    split TEXT,
    recipes TEXT,
    build_status TEXT,
    build_reason TEXT
);

-- One row per measured frame.
CREATE TABLE IF NOT EXISTS measurements (
    path TEXT PRIMARY KEY,
    set_id TEXT NOT NULL,
    status TEXT NOT NULL,
    fwhm_px REAL,
    hfr_px REAL,
    eccentricity REAL,
    stars INTEGER,
    background REAL,
    noise REAL,
    star_snr REAL,
    flags TEXT NOT NULL DEFAULT '[]',
    detail TEXT
);
CREATE INDEX IF NOT EXISTS measurements_set ON measurements(set_id);

-- One row per finished stack made elsewhere from the corpus, such as a PSF
-- Guard group: a target, filter and exposure over all its nights. `sets`
-- lists the corpus sets its frames came from.
CREATE TABLE IF NOT EXISTS stacks (
    id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    path TEXT NOT NULL,
    revision TEXT,
    telescope TEXT,
    target TEXT,
    filter TEXT,
    exposure_seconds REAL,
    frames INTEGER,
    total_exposure_seconds REAL,
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    method TEXT,
    stacking_version TEXT,
    sets TEXT NOT NULL DEFAULT '[]',
    sky_group TEXT,
    split TEXT,
    created TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT
);

-- One row per product: a stack or a pair half. A finished stack's products
-- name its `stacks` id as their set.
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    set_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    recipe TEXT NOT NULL,
    frames TEXT NOT NULL,
    files TEXT NOT NULL,
    stats TEXT NOT NULL,
    versions TEXT NOT NULL,
    created TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS products_set ON products(set_id, kind);

-- A reviewer's decision on a set or its starless background. Nothing that
-- needs review is trained on without one.
CREATE TABLE IF NOT EXISTS reviews (
    subject TEXT NOT NULL,
    kind TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (subject, kind)
);

-- Each run of a pass, with the versions it ran under.
CREATE TABLE IF NOT EXISTS passes (
    name TEXT NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    versions TEXT NOT NULL,
    arguments TEXT NOT NULL,
    summary TEXT
);
"""


def now() -> str:
    """The current UTC time, to the second, as ISO 8601."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dumps(value: Any) -> str:
    """JSON with sorted keys, so equal records compare equal as text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def loads(text: str | None, default: Any = None) -> Any:
    """Decode a JSON column, or `default` when it is empty."""
    if text is None or text == "":
        return default
    return json.loads(text)


class Manifest:
    """The manifest file of one corpus store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60.0)
        self.db.row_factory = sqlite3.Row
        # A rollback journal, not WAL: the store may be on a network share,
        # where WAL's shared-memory index does not work.
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        stored = self.meta("schema_version")
        if stored is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        elif int(stored) != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} has manifest schema {stored}; this code reads {SCHEMA_VERSION}"
            )

    @classmethod
    def read_only(cls, path: str | Path) -> Manifest:
        """The manifest opened for reading alone, as a pass that keeps its
        results beside the store rather than in it opens it: no schema is
        written, no journal mode set, and a write raises. Another machine's
        pass may be writing the same file meanwhile."""
        manifest = cls.__new__(cls)
        manifest.path = Path(path)
        if not manifest.path.exists():
            raise FileNotFoundError(manifest.path)
        manifest.db = sqlite3.connect(f"{manifest.path.resolve().as_uri()}?mode=ro", uri=True, timeout=60.0)
        manifest.db.row_factory = sqlite3.Row
        stored = manifest.meta("schema_version")
        if stored is None or int(stored) != SCHEMA_VERSION:
            raise RuntimeError(f"{manifest.path} has manifest schema {stored}; this code reads {SCHEMA_VERSION}")
        return manifest

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit everything done inside, or nothing."""
        with self.db:
            yield self.db

    def meta(self, key: str) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, dumps(value)),
            )

    def upsert(self, table: str, row: Mapping[str, Any], key: str = "path") -> None:
        """Insert `row`, or update the columns it names on the row with the
        same `key`."""
        columns = list(row)
        quoted = ", ".join(f'"{column}"' for column in columns)
        marks = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f'"{column}" = excluded."{column}"' for column in columns if column != key
        )
        self.db.execute(
            f"INSERT INTO {table} ({quoted}) VALUES ({marks}) "
            f'ON CONFLICT("{key}") DO UPDATE SET {updates}',
            [row[column] for column in columns],
        )

    def update(self, table: str, key_column: str, key: str, values: Mapping[str, Any]) -> None:
        """Set the columns `values` names on one row."""
        assignments = ", ".join(f'"{column}" = ?' for column in values)
        self.db.execute(
            f'UPDATE {table} SET {assignments} WHERE "{key_column}" = ?',
            [*values.values(), key],
        )

    def rows(self, query: str, parameters: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.db.execute(query, tuple(parameters)).fetchall()

    def frame_stamps(self) -> dict[str, tuple[int, float]]:
        """Each crawled file's size and modification time, so an unchanged
        file is not read again."""
        return {
            row["path"]: (row["size"], row["mtime"])
            for row in self.db.execute("SELECT path, size, mtime FROM frames")
        }

    def set_frames(self, set_id: str) -> list[sqlite3.Row]:
        """A set's lights, in capture order."""
        return self.rows(
            "SELECT * FROM frames WHERE set_id = ? ORDER BY captured_at_unix, path", (set_id,)
        )

    def review(self, subject: str, kind: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM reviews WHERE subject = ? AND kind = ?", (subject, kind)
        ).fetchone()

    def record_review(
        self, subject: str, kind: str, decision: str, reviewer: str, note: str | None = None
    ) -> None:
        if decision not in ("accept", "reject"):
            raise ValueError(f"a review decides 'accept' or 'reject', not {decision!r}")
        with self.db:
            self.db.execute(
                "INSERT INTO reviews (subject, kind, decision, reviewer, decided_at, note) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(subject, kind) DO UPDATE SET "
                "decision = excluded.decision, reviewer = excluded.reviewer, "
                "decided_at = excluded.decided_at, note = excluded.note",
                (subject, kind, decision, reviewer, now(), note),
            )

    @contextmanager
    def pass_run(self, name: str, versions: Mapping[str, Any], arguments: Mapping[str, Any]):
        """Record one run of pass `name`. The block's result dictionary is
        stored as the run's summary."""
        started = now()
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO passes (name, started, versions, arguments) VALUES (?, ?, ?, ?)",
                (name, started, dumps(dict(versions)), dumps(dict(arguments))),
            )
        rowid = cursor.lastrowid
        summary: dict[str, Any] = {}
        try:
            yield summary
        finally:
            with self.db:
                self.db.execute(
                    "UPDATE passes SET finished = ?, summary = ? WHERE rowid = ?",
                    (now(), dumps(summary), rowid),
                )
