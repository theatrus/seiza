"""Pass 4: give every set its train, validation or test group, before any
product is built.

Sets of one object, or of fields that overlap on the sky, share a group,
whatever telescope or night they came from, so adjacent sky, its stars and
its processing never cross a split. A group keeps the split it was first
given; a set that joins it later inherits it.

One telescope can be held out: its sets are `held-out`, never trained on
and scored apart, whatever their group's split. The telescopes image the
same objects, so moving every group a telescope touches to test would move
much of the corpus; a held-out telescope tests the instrument, not the sky.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from . import sky
from .manifest import Manifest

SPLITS = ("train", "validation", "test")
HELD_OUT = "held-out"
DEFAULT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def object_key(target: str | None) -> str | None:
    """A target folder's object, so `NGC7000` and `NGC 7000`, or a
    mosaic's `Panel 2`, name one object."""
    if not target:
        return None
    key = target.lower()
    key = re.sub(r"\bpanel\s*\d+\b", "", key)
    key = re.sub(r"[^a-z0-9]+", "", key)
    return key or None


def inscribed_radius(row: Any) -> float:
    """Half the short side of a set's field, in degrees."""
    diagonal, width, height = row["field_deg"], row["width"], row["height"]
    if not diagonal or not width or not height:
        return 0.0
    return 0.5 * diagonal * min(width, height) / math.hypot(width, height)


def sky_groups(sets: list[Any]) -> dict[str, str]:
    """Each set's group: sets of one object, or whose fields overlap."""
    groups = sky.UnionFind()
    by_object: dict[str, list[str]] = defaultdict(list)
    for row in sets:
        groups.find(row["id"])
        key = object_key(row["target"])
        if key is not None:
            by_object[key].append(row["id"])
    for members in by_object.values():
        for member in members[1:]:
            groups.union(members[0], member)
    placed = [row for row in sets if row["ra_deg"] is not None and row["dec_deg"] is not None]
    for index, left in enumerate(placed):
        for right in placed[index + 1:]:
            # Centres closer than the sum of the fields' inscribed radii
            # share sky. The circumscribed circles would chain neighbouring
            # fields across a whole constellation.
            reach = inscribed_radius(left) + inscribed_radius(right)
            if reach <= 0.0:
                continue
            if sky.separation_deg(left["ra_deg"], left["dec_deg"], right["ra_deg"], right["dec_deg"]) < reach:
                groups.union(left["id"], right["id"])
    return {row["id"]: groups.find(row["id"]) for row in sets}


def draw_split(group: str, salt: str, fractions: dict[str, float]) -> str:
    """A group's split, from a hash of its name: stable across runs."""
    digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for split in SPLITS:
        edge += fractions[split]
        if value < edge:
            return split
    return SPLITS[-1]


def assign_splits(
    manifest: Manifest,
    *,
    hold_out: str | None = None,
    salt: str | None = None,
    fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assign every set's group and split, keeping splits already given."""
    stored = manifest.meta("splits") or {}
    salt = stored.get("salt") or salt or "seiza-restoration-2026"
    fractions = stored.get("fractions") or fractions or DEFAULT_FRACTIONS
    if stored.get("hold_out") is not None and hold_out not in (None, stored["hold_out"]):
        raise ValueError(
            f"the corpus already holds out {stored['hold_out']!r}; a different telescope "
            "would move sets between splits"
        )
    hold_out = stored.get("hold_out", hold_out)
    manifest.set_meta("splits", {"salt": salt, "fractions": fractions, "hold_out": hold_out})

    sets = manifest.rows("SELECT * FROM sets")
    group_of = sky_groups(sets)
    members: dict[str, list[Any]] = defaultdict(list)
    for row in sets:
        members[group_of[row["id"]]].append(row)
    conflicts = []
    group_split: dict[str, str] = {}
    with manifest.transaction():
        for group, rows in members.items():
            given = {row["split"] for row in rows if row["split"] in SPLITS}
            if len(given) > 1:
                # Groups merged that were split apart before: keep the most
                # protected split, and report it.
                conflicts.append({"group": group, "splits": sorted(given)})
                split = next(split for split in reversed(SPLITS) if split in given)
            elif given:
                split = given.pop()
            else:
                split = draw_split(group, salt, fractions)
            group_split[group] = split
            for row in rows:
                held = hold_out is not None and row["telescope"] == hold_out
                manifest.update(
                    "sets", "id", row["id"],
                    {"sky_group": group, "split": HELD_OUT if held else split},
                )
        stacks = assign_stack_splits(manifest, sets, group_of, group_split, hold_out, salt, fractions)
    counts = Counter(
        (row["split"]) for row in manifest.rows("SELECT split FROM sets")
    )
    frames = Counter()
    for row in manifest.rows("SELECT split, frames FROM sets"):
        frames[row["split"]] += row["frames"] or 0
    return {
        "groups": len(members),
        "sets": dict(counts),
        "frames": dict(frames),
        "hold_out": hold_out,
        "conflicts": conflicts,
        "stacks": stacks,
    }


def assign_stack_splits(
    manifest: Manifest,
    sets: list[Any],
    group_of: dict[str, str],
    group_split: dict[str, str],
    hold_out: str | None,
    salt: str,
    fractions: dict[str, float],
) -> dict[str, int]:
    """Put each finished stack in the sky group of the sets its frames came
    from, or else of the sets of its object, or else in a group of its own.
    A stack's frames are its sets' frames, so it must never cross a split
    from them."""
    by_object: dict[str, str] = {}
    for row in sets:
        key = object_key(row["target"])
        if key is not None:
            by_object.setdefault(key, group_of[row["id"]])
    counts: Counter[str] = Counter()
    for row in manifest.rows("SELECT * FROM stacks"):
        groups = {group_of[set_id] for set_id in json.loads(row["sets"] or "[]") if set_id in group_of}
        key = object_key(row["target"])
        if not groups and key in by_object:
            groups = {by_object[key]}
        if groups:
            given = {group_split[group] for group in groups}
            # Groups that disagree keep the stack in the most protected split.
            split = next(split for split in reversed(SPLITS) if split in given)
            group = sorted(groups)[0]
        else:
            group = f"object:{key or row['id']}"
            split = row["split"] if row["split"] in SPLITS else draw_split(group, salt, fractions)
        held = hold_out is not None and row["telescope"] == hold_out
        split = HELD_OUT if held else split
        manifest.update("stacks", "id", row["id"], {"sky_group": group, "split": split})
        counts[split] += 1
    return dict(counts)
