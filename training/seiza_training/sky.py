"""Small sky-geometry helpers: header coordinates, separations, field sizes."""

from __future__ import annotations

import math

ARCSEC_PER_RADIAN = 206_264.806


def parse_sexagesimal(text: str, *, hours: bool) -> float | None:
    """Degrees from `"09 56 15"` or `"+69:12:41"`, read as hours when
    `hours`; `None` when the text is not three numbers."""
    parts = text.replace(":", " ").split()
    if len(parts) != 3:
        return None
    try:
        whole, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    sign = -1.0 if parts[0].startswith("-") else 1.0
    value = abs(whole) + minutes / 60.0 + seconds / 3600.0
    return sign * value * (15.0 if hours else 1.0)


def header_position(header: dict) -> tuple[float | None, float | None]:
    """The pointing a header records, in degrees: `RA`/`DEC` in degrees,
    or else `OBJCTRA`/`OBJCTDEC` in sexagesimal."""
    ra, dec = header.get("RA"), header.get("DEC")
    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        return float(ra), float(dec)
    ra, dec = header.get("OBJCTRA"), header.get("OBJCTDEC")
    if isinstance(ra, str) and isinstance(dec, str):
        return parse_sexagesimal(ra, hours=True), parse_sexagesimal(dec, hours=False)
    return None, None


def separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Angular separation of two positions, in degrees (haversine)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = (
        math.sin((dec2 - dec1) / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2.0) ** 2
    )
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))


def pixel_scale_arcsec(
    pixel_size_um: float | None, focal_length_mm: float | None, binning: int = 1
) -> float | None:
    """Arcseconds per (binned) pixel, from the pixel size and focal length."""
    if not pixel_size_um or not focal_length_mm or focal_length_mm <= 0:
        return None
    return ARCSEC_PER_RADIAN * pixel_size_um * 1e-3 * binning / focal_length_mm


def field_diagonal_deg(scale_arcsec: float | None, width: int | None, height: int | None) -> float | None:
    """The field's diagonal, in degrees."""
    if not scale_arcsec or not width or not height:
        return None
    return scale_arcsec * math.hypot(width, height) / 3600.0


class UnionFind:
    """Disjoint sets over string keys."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            # The smaller key is the root, so groups name themselves stably.
            if right < left:
                left, right = right, left
            self.parent[right] = left
