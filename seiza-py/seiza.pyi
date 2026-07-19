"""Type stubs for the seiza extension module."""

from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

StarInput = Union["Star", tuple[float, float, float]]

class SolveError(RuntimeError):
    """The field could not be solved."""

class Star:
    x: float
    y: float
    flux: float
    peak: float
    area: int
    def __init__(
        self,
        x: float,
        y: float,
        flux: float,
        peak: float | None = None,
        area: int = 1,
    ) -> None: ...

class StarCatalog:
    @staticmethod
    def open(path: str | Path | None = None) -> StarCatalog: ...
    @staticmethod
    def from_stars(stars: Sequence[tuple[float, float, float]]) -> StarCatalog: ...
    def star_count(self) -> int: ...
    def cone_search(
        self, ra: float, dec: float, radius_deg: float, limit: int = 100
    ) -> list[tuple[float, float, float]]: ...

class BlindIndex:
    @staticmethod
    def open(path: str | Path | None = None) -> BlindIndex: ...
    def pattern_count(self) -> int: ...

class Wcs:
    @staticmethod
    def from_center_scale_rotation(
        center: tuple[float, float],
        crpix: tuple[float, float],
        scale_arcsec_px: float,
        rotation_deg: float = 0.0,
        flipped: bool = False,
    ) -> Wcs: ...
    @property
    def crval(self) -> tuple[float, float]: ...
    @property
    def crpix(self) -> tuple[float, float]: ...
    @property
    def cd(self) -> tuple[tuple[float, float], tuple[float, float]]: ...
    @property
    def scale_arcsec_px(self) -> float: ...
    @property
    def rotation_deg(self) -> float: ...
    @property
    def flipped(self) -> bool: ...
    @property
    def sip_order(self) -> int | None: ...
    def sip_coefficients(
        self,
    ) -> dict[str, dict[tuple[int, int], float]] | None: ...
    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]: ...
    def world_to_pixel(self, ra: float, dec: float) -> tuple[float, float] | None: ...
    def footprint(self, width: int, height: int) -> list[tuple[float, float]]: ...
    def fits_header_cards(self) -> dict[str, float | int | str]: ...

class Solution:
    @property
    def wcs(self) -> Wcs: ...
    @property
    def matched_stars(self) -> int: ...
    @property
    def rms_arcsec(self) -> float: ...
    @property
    def center(self) -> tuple[float, float]: ...
    @property
    def ra(self) -> float: ...
    @property
    def dec(self) -> float: ...
    @property
    def scale_arcsec_px(self) -> float: ...
    @property
    def rotation_deg(self) -> float: ...
    @property
    def flipped(self) -> bool: ...
    def fits_header_cards(self) -> dict[str, float | int | str]: ...
    def fits_header_text(self) -> str: ...

class TrackSample:
    time_unix: float
    ra_deg: float
    dec_deg: float
    pixel: tuple[float, float] | None
    elevation_deg: float
    range_km: float
    sunlight_fraction: float

class SatelliteTrack:
    norad_id: int | None
    cospar_id: str | None
    name: str
    label: str
    element_epoch_unix: float
    element_age_s: float
    source: str
    sample_interval_s: float
    samples: list[TrackSample]
    clipped_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    max_elevation_deg: float
    max_sunlight_fraction: float
    max_apparent_rate_arcsec_per_s: float | None
    max_pixel_rate_px_per_s: float | None
    clipped_length_px: float

class TrackSearchResult:
    tracks: list[SatelliteTrack]
    elements_considered: int
    propagation_failures: int
    stale_elements: int

class SatelliteCatalog:
    @staticmethod
    def open(path: str | Path) -> SatelliteCatalog: ...
    @staticmethod
    def from_omm_json(text: str, source: str = "inline") -> SatelliteCatalog: ...
    @staticmethod
    def from_tle_text(text: str, source: str = "inline") -> SatelliteCatalog: ...
    @staticmethod
    def fetch_celestrak(cache_dir: str | Path | None = None) -> SatelliteCatalog: ...
    def __len__(self) -> int: ...
    @property
    def source(self) -> str: ...
    @property
    def retrieved_at_unix(self) -> float | None: ...
    @property
    def cache_state(self) -> str | None: ...
    @property
    def cache_path(self) -> Path | None: ...
    @property
    def warning(self) -> str | None: ...
    def tracks_in_footprint(
        self,
        wcs: Wcs,
        width: int,
        height: int,
        *,
        start: float | str | datetime,
        duration_s: float,
        latitude: float | None = None,
        longitude: float | None = None,
        altitude_m: float = 0.0,
        observer_itrf: tuple[float, float, float] | None = None,
        sample_interval_s: float = 1.0,
        coarse_interval_s: float = 10.0,
        min_elevation_deg: float = 0.0,
        max_element_age_s: float | None = 604_800.0,
    ) -> TrackSearchResult: ...

def detect(
    image: npt.NDArray[np.float32] | npt.NDArray[np.uint8],
    *,
    sigma: float = 4.0,
    max_stars: int = 500,
    tile_size: int = 64,
    min_area: int = 3,
    max_area: int = 20_000,
    max_elongation: float = 2.5,
    ignore_border: int = 0,
) -> list[Star]: ...
def solve(
    stars: Sequence[StarInput],
    catalog: StarCatalog,
    width: int,
    height: int,
    *,
    ra: float,
    dec: float,
    scale_arcsec_px: float,
    radius_deg: float = 2.0,
    scale_tolerance: float = 0.2,
    sip_order: int = 0,
) -> Solution: ...
def solve_blind(
    stars: Sequence[StarInput],
    catalog: StarCatalog,
    index: BlindIndex,
    width: int,
    height: int,
    *,
    min_scale_arcsec_px: float = 0.1,
    max_scale_arcsec_px: float = 20.0,
    sip_order: int = 0,
) -> Solution: ...
def fetch_catalogs(
    datasets: str | Sequence[str] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Path]: ...
