"""Type stubs for the seiza extension module."""

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
    def open(path: str | Path) -> StarCatalog: ...
    @staticmethod
    def from_stars(stars: Sequence[tuple[float, float, float]]) -> StarCatalog: ...
    def star_count(self) -> int: ...
    def cone_search(
        self, ra: float, dec: float, radius_deg: float, limit: int = 100
    ) -> list[tuple[float, float, float]]: ...

class BlindIndex:
    @staticmethod
    def open(path: str | Path) -> BlindIndex: ...
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
