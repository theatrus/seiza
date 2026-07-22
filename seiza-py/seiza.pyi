"""Type stubs for the seiza extension module."""

from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

StarInput = Union["Star", tuple[float, float, float]]

class EngineError(RuntimeError):
    """Stretching, color composition, background fitting, or deconvolution failed."""

class SolveError(RuntimeError):
    """The field could not be solved."""

class StackError(RuntimeError):
    """Calibration, registration, or stacking failed."""

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
    @staticmethod
    def resolve(
        time: float | str | datetime, cache_dir: str | Path | None = None
    ) -> SatelliteCatalog: ...
    def __len__(self) -> int: ...
    @property
    def source(self) -> str: ...
    @property
    def retrieved_at_unix(self) -> float | None: ...
    @property
    def provider(self) -> str | None: ...
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

class StackOptions:
    normalization: str
    local_tile_size: int | None
    rejection: str
    maximum_drift_pixels: float
    maximum_drift_fraction: float
    def __init__(
        self,
        *,
        normalization: str = "global",
        local_tile_size: int = 256,
        rejection: str = "delta-sigma",
        sigma_low: float = 3.0,
        sigma_high: float = 3.0,
        rejection_warmup: int = 5,
        rejection_minimum_sigma: float = 1.0e-6,
        detection_sigma: float = 4.0,
        maximum_stars: int = 200,
        triangle_stars: int = 24,
        descriptor_tolerance: float = 0.015,
        scale_tolerance: float = 0.08,
        match_tolerance_pixels: float = 2.5,
        maximum_drift_pixels: float = 256.0,
        maximum_drift_fraction: float = 0.15,
        minimum_matches: int = 6,
        maximum_candidates: int = 384,
        maximum_registration_rms: float = 2.0,
        maximum_scale_deviation: float = 0.04,
        maximum_rotation_degrees: float = 10.0,
        minimum_overlap: float = 0.60,
        minimum_normalization_gain: float = 0.25,
        maximum_normalization_gain: float = 4.0,
        minimum_integrated_fraction: float = 0.50,
    ) -> None: ...

class FrameDisposition:
    source: Path | None
    accepted: bool
    reason: str | None
    matched_stars: int | None
    registration_rms_pixels: float | None
    registration_drift_pixels: float | None
    scale: float | None
    rotation_degrees: float | None
    translation_x: float | None
    translation_y: float | None
    normalization_mean_gain: float | None
    normalization_mean_offset: float | None
    overlap_fraction: float | None
    integrated_fraction: float | None
    accepted_samples: int | None
    rejected_samples: int | None
    def __bool__(self) -> bool: ...

class StackResult:
    output: Path
    accepted_frames: int
    rejected_frames: int
    width: int
    height: int
    channels: int
    frames: list[FrameDisposition]

class StackSnapshot:
    width: int
    height: int
    channels: int
    accepted_frames: int
    rejected_frames: int
    image: npt.NDArray[np.float32]
    variance: npt.NDArray[np.float32]
    coverage: npt.NDArray[np.uint32]
    rejected_samples: npt.NDArray[np.uint32]

class LiveStacker:
    def __init__(
        self,
        reference: str | Path,
        *,
        options: StackOptions | None = None,
        bias: str | Path | None = None,
        dark: str | Path | None = None,
        flat: str | Path | None = None,
        dark_exposure_seconds: float | None = None,
    ) -> None: ...
    @staticmethod
    def from_array(
        reference: npt.NDArray[np.float32],
        *,
        options: StackOptions | None = None,
    ) -> LiveStacker: ...
    @property
    def accepted_frames(self) -> int: ...
    @property
    def rejected_frames(self) -> int: ...
    def push_fits(self, path: str | Path) -> FrameDisposition: ...
    def push(self, image: npt.NDArray[np.float32]) -> FrameDisposition: ...
    def snapshot(self) -> StackSnapshot: ...
    def finish(self, output: str | Path | None = None) -> StackSnapshot: ...

class MasterResult:
    output: Path
    kind: str
    width: int
    height: int
    channels: int
    input_frames: int
    accepted_samples: int
    rejected_samples: int
    fallback_pixels: int
    bias_subtracted: bool
    dark_subtracted: bool
    normalized: bool
    exposure_seconds: float | None
    input_statistics: list[tuple[int, int]]

class BackgroundModel:
    width: int
    height: int
    channels: int
    reference: list[float]
    diagnostics: dict[str, int]
    def samples(
        self,
    ) -> list[tuple[int, int, list[float], float, float, str]]: ...
    def render(self) -> npt.NDArray[np.float32]: ...
    def correct(
        self,
        image: npt.NDArray[np.float32],
        *,
        mode: str = "subtract",
    ) -> npt.NDArray[np.float32]: ...

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
def fit_background(
    image: npt.NDArray[np.float32],
    *,
    mask: npt.NDArray[np.bool_] | None = None,
    degree: int = 2,
    ridge: float = 1.0e-8,
    samples_per_axis: int = 12,
    sample_radius: int | None = None,
    search_steps: int = 4,
    sample_rejection_sigma: float = 3.5,
    fit_rejection_sigma: float = 3.0,
    fit_rejection_iterations: int = 3,
    border_fraction: float = 0.03,
) -> BackgroundModel: ...
def deconvolve(
    image: npt.NDArray[np.float32],
    *,
    psf_fwhm: float,
    iterations: int = 4,
    amount: float = 0.35,
    noise_fraction: float = 0.001,
    max_correction: float = 2.0,
    masked: bool = False,
) -> npt.NDArray[np.float32]: ...
def combine_rgb(
    red: npt.NDArray[np.float32],
    green: npt.NDArray[np.float32],
    blue: npt.NDArray[np.float32],
    *,
    luminance_mode: str = "native",
    normalization: str = "percentile",
    black_percentile: float = 0.001,
    white_percentile: float = 0.995,
    normalization_samples: int = 1_000_000,
) -> npt.NDArray[np.float32]: ...
def combine_lrgb(
    luminance: npt.NDArray[np.float32],
    red: npt.NDArray[np.float32],
    green: npt.NDArray[np.float32],
    blue: npt.NDArray[np.float32],
    *,
    luminance_weight: float = 1.0,
    luminance_mode: str = "replace",
    normalization: str = "percentile",
    black_percentile: float = 0.001,
    white_percentile: float = 0.995,
    normalization_samples: int = 1_000_000,
) -> npt.NDArray[np.float32]: ...
def combine_narrowband(
    ha: npt.NDArray[np.float32],
    oiii: npt.NDArray[np.float32],
    sii: npt.NDArray[np.float32] | None = None,
    *,
    palette: str = "sho",
    normalization: str = "percentile",
    black_percentile: float = 0.001,
    white_percentile: float = 0.995,
    normalization_samples: int = 1_000_000,
    foraxx_target_median: float = 0.2,
    foraxx_shadows_clip: float = -2.8,
) -> npt.NDArray[np.float32]: ...
def stretch(
    image: npt.NDArray[np.float32],
    *,
    model: str = "percentile-asinh",
    color_strategy: str = "linked",
    max_analysis_samples: int = 200_000,
    black: float = 0.0,
    white: float = 1.0,
    strength: float = 10.0,
    black_percentile: float = 0.01,
    white_percentile: float = 0.995,
    shadows: float = 0.0,
    midtone: float = 0.5,
    highlights: float = 1.0,
    stretch_factor: float = 1.0,
    local_intensity: float = 0.0,
    symmetry_point: float = 0.0,
    protect_shadows: float = 0.0,
    protect_highlights: float = 1.0,
    target_median: float = 0.2,
    shadows_clip: float = -2.8,
) -> npt.NDArray[np.float32]: ...
def stack_fits(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    options: StackOptions | None = None,
    bias: str | Path | None = None,
    dark: str | Path | None = None,
    flat: str | Path | None = None,
    dark_exposure_seconds: float | None = None,
) -> StackResult: ...
def build_bias(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> MasterResult: ...
def build_dark(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    bias: str | Path | None = None,
    exposure_seconds: float | None = None,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> MasterResult: ...
def build_flat(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    bias: str | Path | None = None,
    dark_flat: str | Path | None = None,
    dark_flat_exposure_seconds: float | None = None,
    exposure_seconds: float | None = None,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> MasterResult: ...
def gaussian_blur(
    image: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
    sigma: float,
    *,
    ksize: int = 0,
    border: str = "reflect101",
) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]: ...
def median_blur3(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
def canny(
    image: npt.NDArray[np.uint8],
    low: int,
    high: int,
) -> npt.NDArray[np.uint8]: ...
def otsu_threshold(image: npt.NDArray[np.uint8]) -> int: ...
def otsu_binary(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
def erode(
    image: npt.NDArray[np.uint8],
    *,
    shape: str = "ellipse",
    ksize: int = 3,
    border: str = "ignore",
) -> npt.NDArray[np.uint8]: ...
def dilate(
    image: npt.NDArray[np.uint8],
    *,
    shape: str = "ellipse",
    ksize: int = 3,
    border: str = "ignore",
) -> npt.NDArray[np.uint8]: ...
def find_contours(image: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.int32]]: ...
def contour_area(contour: npt.NDArray[np.int32]) -> float: ...
def dt_filter(
    guide: npt.NDArray[np.float32],
    src: npt.NDArray[np.float32],
    sigma_spatial: float,
    sigma_color: float,
    *,
    num_iters: int = 3,
) -> npt.NDArray[np.float32]: ...
def remove_structures(
    image: npt.NDArray[np.float64],
    *,
    layers: int = 4,
    method: str = "filtered",
) -> npt.NDArray[np.float64]: ...
