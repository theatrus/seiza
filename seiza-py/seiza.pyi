"""Type stubs for the seiza extension module."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str
STACKING_OUTPUT_VERSION: str
"""The ``seiza-stacking`` crate version, not the Seiza release, whose stacking,
master building and registration output this build reproduces. Key cached stacks and masters on it rather than on
``__version__``: it moves only when that output changes."""
STRETCH_OUTPUT_VERSION: str
"""The ``seiza-stretch`` crate version, not the Seiza release, whose stretch
output this build reproduces. Key cached stretched images on it; it moves only
when that output changes."""

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

class MeasuredStar:
    x: float
    y: float
    hfr: float
    fwhm: float
    brightness: float
    background: float
    snr: float
    flux: float
    pixel_count: int
    saturated: bool
    eccentricity: float | None
    # Ellipse major-axis orientation in radians over [0, pi).
    theta: float | None
    r_squared: float | None

class StarDetectionResult:
    stars: list[MeasuredStar]
    average_hfr: float
    average_fwhm: float
    noise_sigma: float
    background_mean: float
    width: int
    height: int

class TiltCell:
    row: int
    col: int
    star_count: int
    median_hfr: float | None
    median_eccentricity: float | None
    mean_theta: float | None
    theta_coherence: float

class TiltSummary:
    center_hfr: float | None
    corners: list[tuple[str, float | None]]
    mean_hfr: float | None
    tilt_percent: float | None
    curvature_percent: float | None
    worst_corner: str | None
    best_corner: str | None

class TriangleTiltCenter:
    star_count: int
    median_hfr: float | None

class TriangleTiltSector:
    # Stable 1-based adjustment-screw identifier.
    sector: int
    # Image-coordinate degrees: 0 points up, positive turns clockwise.
    axis_angle_degrees: float
    star_count: int
    median_hfr: float | None

class TriangleTiltSummary:
    angle_degrees: float
    inner_radius_pixels: float
    outer_radius_pixels: float
    minimum_stars_per_region: int
    ready: bool
    center: TriangleTiltCenter
    sectors: list[TriangleTiltSector]
    # Median HFR of every usable annular star, not the sector medians.
    overall_median_hfr: float | None
    tilt_percent: float | None
    best_sector: int | None
    worst_sector: int | None

class StarCatalog:
    @staticmethod
    def open(path: str | Path | None = None) -> StarCatalog: ...
    @staticmethod
    def from_stars(stars: Sequence[tuple[float, float, float]]) -> StarCatalog: ...
    def star_count(self) -> int: ...
    @property
    def has_proper_motions(self) -> bool: ...
    def cone_search(
        self, ra: float, dec: float, radius_deg: float, limit: int = 100
    ) -> list[tuple[float, float, float]]: ...

class BlindIndex:
    @staticmethod
    def open(path: str | Path | None = None) -> BlindIndex: ...
    def pattern_count(self) -> int: ...

class Wcs:
    @staticmethod
    def from_header(header: Mapping[str, Any]) -> Wcs | None: ...
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
    def transformed(
        self,
        matrix: tuple[tuple[float, float], tuple[float, float]],
        translation_x: float = 0.0,
        translation_y: float = 0.0,
    ) -> Wcs: ...
    def fits_header_cards(self) -> dict[str, float | int | str]: ...

class Solution:
    @property
    def wcs(self) -> Wcs: ...
    @property
    def matched_stars(self) -> int: ...
    @property
    def rms_arcsec(self) -> float: ...
    @property
    def false_match_probability(self) -> float: ...
    @property
    def source(self) -> str | None: ...
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

class ObjectContour:
    closed: bool
    points: list[tuple[float, float]]

class ObjectOutline:
    geometry_id: str
    source_record_id: str
    role: str
    quality: str
    method: str
    evidence: str
    level: str | None
    contours: list[ObjectContour]

class FieldObject:
    stable_id: str | None
    name: str
    common_name: str
    kind: str
    source: str
    catalog_source: str | None
    x: float
    y: float
    semi_major_pixels: float
    semi_minor_pixels: float
    angle_deg: float | None
    prominence: float | None
    ra_deg: float
    dec_deg: float
    discovered: str | None
    near_capture: bool | None
    distance_au: float | None
    motion_arcsec_per_hour: float | None
    direction_position_angle_deg: float | None
    direction_image_angle_deg: float | None
    outlines: list[ObjectOutline]
    @property
    def layer(self) -> str: ...

class FieldObjects:
    objects: list[FieldObject]
    object_catalog_error: str | None
    transient_catalog_error: str | None
    minor_body_error: str | None

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
        duration_s: float | None = None,
        end: float | str | datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        altitude_m: float = 0.0,
        observer_itrf: tuple[float, float, float] | None = None,
        sample_interval_s: float = 1.0,
        coarse_interval_s: float = 10.0,
        min_elevation_deg: float = 0.0,
        max_element_age_s: float | None = 604_800.0,
    ) -> TrackSearchResult: ...

class FrameSignature:
    camera: str | None
    telescope: str | None
    width: int | None
    height: int | None
    channels: int | None
    binning_x: int | None
    binning_y: int | None
    gain: int | None
    offset: int | None
    readout_mode: int | None
    bayer_pattern: str | None
    filter: str | None
    focal_length_mm: float | None
    rotation_deg: float | None
    exposure_seconds: float | None
    camera_temp_c: float | None
    captured_at_unix: int | None
    def __init__(
        self,
        camera: str | None = None,
        telescope: str | None = None,
        width: int | None = None,
        height: int | None = None,
        channels: int | None = None,
        binning_x: int | None = None,
        binning_y: int | None = None,
        gain: int | None = None,
        offset: int | None = None,
        readout_mode: int | None = None,
        bayer_pattern: str | None = None,
        filter: str | None = None,
        focal_length_mm: float | None = None,
        rotation_deg: float | None = None,
        exposure_seconds: float | None = None,
        camera_temp_c: float | None = None,
        captured_at_unix: int | None = None,
    ) -> None: ...

class MatchTolerances:
    exposure_seconds: float
    exposure_fraction: float
    dark_temperature_c: float
    master_temperature_c: float
    rotation_deg: float
    focal_length_mm: float
    flat_session_seconds: int
    light_rotation: float | str
    def __init__(
        self,
        exposure_seconds: float | None = None,
        exposure_fraction: float | None = None,
        dark_temperature_c: float | None = None,
        master_temperature_c: float | None = None,
        rotation_deg: float | None = None,
        focal_length_mm: float | None = None,
        flat_session_seconds: int | None = None,
        light_rotation: float | str | None = None,
    ) -> None: ...

class FrameCalibrationState:
    bias_subtracted: bool
    dark_subtracted: bool
    flat_normalized: bool

class FrameMetadata:
    role: str
    is_master: bool
    signature: FrameSignature
    calibration_state: FrameCalibrationState

class FrameProbe:
    path: Path
    format: str
    raw_image_type: str | None
    role: str
    is_master: bool
    signature: FrameSignature
    calibration_state: FrameCalibrationState

class CalibrationRecord:
    path: Path
    role: str
    signature: FrameSignature
    def __init__(
        self,
        path: str | Path,
        signature: FrameSignature,
        *,
        role: str = "unknown",
    ) -> None: ...

class DarkMeasurement:
    algorithm_version: int
    width: int
    height: int
    level: float
    sigma: float
    noise: float
    row_banding: float
    column_banding: float
    hot_pixels: int
    hot_threshold: float
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> DarkMeasurement: ...

class IlluminationProfile:
    columns: int
    rows: int
    response: npt.NDArray[np.float32]

class FlatFeature:
    x: float
    y: float
    left: int
    top: int
    right: int
    bottom: int
    pixels: int
    depth: float
    shape: str
    radius: float
    hole: float | None
    distance_mm: float | None
    seen_in: tuple[int, int] | None

class FlatMeasurement:
    algorithm_version: int
    width: int
    height: int
    full_scale: float
    level: float
    channel_levels: list[float]
    clipped_fraction: float
    noise: float
    detection_depth: float
    illumination: IlluminationProfile
    feature_count: int
    features: list[FlatFeature]
    vignetting: list[dict[str, str | float]]
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> FlatMeasurement: ...

class Finding:
    code: str
    grade: str
    value: float
    limit: float

class FrameVerdict:
    grade: str
    findings: list[Finding]

class FlatSetReport:
    frames: list[FrameVerdict]
    master: FrameVerdict
    steady_features: int
    changing_features: int
    carried_features: int

class ResidualFlatDiagnostics:
    algorithm_version: int
    sample_count: int
    corrected_samples: int
    total_samples: int
    largest_connected_pixels: int
    minimum_response: float
    maximum_applied_gain: float

class IntegratedFrame:
    path: Path
    finite_samples: int
    integrated_samples: int

class CalibrationPlan:
    kind: str
    minimum: int
    ready: bool
    matched_paths: list[Path]
    selected_paths: list[Path]
    excluded: list[tuple[Path, str]]

class DefectMap:
    width: int
    height: int
    frames: int
    pixels: list[int]
    excess: list[float]
    exposure_seconds: float | None
    def __init__(self, width: int, height: int, pixels: Sequence[int]) -> None: ...
    def __len__(self) -> int: ...

class StackOptions:
    normalization: str
    local_tile_size: int | None
    demosaic: str
    registration_model: str
    weighting: str
    rejection: str
    maximum_drift_pixels: float
    maximum_drift_fraction: float
    defects: DefectMap | None
    match_tolerances: MatchTolerances
    bin_factor: int
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
        minimum_overlap: float | None = None,
        minimum_normalization_gain: float = 0.25,
        maximum_normalization_gain: float = 4.0,
        minimum_integrated_fraction: float = 0.50,
        demosaic: str = "vng",
        match_tolerances: MatchTolerances | None = None,
        registration_model: str = "similarity",
        weighting: str = "equal",
        defects: DefectMap | None = None,
        bin_factor: int = 1,
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
    noise: list[float] | None
    weight: list[float] | None
    @property
    def mapping(self) -> dict[str, Any] | None: ...
    def __bool__(self) -> bool: ...

class StackResult:
    output: Path
    accepted_frames: int
    rejected_frames: int
    width: int
    height: int
    channels: int
    reference_light: Path
    frames: list[FrameDisposition]

class ReferenceScore:
    stars: int
    median_star_area: float
    background: float
    background_variation: float
    score: float

class ReferenceChoice:
    reference_index: int
    reference_path: Path
    ranking: list[int]
    scores: list[ReferenceScore | None]

class ChannelReferenceSelection:
    index: int
    path: Path
    skipped: list[tuple[int, Path, str]]

class SnrSample:
    """One reading of a stack's depth, in the stack's own units."""
    @property
    def frames(self) -> int: ...
    @property
    def noise(self) -> float: ...
    @property
    def background(self) -> float: ...
    @property
    def signal(self) -> float: ...
    @property
    def channel_noise(self) -> list[float]: ...
    @property
    def bin_factor(self) -> int: ...
    @property
    def snr(self) -> float: ...

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
    def write_fits(self, path: str | Path) -> None: ...

class DrizzleResult:
    width: int
    height: int
    channels: int
    scale: int
    accepted_frames: int
    rejected_frames: int
    image: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]
    def write_fits(self, path: str | Path) -> None: ...

class LiveStackState:
    configuration_fingerprint: str
    width: int
    height: int
    channels: int
    accepted_frames: int
    rejected_frames: int
    input_mode: str
    input_paths: list[Path]
    reference_frame: FrameMetadata
    reference_integrated: bool
    reference_image_path: Path | None
    reintegration_unavailable: str | None

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
    @staticmethod
    def open_prepared_fits(
        path: str | Path,
        *,
        options: StackOptions | None = None,
    ) -> LiveStacker: ...
    @staticmethod
    def open_fits_with_reference_image(
        reference_image: str | Path,
        channel_reference: str | Path,
        *,
        options: StackOptions | None = None,
        bias: str | Path | None = None,
        dark: str | Path | None = None,
        flat: str | Path | None = None,
        dark_exposure_seconds: float | None = None,
    ) -> LiveStacker: ...
    @staticmethod
    def open_fits_with_reference_image_candidates(
        reference_image: str | Path,
        channel_references: Sequence[str | Path],
        *,
        options: StackOptions | None = None,
        bias: str | Path | None = None,
        dark: str | Path | None = None,
        flat: str | Path | None = None,
        dark_exposure_seconds: float | None = None,
    ) -> tuple[LiveStacker, ChannelReferenceSelection]: ...
    @staticmethod
    def open_fits_expanding(
        paths: Sequence[str | Path],
        *,
        options: StackOptions | None = None,
        bias: str | Path | None = None,
        dark: str | Path | None = None,
        flat: str | Path | None = None,
        dark_exposure_seconds: float | None = None,
        max_canvas_megasamples: int = 256,
        cancel: Callable[[], bool] | None = None,
    ) -> LiveStacker: ...
    @staticmethod
    def open_context(path: str | Path) -> LiveStacker: ...
    @property
    def accepted_frames(self) -> int: ...
    @property
    def rejected_frames(self) -> int: ...
    @property
    def bin_factor(self) -> int: ...
    @property
    def reference_integrated(self) -> bool: ...
    @property
    def reference_noise(self) -> list[float]: ...
    @property
    def reference_mapping(self) -> dict[str, Any]:
        """Processing provenance; superpixel maps include RGB channel_sites offsets."""
        ...
    def compatible_calibration(
        self,
        light: FrameSignature,
        tolerances: MatchTolerances | None = None,
    ) -> tuple[list[str], list[str]]: ...
    def validate_calibration(
        self,
        light: FrameSignature,
        tolerances: MatchTolerances | None = None,
    ) -> None: ...
    def match_tolerances(self) -> MatchTolerances: ...
    def set_match_tolerances(self, tolerances: MatchTolerances | None = None) -> None: ...
    def set_calibration_fits(
        self,
        *,
        bias: str | Path | None = None,
        dark: str | Path | None = None,
        flat: str | Path | None = None,
        dark_exposure_seconds: float | None = None,
    ) -> None: ...
    def state(self) -> LiveStackState: ...
    def push_fits(self, path: str | Path) -> FrameDisposition: ...
    def push_fits_pipelined(
        self,
        paths: Sequence[str | Path],
        workers: int | None = None,
        max_in_flight_bytes: int | None = None,
        normalized_full_scale: float | None = None,
    ) -> tuple[list[FrameDisposition], PipelineReport]: ...
    def push(self, image: npt.NDArray[np.float32]) -> FrameDisposition: ...
    def save_context(self, path: str | Path) -> None: ...
    @property
    def expanded_lens(self) -> dict[str, float | None] | None: ...
    @property
    def expansion_planning_notes(self) -> list[str]: ...
    def measure_depth(self) -> SnrSample | None: ...
    def snapshot(self) -> StackSnapshot: ...
    def write_fits(self, path: str | Path) -> None: ...
    def retain_frames_for_reintegration(
        self,
        *,
        scratch_directory: str | Path | None = None,
    ) -> None: ...
    def reintegrate(
        self,
        *,
        sigma_low: float = 3.0,
        sigma_high: float = 3.0,
        scratch_directory: str | Path | None = None,
    ) -> StackSnapshot: ...
    def reintegrate_drizzled(
        self,
        *,
        scale: int = 1,
        drop_shrink: float | None = None,
        sigma_low: float = 3.0,
        sigma_high: float = 3.0,
        scratch_directory: str | Path | None = None,
    ) -> tuple[StackSnapshot, DrizzleResult]: ...
    def reintegrate_full_resolution(
        self,
        *,
        refine: bool = True,
        sigma_low: float = 3.0,
        sigma_high: float = 3.0,
        scratch_directory: str | Path | None = None,
    ) -> tuple[StackSnapshot, int, float | None]: ...
    def finish(self, output: str | Path | None = None) -> StackSnapshot: ...

class PipelineReport:
    integrated: int
    rejected: int
    failed: int

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
    defect_pixels_replaced: int
    bias_subtracted: bool
    dark_subtracted: bool
    normalized: bool
    exposure_seconds: float | None
    input_statistics: list[tuple[int, int]]
    masked_samples: int
    input_masked_samples: list[int]
    rejection_method: str
    flat_star_masking: dict[str, Any] | None
    skipped_inputs: list[tuple[Path, str]]
    input_paths: list[Path]

class FlatStarMasking:
    detection_sigma: float
    halo_radius_factor: float
    minimum_radius_pixels: float
    saturation_level: float | None
    minimum_clean_samples: int
    def __init__(
        self,
        *,
        detection_sigma: float | None = None,
        halo_radius_factor: float | None = None,
        minimum_radius_pixels: float | None = None,
        saturation_level: float | None = None,
        minimum_clean_samples: int | None = None,
    ) -> None: ...

class BackgroundModel:
    width: int
    height: int
    channels: int
    reference: list[float]
    model_kind: str
    diagnostics: dict[str, object]
    def samples(
        self,
    ) -> list[tuple[int, int, list[float], float, float, str]]: ...
    def render(self) -> npt.NDArray[np.float32]: ...
    def correct(
        self,
        image: npt.NDArray[np.float32],
        *,
        mode: str = "subtract",
        strength: float = 1.0,
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
def refine_centroids(
    stars: Sequence[StarInput],
    image: npt.NDArray[np.float32] | npt.NDArray[np.uint16],
    *,
    binning: int = 1,
) -> list[Star]: ...
def detect_measured_stars(
    image: npt.NDArray[np.uint16],
    *,
    preset: str | None = None,
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    psf_type: str = "moffat4",
    structure_removal: str | None = None,
    detection_binning: int | None = None,
    keep_saturated: bool | None = None,
    noise_reduction_radius: int | None = None,
    sensitivity: float | None = None,
    target_star_count: int | None = None,
) -> StarDetectionResult: ...
def detect_measured_stars_file(
    path: str | Path,
    *,
    preset: str | None = None,
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    psf_type: str = "moffat4",
    structure_removal: str | None = None,
    detection_binning: int | None = None,
    keep_saturated: bool | None = None,
    noise_reduction_radius: int | None = None,
    sensitivity: float | None = None,
    target_star_count: int | None = None,
) -> StarDetectionResult: ...
def pixel_scale_arcsec(focal_length_mm: float, pixel_size_um: float) -> float | None: ...
def classify_pixel_scale(scale_arcsec_px: float) -> str: ...
def recommend_detection_binning(measured_hfr_px: float) -> int: ...
def tilt_analysis(
    result: StarDetectionResult,
) -> tuple[list[TiltCell], TiltSummary]: ...
def triangle_tilt_analysis(
    result: StarDetectionResult,
    *,
    angle_degrees: float = 0.0,
) -> TriangleTiltSummary: ...

class RcAstroParameter:
    name: str
    flag: str | None
    label: str
    description: str
    type: str
    default: float | bool | int
    min: float | None
    max: float | None

class RcAstroSchema:
    contract_version: int
    cli_version: str
    key: str
    name: str
    ml_version: int | None
    licensed: bool
    license_message: str | None
    parameters: list[RcAstroParameter]

class RcAstroRun:
    primary: Path
    sidecars: list[Path]
    device: str | None
    warnings: list[str]
    cli_version: str
    ml_version: int | None

def rc_astro_locate() -> str | None: ...
def rc_astro_tool_schema(
    tool: str,
    *,
    executable: str | Path | None = None,
) -> RcAstroSchema: ...
def rc_astro_process_file(
    tool: str,
    input: str | Path,
    output: str | Path,
    *,
    parameters: dict[str, float | bool | int] | None = None,
    device: str | None = None,
    executable: str | Path | None = None,
    host: str | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> RcAstroRun: ...
def rc_astro_split_stars(
    image: npt.NDArray[np.float32],
    *,
    executable: str | Path | None = None,
    host: str | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

def plan_parallax_tour(
    width: int,
    height: int,
    wcs: Wcs,
    *,
    objects: str | Path | None = None,
    size: str | tuple[int, int] | None = None,
    targets: int | None = None,
    hold: float = 1.5,
    motion: float = 1.0,
    crop: tuple[int, int, int, int] | None = None,
    framing: str = "fill",
) -> dict[str, Any]: ...

ParallaxImage = Union[str, Path, npt.NDArray[np.float32], npt.NDArray[np.uint8]]

class ParallaxFrames:
    def __iter__(self) -> ParallaxFrames: ...
    def __next__(self) -> npt.NDArray[np.uint8]: ...

class ParallaxVideo:
    def __init__(
        self,
        starless: ParallaxImage,
        stars: ParallaxImage,
        wcs: Wcs,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        focus: tuple[float, float] | None = None,
        distance_pc: float | None = None,
        distance_focus: tuple[float, float] | None = None,
        unmatched_distance_pc: float | None = None,
        objects: str | Path | None = None,
        object_distances: str | Path | None = None,
        star_distances: str | Path | None = None,
        gaia_max_mag: float = 16.0,
        gaia_cache: str | Path | None = None,
        online: bool = True,
        max_stars: int | None = None,
        small_stars: str = "drop",
        keep_galaxies: bool = False,
        dust: bool = True,
        dust_opacity: float = 3.0,
        dust_map: str | Path | npt.NDArray[np.float32] | npt.NDArray[np.uint8] | None = None,
        planes: Sequence[
            tuple[str | Path | npt.NDArray[np.float32] | npt.NDArray[np.uint8], float]
            | tuple[str | Path | npt.NDArray[np.float32] | npt.NDArray[np.uint8], float, str]
        ] = (),
        auto_planes: bool = False,
        distance_overrides: Mapping[str, float] | Sequence[tuple[str, float]] = (),
        crop: tuple[int, int, int, int] | None = None,
        start: str = "focus",
        framing: str = "fill",
        dolly: float = 0.4,
        truck: float = 0.0,
        truck_angle_deg: float = 0.0,
        pan: float = 0.0,
        zoom: float = 1.0,
        zoom_end: float = 1.0,
        rotate_deg: tuple[float, float] = (0.0, 0.0),
        easing: str = "in_out",
        quality: str = "standard",
        growth_limit: float = 4.0,
        fade_from: float = 6.0,
        tour: Sequence[str | Mapping[str, Any]] = (),
        auto_tour: bool | int | Mapping[str, Any] | None = None,
        tour_glide: float = 0.2,
        tour_titles: bool = False,
        tour_loop: bool = False,
        size: str | tuple[int, int] = "1080p",
        seconds: float = 8.0,
        fps: int = 30,
        overlay: bool = False,
        overlay_density: float = 0.6,
        labels: Sequence[tuple[float, float, str] | tuple[float, float, float, str]] = (),
        label_color: str = "#f0f4f8",
        watermark: bool | str | Mapping[str, Any] = False,
        intro_card: str | Path | None = None,
        outro_card: str | Path | None = None,
        card_seconds: float = 3.0,
    ) -> None: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def fps(self) -> int: ...
    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def summary(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> ParallaxFrames: ...
    def frame(self, index: int, format: str = "rgb") -> npt.NDArray[np.uint8]: ...
    def reconfigure(
        self,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        **options: Any,
    ) -> ParallaxVideo: ...
    def plan_tour(
        self,
        auto_tour: bool | int | Mapping[str, Any] | None = None,
        *,
        size: str | tuple[int, int] | None = None,
        framing: str | None = None,
    ) -> dict[str, Any]: ...
    def with_dust_map(
        self,
        dust_map: str | Path | npt.NDArray[np.float32] | npt.NDArray[np.uint8] | None,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ParallaxVideo: ...
    def dust_map(
        self, *, counted: bool = False, size: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32] | None: ...
    def write_dust_map(self, path: str | Path, *, counted: bool = False) -> bool: ...
    def render(
        self,
        on_frame: Callable[[npt.NDArray[np.uint8], int], bool | None],
        *,
        format: str = "rgb",
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> bool: ...
    def write(
        self,
        output: str | Path,
        *,
        encoder: str = "auto",
        codec: str = "h264",
        ffmpeg: str | Path | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> bool: ...

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
    time_limit: float | None = None,
    cancel: Callable[[], bool] | None = None,
    epoch: float | None = None,
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
    time_limit: float | None = None,
    cancel: Callable[[], bool] | None = None,
    epoch: float | None = None,
) -> Solution: ...
def read_solve_prior(path: str | Path) -> dict[str, Any]: ...
def read_capture_time(path: str | Path) -> str | None: ...
def field_objects(
    wcs: Wcs,
    width: int,
    height: int,
    *,
    catalog_dir: str | Path | None = None,
    capture_time: str | None = None,
) -> FieldObjects: ...
def solve_image(
    path: str | Path,
    *,
    catalog_dir: str | Path | None = None,
    min_scale_arcsec_px: float = 0.1,
    max_scale_arcsec_px: float = 20.0,
    sip_order: int = 0,
    time_limit: float | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Solution: ...
def solve_with_prior(
    stars: Sequence[StarInput],
    catalog: StarCatalog,
    width: int,
    height: int,
    prior: Mapping[str, Any],
    *,
    index: BlindIndex | None = None,
    min_scale_arcsec_px: float = 0.1,
    max_scale_arcsec_px: float = 20.0,
    sip_order: int = 0,
    time_limit: float | None = None,
    cancel: Callable[[], bool] | None = None,
    epoch: float | None = None,
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
    model: str = "polynomial",
    degree: int = 2,
    ridge: float = 1.0e-8,
    rbf_smoothing: float = 0.01,
    max_control_points: int = 192,
    allow_radial_basis: bool = False,
    minimum_improvement: float = 0.12,
    samples_per_axis: int = 12,
    sample_radius: int | None = None,
    search_steps: int = 4,
    sample_rejection_sigma: float = 3.5,
    fit_rejection_sigma: float = 3.0,
    fit_rejection_iterations: int = 3,
    border_fraction: float = 0.03,
) -> BackgroundModel: ...
def select_shared_model(
    models: Sequence[BackgroundModel],
    *,
    degree: int = 2,
    ridge: float = 1.0e-8,
    rbf_smoothing: float = 0.01,
    max_control_points: int = 192,
    allow_radial_basis: bool = False,
    minimum_improvement: float = 0.12,
) -> dict[str, Any] | None: ...
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
    crop: str = "none",
    output: str | Path | None = None,
    header: Mapping[str, Any] | None = None,
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
    crop: str = "none",
    output: str | Path | None = None,
    header: Mapping[str, Any] | None = None,
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
    crop: str = "none",
    output: str | Path | None = None,
    header: Mapping[str, Any] | None = None,
) -> npt.NDArray[np.float32]: ...
def combine_narrowband_matrix(
    ha: npt.NDArray[np.float32],
    oiii: npt.NDArray[np.float32],
    sii: npt.NDArray[np.float32] | None = None,
    *,
    matrix: Mapping[str, Mapping[str, float]],
    normalization: str = "percentile",
    black_percentile: float = 0.001,
    white_percentile: float = 0.995,
    normalization_samples: int = 1_000_000,
    crop: str = "none",
    output: str | Path | None = None,
    header: Mapping[str, Any] | None = None,
) -> npt.NDArray[np.float32]: ...
def crop_report(
    channels: dict[str, npt.NDArray[np.float32]],
    *,
    crop: str = "inscribed",
) -> dict[str, object]: ...
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
    target_median: float | None = None,
    shadows_clip: float = -2.8,
    black_point_sigma: float | None = 2.7,
    curves_boost: float = 0.0,
    hdr_amount: float = 0.0,
    hdr_knee: float = 0.75,
    normalize: bool = False,
    target_background: float = 0.15,
    aggressiveness: float = 0.5,
    dynamic_range_compression: float = 0.0,
    scale_separation: int = 256,
    contrast_recovery: float = 0.0,
    saturation_amount: float = 0.0,
    saturation_boost: float = 0.5,
    lightness_mask: bool = True,
) -> npt.NDArray[np.float32]: ...
def checkpoint_depths(total: int) -> list[int]: ...
def stack_fits(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    options: StackOptions | None = None,
    bias: str | Path | None = None,
    dark: str | Path | None = None,
    flat: str | Path | None = None,
    dark_exposure_seconds: float | None = None,
    reference_image: str | Path | None = None,
    reference: str = "first",
    defects: str = "auto",
) -> StackResult: ...
def detect_defects(
    images: Sequence[str | Path],
    *,
    bias: str | Path | None = None,
    dark: str | Path | None = None,
    flat: str | Path | None = None,
    dark_exposure_seconds: float | None = None,
    sigma: float = 5.0,
    persistence: float = 0.6,
    maximum_frames: int = 16,
) -> DefectMap: ...
def choose_reference(
    images: Sequence[str | Path],
    *,
    concurrency: int = 8,
) -> ReferenceChoice: ...
def build_bias(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    cancel: Callable[[], bool] | None = None,
) -> MasterResult: ...
def build_dark(
    images: Sequence[str | Path],
    output: str | Path,
    *,
    bias: str | Path | None = None,
    exposure_seconds: float | None = None,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    cancel: Callable[[], bool] | None = None,
    dark_level_screening: float | None = None,
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
    cancel: Callable[[], bool] | None = None,
    flat_star_masking: FlatStarMasking | None = None,
    scratch_directory: str | Path | None = None,
    defect_suppression: bool = False,
    defect_sigma_low: float | None = None,
    defect_sigma_high: float | None = None,
) -> MasterResult: ...
def integrate_registered_frames(
    images: Sequence[str | Path],
    *,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    minimum_sigma: float | None = None,
    frame_weights: Sequence[Sequence[float]] | None = None,
    scratch_directory: str | Path | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[StackSnapshot, list[IntegratedFrame]]: ...
def frame_noise(image: npt.NDArray[np.float32]) -> list[float] | None: ...
def probe_frame(path: str | Path) -> FrameProbe: ...
def plan_calibration(
    kind: str,
    reference: FrameProbe | CalibrationRecord,
    candidates: Sequence[FrameProbe | CalibrationRecord],
    *,
    references: Sequence[FrameProbe | CalibrationRecord] | None = None,
    minimum: int = 2,
    tolerances: MatchTolerances | None = None,
    bias_available: bool = False,
) -> CalibrationPlan: ...
def sensor_matches(reference: FrameSignature, candidate: FrameSignature) -> bool: ...
def optics_match(
    reference: FrameSignature,
    candidate: FrameSignature,
    tolerances: MatchTolerances | None = None,
) -> bool: ...
def describe_sensor_mismatch(reference: FrameSignature, candidate: FrameSignature) -> str: ...
def describe_optics_mismatch(
    reference: FrameSignature,
    candidate: FrameSignature,
    tolerances: MatchTolerances | None = None,
) -> str: ...
def rotation_matches(
    reference: float | None,
    candidate: float | None,
    tolerance_deg: float | None = None,
) -> bool: ...
def exposure_matches(
    reference: FrameSignature,
    candidate: FrameSignature,
    tolerances: MatchTolerances | None = None,
) -> bool: ...
def temperature_matches(
    reference: FrameSignature,
    candidate: FrameSignature,
    tolerances: MatchTolerances | None = None,
) -> bool: ...
def coherent_subset_indices(
    candidates: Sequence[FrameSignature],
    flats: bool = False,
    minimum: int = 2,
    tolerances: MatchTolerances | None = None,
) -> list[int]: ...
def coherent_subset(
    candidates: Sequence[FrameSignature],
    flats: bool = False,
    minimum: int = 2,
    tolerances: MatchTolerances | None = None,
) -> list[FrameSignature]: ...
def sort_by_proximity(
    frames: Sequence[FrameSignature],
    reference_unix: int | None = None,
) -> list[FrameSignature]: ...
def fit_flat_pedestal(
    light: npt.NDArray[np.float32],
    flat: npt.NDArray[np.float32],
) -> float | None: ...
def measure_dark(path: str | Path, *, hot_sigma: float | None = None) -> DarkMeasurement: ...
def measure_flat(
    path: str | Path,
    *,
    cfa: bool | None = None,
    full_scale: float | None = None,
    feature_depth: float | None = None,
    feature_sigma: float | None = None,
    large_features: bool | None = None,
    focal_ratio: float | None = None,
    pixel_size_um: float | None = None,
    obstruction: float | None = None,
) -> FlatMeasurement: ...
def screen_bias(
    measurements: Sequence[DarkMeasurement],
    *,
    level: tuple[float, float] | None = None,
    noise: tuple[float, float] | None = None,
    banding: tuple[float, float] | None = None,
) -> list[FrameVerdict]: ...
def screen_darks(
    measurements: Sequence[DarkMeasurement],
    *,
    level_excess: tuple[float, float] | None = None,
    noise: tuple[float, float] | None = None,
    hot_pixels: tuple[float, float] | None = None,
    banding: tuple[float, float] | None = None,
) -> list[FrameVerdict]: ...
def screen_flats(
    measurements: Sequence[FlatMeasurement],
    *,
    clipped: tuple[float, float] | None = None,
    bright: tuple[float, float] | None = None,
    dim: tuple[float, float] | None = None,
    illumination: tuple[float, float] | None = None,
    features: tuple[float, float] | None = None,
    changing_features: tuple[float, float] | None = None,
    carried_features: tuple[float, float] | None = None,
    steady_presence: float | None = None,
    steady_depth_range: float | None = None,
    match_distance: int | None = None,
) -> FlatSetReport: ...
def build_residual_flat(
    samples: Sequence[npt.NDArray[np.float32]],
    *,
    minimum_samples: int | None = None,
    minimum_depth: float | None = None,
    minimum_consensus: float | None = None,
    maximum_gain: float | None = None,
    background_edge_fraction: float | None = None,
    smoothing_sigma: float | None = None,
    edge_feather_fraction: float | None = None,
    minimum_corrected_samples: int | None = None,
    minimum_connected_pixels: int | None = None,
) -> tuple[npt.NDArray[np.float32], ResidualFlatDiagnostics]: ...
def validate_residual_flat_response(response: npt.NDArray[np.float32]) -> None: ...
def apply_residual_flat_response(
    response: npt.NDArray[np.float32],
    image: npt.NDArray[np.float32],
    x: int,
    y: int,
) -> npt.NDArray[np.float32]: ...
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

SOLAR_BP_RP: float

class ColorCalibration:
    gains: tuple[float, float, float]
    offsets: tuple[float, float, float]
    background: tuple[float, float, float]
    red_fit: dict[str, float]
    blue_fit: dict[str, float]
    white_bp_rp: float
    aperture_radius: float
    aperture_correction: tuple[float, float, float]
    stars_offered: int
    stars_measured: int
    measurements: list[dict[str, Any]]
    def apply(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...

def gaia_photometry_cone(
    ra: float, dec: float, radius_deg: float, max_mag: float = 15.0
) -> list[dict[str, float | None]]: ...
def gaia_photometry_catalog_cone(
    path: str | Path,
    ra: float,
    dec: float,
    radius_deg: float,
    max_mag: float = 15.0,
) -> list[dict[str, float | None]]: ...
def read_photo_metadata(
    path: str | Path, width: int | None = None, height: int | None = None
) -> dict[str, Any]: ...
def calibrate_color(
    image: npt.NDArray[np.float32],
    wcs: Wcs,
    gaia: Sequence[Mapping[str, float | None]],
    *,
    epoch: float | None = None,
    white_bp_rp: float = ...,
    fit_bp_rp_range: tuple[float, float] = (0.0, 1.8),
    aperture_radius: float | None = None,
    rejection_sigma: float = 3.0,
    minimum_stars: int = 20,
    neutralize_background: bool = True,
    calibrator_max_g: float | None = None,
) -> ColorCalibration: ...

class ColorCalibrationOutcome:
    status: str
    reason: str | None
    detail: str | None
    calibration: ColorCalibration | None

class CalibratedStretch:
    image: npt.NDArray[np.float32]
    calibration: ColorCalibrationOutcome
    stretch_plans: list[dict[str, Any]]
    linked_stages: list[int]

def calibrate_color_image(
    image: npt.NDArray[np.float32],
    wcs: Wcs | None = None,
    *,
    gaia: Sequence[Mapping[str, float | None]] | None = None,
    catalog: str | Path | None = None,
    epoch: float | None = None,
    white_bp_rp: float = ...,
    fit_bp_rp_range: tuple[float, float] = (0.0, 1.8),
    aperture_radius: float | None = None,
    rejection_sigma: float = 3.0,
    minimum_stars: int = 20,
    neutralize_background: bool = True,
    calibrator_max_g: float | None = None,
) -> tuple[npt.NDArray[np.float32], ColorCalibrationOutcome]: ...
def calibrate_and_stretch(
    image: npt.NDArray[np.float32],
    wcs: Wcs | None = None,
    *,
    stretch: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    gaia: Sequence[Mapping[str, float | None]] | None = None,
    catalog: str | Path | None = None,
    epoch: float | None = None,
    white_bp_rp: float = ...,
    fit_bp_rp_range: tuple[float, float] = (0.0, 1.8),
    aperture_radius: float | None = None,
    rejection_sigma: float = 3.0,
    minimum_stars: int = 20,
    neutralize_background: bool = True,
    calibrator_max_g: float | None = None,
) -> CalibratedStretch: ...

# Flat sessions and their dust. Planes are 2-D float32 arrays; results are
# dicts of the core's own JSON, as the C ABI's seiza_flat_* and seiza_dust_*
# calls return it; options are a dict of DustOptions fields.
def flat_bin_factor(width: int, height: int, cfa: bool = False) -> int: ...
def flat_bin_frame(image: npt.NDArray[np.float32], factor: int) -> npt.NDArray[np.float32]: ...
def flat_build_response(
    planes: Sequence[npt.NDArray[np.float32]], bin: int
) -> dict[str, Any] | None: ...
def flat_compare(
    earlier: npt.NDArray[np.float32],
    later: npt.NDArray[np.float32],
    bin: int,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None: ...
def flat_change_map(
    before: npt.NDArray[np.float32], after: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32] | None: ...
def flat_apply_change_map(
    image: npt.NDArray[np.float32], map: npt.NDArray[np.float32], bin: int, divide: bool
) -> None: ...
def flat_lights_background(
    planes: Sequence[npt.NDArray[np.float32]], pedestal: float, bin: int
) -> npt.NDArray[np.float32] | None: ...
def flat_fit_change(
    background: npt.NDArray[np.float32],
    reference: npt.NDArray[np.float32],
    map: npt.NDArray[np.float32],
    reference_side: str,
) -> dict[str, Any]: ...
def flat_combine_verdicts(verdicts: Sequence[str]) -> str: ...
def flat_choose_response(
    background: npt.NDArray[np.float32],
    candidates: Sequence[npt.NDArray[np.float32]],
    alike: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]: ...
def flat_encode_plane(plane: npt.NDArray[np.float32]) -> bytes: ...
def flat_decode_plane(data: bytes) -> npt.NDArray[np.float32] | None: ...
def dust_find(
    plane: npt.NDArray[np.float32], bin: int, options: Mapping[str, Any] | None = None
) -> dict[str, Any] | None: ...
def dust_find_session(
    planes: Sequence[npt.NDArray[np.float32]],
    bin: int,
    pedestal: float,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None: ...
def dust_frame(
    frame: npt.NDArray[np.float32],
    pedestal: float,
    session: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None: ...
def dust_find_changes(
    earlier: npt.NDArray[np.float32],
    later: npt.NDArray[np.float32],
    bin: int,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None: ...
def dust_match_features(
    earlier: Sequence[Mapping[str, Any]],
    later: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]: ...

# Constellation stick figures (Constellation Lines, Marc van der Sluys, CC BY
# 4.0): anything that shows them must carry the returned "attribution".
def constellation_figures() -> dict[str, Any]: ...
def project_constellations(wcs: Wcs, width: int, height: int) -> dict[str, Any]: ...
