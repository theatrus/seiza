"""Typed calibration records: the values the native classes take and give.

These are ordinary dictionaries at runtime; validation remains in Seiza core.
Optional fields use the core's documented defaults.
"""
from typing import Any, Literal, Mapping, NotRequired, Required, Sequence, TypedDict

CalibrationAssociation = Literal["automatic", "one_set"]
CalibrationKind = Literal["bias", "dark", "dark_flat", "flat"]
CalibrationRecipeAction = Literal["external", "integrate", "unavailable", "failed", "blocked"]
FlatRetryAction = Literal["publish_first", "try_trimmed", "publish_trimmed", "try_alternate", "publish_alternate", "withhold"]

class CalibrationFrame(TypedDict, total=False):
    id: int
    frame_uuid: Required[str]
    rig_uuid: str
    kind: CalibrationKind
    source_path: str
    source_fingerprint: str
    captured_at: int | None
    telescope: str | None
    camera: str | None
    width: int | None
    height: int | None
    channels: int | None
    binning_x: int | None
    binning_y: int | None
    gain: int | None
    offset: int | None
    readout_mode: int | None
    readout_mode_name: str | None
    is_master: bool
    bayer_pattern: str | None
    exposure_s: float | None
    camera_temp: float | None
    filter: str | None
    focal_length_mm: float | None
    rotation: float | None
    valid_direction: Literal["forward", "backward"] | None
    kept_out: str | None
    source_verified: bool

class CalibrationLight(TypedDict, total=False):
    path: str
    rig_uuid: str
    camera: str | None
    telescope: str | None
    bayer_pattern: str | None
    filter: str | None
    width: int | None
    height: int | None
    channels: int | None
    binning_x: int | None
    binning_y: int | None
    gain: int | None
    offset: int | None
    readout_mode: int | None
    readout_mode_name: str | None
    focal_length_mm: float | None
    rotator_position: float | None
    exposure_s: float | None
    camera_temp: float | None
    timestamp: int | None

class CalibrationRigLayer(TypedDict, total=False):
    name: str | None
    values: Mapping[str, Any]

class CalibrationNightBoundary(TypedDict):
    rig_uuid: str
    frame_uuid: str
    seconds: int

class CalibrationSettingCheck(TypedDict):
    key: str
    valid: bool
    error: str | None
    text: str | None
    default: bool

class CalibrationQualityRecord(TypedDict, total=False):
    frame_uuid: str
    source_fingerprint: str | None
    algorithm_version: int | None
    measurement: Mapping[str, Any] | None
    measure_error: str | None
    grade: Literal["ok", "suspect", "poor"] | None
    reason: str | None
    stray_light: str | None
    override_choice: Literal["use", "exclude"] | None

class CalibrationJudged(TypedDict):
    grade: Literal["ok", "suspect", "poor"]
    code: str | None
    reason: str | None
    findings: list[dict[str, Any]]
    stray_light: str | None

class CalibrationCohort(TypedDict):
    kept: list[CalibrationFrame]
    dropped: list[CalibrationFrame]

class CalibrationEvidenceLight(TypedDict):
    path: str
    at: int
    angle: NotRequired[float | None]
    filter: NotRequired[str]
    target: NotRequired[str]

class CalibrationPedestalLevel(TypedDict):
    rig_uuid: str
    gain: NotRequired[int | None]
    offset: NotRequired[int | None]
    kind: CalibrationKind
    exposure_s: NotRequired[float | None]
    level: float

class FlatRetryPolicy(TypedDict, total=False):
    """All flags default true. enabled controls retry decisions, not measurement."""
    enabled: bool
    trim_unstable: bool
    try_next_session: bool
    retain_unstable: bool

class CalibrationLightRotation(TypedDict):
    basis: Literal["recorded", "measured", "unknown"]
    degrees: NotRequired[float]

class CalibrationTolerances(TypedDict):
    exposure_seconds: float
    exposure_fraction: float
    dark_temperature_c: float
    master_temperature_c: float
    rotation_deg: float
    focal_length_mm: float
    flat_session_seconds: int
    allow_flat_gain_offset_mismatch: NotRequired[bool]
    light_rotation: NotRequired[CalibrationLightRotation]

class CalibrationMasterRecipe(TypedDict):
    kind: CalibrationKind
    action: CalibrationRecipeAction
    inputs: list[str]
    dependencies: list[CalibrationKind]
    dropped: list[str]
    coherent_inputs: list[str]
    input_limit: int
    requires_source_verification: bool
    failure: str | None
    exposure_seconds: float | None
    defect_suppression: bool
    flat_star_masking: bool
    dark_level_screening: bool

class CalibrationApplication(TypedDict):
    flat_use: Literal["unavailable", "with_masters", "needs_pedestal_fit", "fitted_pedestal", "forced_without_pedestal", "withheld"]
    pedestal: float | None
    sample_lights: list[str]
    pedestal_hint: float | None

class CalibrationSessionRecord(TypedDict):
    key: str
    tolerances: CalibrationTolerances
    recipes: list[CalibrationMasterRecipe]
    application: CalibrationApplication
    flat_stability: FlatRetryPolicy
    flat_held_back: list[str]

class FlatAttempt(TypedDict, total=False):
    used: Required[Sequence[str]]
    disagreement: Sequence[int]
    drift_threshold: int
    feature_pixels: int
    memo: bool
    label: str | None

class CalibrationFlatRetryDecision(TypedDict):
    action: FlatRetryAction
    inputs: list[str]
    rebuild_first: bool
    note: str | None

class FlatStabilityInput(TypedDict):
    path: str
    exposure_seconds: NotRequired[float | None]
    skipped: NotRequired[bool]

class FlatStabilityRequest(TypedDict):
    master: str
    inputs: Sequence[FlatStabilityInput]
    bias: NotRequired[str | None]
    dark: NotRequired[str | None]
    dark_exposure_seconds: NotRequired[float | None]
    normalized_full_scale: NotRequired[float | None]

class FlatSetStability(TypedDict):
    algorithm_version: int
    errors: list[list[int | str]]
    feature_pixels: int
    drift_threshold: int
    disagreement: list[int]

class FlatStabilityBayer(TypedDict):
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"]
    x_offset: NotRequired[int]
    y_offset: NotRequired[int]

class FlatStabilityImageMetadata(TypedDict, total=False):
    exposure_seconds: float | None
    bayer: FlatStabilityBayer | None
    skipped: bool

_calibration_exports = [
    'CalibrationAssociation',
    'CalibrationKind',
    'CalibrationRecipeAction',
    'FlatRetryAction',
    'CalibrationFrame',
    'CalibrationLight',
    'CalibrationRigLayer',
    'CalibrationNightBoundary',
    'CalibrationSettingCheck',
    'CalibrationQualityRecord',
    'CalibrationJudged',
    'CalibrationCohort',
    'CalibrationEvidenceLight',
    'CalibrationPedestalLevel',
    'FlatRetryPolicy',
    'CalibrationLightRotation',
    'CalibrationTolerances',
    'CalibrationMasterRecipe',
    'CalibrationApplication',
    'CalibrationSessionRecord',
    'FlatAttempt',
    'CalibrationFlatRetryDecision',
    'FlatStabilityInput',
    'FlatStabilityRequest',
    'FlatSetStability',
    'FlatStabilityBayer',
    'FlatStabilityImageMetadata',
]
