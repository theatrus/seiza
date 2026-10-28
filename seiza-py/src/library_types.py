"""Typed calibration records. Optional fields use the core's documented defaults.

These are ordinary dictionaries at runtime; validation remains in Seiza core.
"""
from typing import Any, Literal, Mapping, NotRequired, Required, Sequence, TypedDict

CalibrationAssociation = Literal["automatic", "one_set"]
CalibrationKind = Literal["bias", "dark", "dark_flat", "flat"]
CalibrationRecipeAction = Literal["external", "integrate", "unavailable", "failed", "blocked"]
FlatRetryAction = Literal["publish_first", "try_trimmed", "publish_trimmed", "try_alternate", "publish_alternate", "withhold"]

class CalibrationFlatResponsePlan(TypedDict):
    row: dict[str, Any]
    sample_frames: list[str]

class CalibrationFlatResponsePlans(TypedDict):
    algorithm_version: int
    plans: list[CalibrationFlatResponsePlan]
    limits: list[dict[str, Any]]
    night_boundaries: list[dict[str, Any]]
    unplanned: list[str]
    grades: list[dict[str, Any]]

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

class CalibrationSetting(TypedDict):
    key: str
    value: Any

class CalibrationSettingCheck(TypedDict):
    key: str
    valid: bool
    error: str | None
    text: str | None
    default: bool

class CalibrationRigSettings(TypedDict, total=False):
    rig_uuid: str
    name: str | None
    settings: Sequence[CalibrationSetting]

class CalibrationQuality(TypedDict, total=False):
    frame_uuid: str
    source_fingerprint: str | None
    algorithm_version: int | None
    measurement: Mapping[str, Any] | None
    measure_error: str | None
    grade: str | None
    reason: str | None
    stray_light: str | None
    override_choice: Literal["use", "exclude"] | None

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

class CalibrationSession(TypedDict):
    key: str
    tolerances: CalibrationTolerances
    recipes: list[CalibrationMasterRecipe]
    application: CalibrationApplication
    flat_stability: FlatRetryPolicy
    flat_held_back: list[str]

class CalibrationLibrarySnapshot(TypedDict):
    algorithm_version: int
    selections: list[Mapping[str, Any]]
    grades: list[Mapping[str, Any]]
    sessions: list[CalibrationSession]
    assignments: list[int]
    settings_digest: str | None
    settings_schema: list[Mapping[str, Any]]

class CalibrationMasterFailure(TypedDict):
    session_key: str
    kind: CalibrationKind
    reason: str

class CalibrationPedestalEvidence(TypedDict, total=False):
    session_key: str
    estimates: Sequence[float]
    pinned: float | None

class CalibrationMasterPlanRequest(TypedDict, total=False):
    flat_association: CalibrationAssociation
    dark_association: CalibrationAssociation
    frames: Sequence[CalibrationFrame]
    lights: Sequence[CalibrationLight]
    quality: Sequence[CalibrationQuality]
    global_settings: Sequence[CalibrationSetting]
    rigs: Sequence[CalibrationRigSettings]
    flat_basis: Mapping[str, Any]
    flat_family: Mapping[str, Any] | None
    flat_held_back: Sequence[str]
    flat_star_masking: bool
    flat_stability: FlatRetryPolicy
    master_failures: Sequence[CalibrationMasterFailure]
    pedestal_evidence: Sequence[CalibrationPedestalEvidence]
    force_flat_without_pedestal: bool

class CalibrationMasterCohortRequest(TypedDict, total=False):
    kind: CalibrationKind
    frames: Sequence[CalibrationFrame]
    settings: Sequence[CalibrationSetting]

class CalibrationMasterCohort(TypedDict):
    kept: list[str]
    dropped: list[str]

class FlatAttempt(TypedDict, total=False):
    used: Required[Sequence[str]]
    disagreement: Sequence[int]
    drift_threshold: int
    feature_pixels: int
    memo: bool
    label: str | None

class CalibrationFlatRetryRequest(TypedDict, total=False):
    session: CalibrationSession | None
    candidates: Sequence[CalibrationFrame]
    held_back: Sequence[str]
    settings: Sequence[CalibrationSetting]
    policy: FlatRetryPolicy | None
    first: Required[FlatAttempt]
    trimmed_attempted: bool
    trimmed: FlatAttempt | None
    alternate_attempted: bool
    alternate: FlatAttempt | None

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


class CalibrationFlatLightContext(TypedDict, total=False):
    light_index: Required[int]
    id: int | None
    profile: str | None
    target: str
    excluded: bool
    pedestal_adu: float | None

class CalibrationFlatWorkRequest(TypedDict, total=False):
    library: Mapping[str, Any]
    light_context: Sequence[CalibrationFlatLightContext]

class CalibrationFlatWorkPlans(TypedDict):
    algorithm_version: int
    grouping: dict[str, Any]
    families: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    corrections: list[dict[str, Any]]
    overrides: list[dict[str, Any]]

class CalibrationFlatWorkAnswer(TypedDict, total=False):
    kind: Required[Literal["family", "change"]]
    question: Required[Mapping[str, Any]]
    choice: Mapping[str, Any]
    lights: int
    fits: Sequence[Mapping[str, Any]]

class CalibrationFlatWorkResult(TypedDict):
    kind: Literal["family", "change"]
    row: dict[str, Any]

_calibration_exports = ["CalibrationFlatLightContext", "CalibrationFlatWorkRequest", "CalibrationFlatWorkPlans", "CalibrationFlatWorkAnswer", "CalibrationFlatWorkResult",'CalibrationAssociation', 'CalibrationFlatResponsePlan', 'CalibrationFlatResponsePlans', 'CalibrationLibrarySnapshot', 'CalibrationKind', 'CalibrationRecipeAction', 'FlatRetryAction', 'CalibrationFrame', 'CalibrationLight', 'CalibrationSetting', 'CalibrationSettingCheck', 'CalibrationRigSettings', 'CalibrationQuality', 'FlatRetryPolicy', 'CalibrationLightRotation', 'CalibrationTolerances', 'CalibrationMasterRecipe', 'CalibrationApplication', 'CalibrationSession', 'CalibrationMasterFailure', 'CalibrationPedestalEvidence', 'CalibrationMasterPlanRequest', 'CalibrationMasterCohortRequest', 'CalibrationMasterCohort', 'FlatAttempt', 'CalibrationFlatRetryRequest', 'CalibrationFlatRetryDecision', 'FlatStabilityInput', 'FlatStabilityRequest', 'FlatSetStability', 'FlatStabilityBayer', 'FlatStabilityImageMetadata']
