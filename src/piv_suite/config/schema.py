"""Canonical PIV settings schema -- one definition, replacing the 4
independent DEFAULT_CONFIG/Controls pairs that existed one-per-repo in the
original scripts. GPU (`piv_settings` dict) and CPU (`PIVSettings`
dataclass fields) each get their own vocabulary via config.legacy's
adapters; nothing here is backend-specific.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ProjectSettings:
    input_mode: str = "set"          # "set" or "loose"
    input_path: str = ""
    output_dir: str = "piv_output"
    backend: str = "cpu"             # "cpu" or "gpu"
    mode: str = "planar"             # "planar" or "stereo"

    # "set" mode only
    multiset_index: int = 0

    # "loose" mode only
    loose_glob: str = "*.im7"
    suffix_a: str = "_a.im7"          # planar
    suffix_b: str = "_b.im7"
    suffix_cam0: str = "_cam1.im7"    # stereo
    suffix_cam1: str = "_cam2.im7"
    stereo_frame_order: str = "camera_major"  # "camera_major" or "frame_major"


@dataclass
class PassSettings:
    """One multi-pass PIV pass: a (square) interrogation window size and
    its overlap fraction. PIV windows are rectangular/square in both
    engines, never arbitrary shapes -- ordered coarse-to-fine, matching
    CPU's windowsizes/overlap convention (which both engines' actual
    defaults agree on once expanded -- see config.legacy)."""
    window_size: int
    overlap_fraction: float


def _default_passes() -> List[PassSettings]:
    # Matches both repos' original defaults once expanded: CPU's
    # windowsizes=[64,32,32,32]/overlap=[32,24,24,24] and GPU's
    # min_search_size=32/search_size_iters=[1,3]/overlap_ratio=[0.5,0.75]
    # both expand to this same 4-pass schedule.
    return [
        PassSettings(64, 0.5),
        PassSettings(32, 0.75),
        PassSettings(32, 0.75),
        PassSettings(32, 0.75),
    ]


@dataclass
class CorrelationSettings:
    passes: List[PassSettings] = field(default_factory=_default_passes)
    dt: float = 1.0
    correlation_method: str = "circular"     # CPU-only; ignored by GPU adapter
    subpixel_method: str = "gaussian"
    deformation_method: str = "symmetric"    # CPU-only; ignored by GPU adapter
    interpolation_order: int = 3             # CPU-only; ignored by GPU adapter
    batch_size: Optional[int] = None         # GPU-only; ignored by CPU adapter

    # ---- GPU tiling (ignored entirely when backend == "cpu") ----
    use_tiling: bool = False
    n_tiles_y: int = 1
    n_tiles_x: int = 1
    tile_margin_px: Optional[int] = None


@dataclass
class ValidationSettings:
    """Per-pass validation/replacement the CPU/GPU engines run internally
    between multi-pass window-deformation passes (openpiv.windef /
    piv_gpu need SOME threshold to decide which vectors get replaced
    before the next pass can deform the image) -- kept as an internal
    dataclass with fixed defaults, no longer exposed in the GUI. The
    user-facing "remove invalid vectors" step is PostProcessSettings'
    std-dev and residual filters instead, applied once after the engine
    has already produced its final field, not this per-pass mechanism."""
    sig2noise_method: str = "peak2mean"
    sig2noise_threshold: float = 1.05
    sig2noise_validate: bool = True
    validation_first_pass: bool = True
    replace_vectors: bool = True
    filter_method: str = "localmean"
    max_filter_iteration: int = 4
    filter_kernel_size: int = 2
    smoothn: bool = False
    smoothn_p: float = 0.05


@dataclass
class RangeFilterSettings:
    """Config surface for processing.postprocess.range_filter -- rejects a
    vector whose distance from its local window median displacement
    exceeds residual_max. This is the sole "remove if residual..."
    detection method (no magnitude/component range option -- see
    PostProcessSettings' docstring for why)."""
    enabled: bool = False
    residual_max: Optional[float] = None
    window_size: int = 3

    def to_kwargs(self):
        """None if disabled or residual_max isn't set (so
        pipeline.process_frames can skip the filter entirely), else the
        kwargs dict for postprocess.range_filter()."""
        if not self.enabled or self.residual_max is None:
            return None
        return {"residual_max": self.residual_max, "window_size": self.window_size}


@dataclass
class PostProcessSettings:
    """Removing invalid vectors uses exactly two detection methods,
    deliberately -- not the engines' own internal per-pass validation
    (see ValidationSettings' docstring): "remove if difference to
    standard deviation exceeds n_std" (global_outlier_std) and "remove if
    residual [from the local window median] exceeds residual_max"
    (range_filter, with a window_size control). replace_invalid/
    smooth_field are a separate, later step (filling gaps in what's left
    after removal), not a third detection method."""
    apply_v_sign_flip: bool = False
    global_outlier_std: Optional[float] = None   # std-dev spurious-vector filter; None disables
    range_filter: RangeFilterSettings = field(default_factory=RangeFilterSettings)
    replace_invalid: bool = False
    smooth_field: bool = False
    smooth_sigma: float = 1.0

    def for_pipeline(self):
        """A tiny namespace matching what processing.pipeline.process_frames
        expects for its `post` argument (range_filter as a kwargs dict or
        None, not a RangeFilterSettings object)."""
        class _Post:
            pass
        p = _Post()
        p.apply_v_sign_flip = self.apply_v_sign_flip
        p.global_outlier_std = self.global_outlier_std
        p.range_filter = self.range_filter.to_kwargs()
        p.replace_invalid = self.replace_invalid
        p.smooth_field = self.smooth_field
        p.smooth_sigma = self.smooth_sigma
        return p


@dataclass
class CalibrationSettings:
    pixel_pitch_mm: Optional[float] = None   # mm/pixel; None keeps units px/frame
    frame_dt_s: Optional[float] = None       # s between frames; None keeps units px/frame


@dataclass
class CameraMappingSettings:
    x0: float = 0.0
    x_span: float = 1.0
    y0: float = 0.0
    y_span: float = 1.0
    dx_coefs: Dict[str, float] = field(default_factory=lambda: {
        k: 0.0 for k in ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")
    })
    dy_coefs: Dict[str, float] = field(default_factory=lambda: {
        k: 0.0 for k in ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")
    })
    name: str = ""


@dataclass
class StereoSettings:
    cam0_mapping: CameraMappingSettings = field(default_factory=CameraMappingSettings)
    cam1_mapping: CameraMappingSettings = field(default_factory=CameraMappingSettings)
    world_shape: Tuple[int, int] = (0, 0)
    world_scale_px_per_mm: float = 1.0
    dewarp_order: int = 1
    alpha1_deg: float = 0.0
    alpha2_deg: float = 0.0
    beta1_deg: float = 0.0
    beta2_deg: float = 0.0


@dataclass
class OutputSettings:
    save_npz: bool = True
    save_plot: bool = False
    save_summary_csv: bool = False
    plot_dpi: int = 150
    quiver_scale: float = 1000
    show_plots: bool = False
    verbose: bool = True


@dataclass
class ProjectConfig:
    """The full, canonical settings tree for one PIV project -- what
    gets saved to/loaded from a `.pivproj` JSON file (config.io)."""
    project: ProjectSettings = field(default_factory=ProjectSettings)
    correlation: CorrelationSettings = field(default_factory=CorrelationSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    postprocess: PostProcessSettings = field(default_factory=PostProcessSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    stereo: StereoSettings = field(default_factory=StereoSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
