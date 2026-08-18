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
    """Config surface for processing.postprocess.range_filter -- both
    "remove residuals above a certain range" and a hard displacement/
    magnitude range are supported, independently toggleable."""
    enabled: bool = False
    u_range: Optional[Tuple[float, float]] = None
    v_range: Optional[Tuple[float, float]] = None
    magnitude_range: Optional[Tuple[float, float]] = None
    residual_max: Optional[float] = None
    neighborhood_size: int = 3

    def to_kwargs(self):
        """None if disabled or no bound is actually set (so
        pipeline.process_frames can skip the filter entirely), else the
        kwargs dict for postprocess.range_filter()."""
        if not self.enabled:
            return None
        kwargs = {}
        if self.u_range is not None:
            kwargs["u_range"] = tuple(self.u_range)
        if self.v_range is not None:
            kwargs["v_range"] = tuple(self.v_range)
        if self.magnitude_range is not None:
            kwargs["magnitude_range"] = tuple(self.magnitude_range)
        if self.residual_max is not None:
            kwargs["residual_max"] = self.residual_max
            kwargs["neighborhood_size"] = self.neighborhood_size
        return kwargs or None


@dataclass
class PostProcessSettings:
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
