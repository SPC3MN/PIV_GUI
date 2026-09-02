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

    # planar-only: DaVis "SideBySide2D" acquisition -- two COPLANAR
    # cameras imaging different (overlapping) regions of one larger flat
    # plane, stitched into one wider field, as opposed to stereo's two
    # cameras at an angle on the SAME region for triangulated 3-component
    # velocity. Deliberately a flag alongside mode="planar" rather than a
    # 3rd mode string: the correlation itself is ordinary single-camera
    # planar PIV run twice (see pipeline.combine_dual_planar_pair), not a
    # different processing model the way stereo's triangulation is. See
    # config.schema.DualPlanarSettings and
    # davis_set.read_dual_planar_calibration_from_set.
    dual_camera: bool = False

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
    # 5, not 3. This is the spline order used to RESAMPLE THE IMAGES during
    # window deformation, and it is the dominant remaining source of
    # sub-pixel bias. Measured max |bias| over a sweep of sub-pixel shifts:
    # order 1 (bilinear) 0.0659 px, order 3 0.0197 px, order 5 0.0069 px.
    # The residual periodic bias this app shows is NOT classic peak-locking
    # (it has period 2.0 px in total displacement, i.e. period 1 in the
    # HALF-displacement windef applies to each frame for the symmetric
    # split) -- it is deformation-interpolation error, so raising this is
    # the lever that actually moves it.
    interpolation_order: int = 5             # CPU-only; ignored by GPU adapter
    batch_size: Optional[int] = None         # GPU-only; ignored by CPU adapter

    # ---- GPU tiling (ignored entirely when backend == "cpu") ----
    use_tiling: bool = False
    n_tiles_y: int = 1
    n_tiles_x: int = 1
    tile_margin_px: Optional[int] = None


@dataclass
class ValidationSettings:
    """Internal per-pass NUMERICAL STABILITY mechanism only -- NOT
    validation in the user-facing sense, despite the class name (kept for
    config-file/backward-compatibility continuity). CPU/GPU both need
    SOME per-pass fill so a NaN cell doesn't poison the next pass's
    deformation grid (openpiv.windef.multipass_img_deform calls
    validation.typical_validation + filters.replace_outliers
    unconditionally, with no flag to disable it) -- but on the CPU
    backend that per-pass check is patched (see
    engines/_openpiv_speedups.loose_typical_validation) to flag literal
    NaN ONLY, never a real outlier, and on the GPU backend
    config.legacy.to_gpu_settings hard-codes every ValidationGPU
    tolerance to None so nothing is ever flagged there either. No vector
    is ever counted invalid because of anything here -- UNLESS
    per_pass_validation is turned on (see below; ON by default). The
    user-facing "remove invalid vectors" step is PostProcessSettings'
    std-dev and residual filters instead, applied exactly once, after the
    engine has already produced its final field -- see that class's
    docstring."""
    filter_method: str = "localmean"
    max_filter_iteration: int = 4
    filter_kernel_size: int = 2

    # Inter-pass smoothing of the displacement field (openpiv.smoothn's own
    # Garcia robust-smoothing routine, applied between passes to the field
    # that gets deformed into the next, finer pass) -- an existing feature
    # that sat unused at a negligible default strength until a real-dataset
    # investigation traced it to be the actual explanation for most of this
    # app's correlation/density gap vs LaVision DaVis on real planar data.
    # DaVis's own real vc7 output carries a MultiPassSmoothingMode=5
    # attribute (decoded from real JobHistory.xml, never previously acted
    # on) -- i.e. DaVis's real production pipeline always applies real
    # inter-pass smoothing, which this app's old smoothn=False default
    # never replicated.
    #
    # Turning it ON was right and stays. The STRENGTH below was not -- see
    # smoothn_p's own comment for the ground-truth re-measurement that
    # replaced 15.0 with 0.75, and for why the original real-data sweep
    # (recorded in DATASET_VALIDATION_REPORT.md's 2026-08-29 section, which
    # reported density 95.41%->98.66% and corr(U) 0.957->0.978 for p=15.0)
    # was measuring agreement with a smoothed reference rather than
    # measurement quality. Read that section's numbers as a record of what
    # was run, not as support for the value it recommended.
    smoothn: bool = True
    # 0.75, NOT 15.0. The 15.0 that used to be here was chosen by a sweep
    # scored against DaVis's own PostProc output -- but that reference is
    # itself heavily post-smoothed (denoisingFilter=3, anisotropic kernel 25,
    # strength 3.5, straight from a real JobHistory.xml), so smoothing this
    # app's field raised agreement with it, and raised "density" too because
    # a flatter field stops tripping the local-median outlier test. Neither
    # is evidence of a better measurement, and the metric rewarded exactly
    # the wrong end of the sweep it ran.
    #
    # Re-measured against KNOWN ground truth (synthetic particle images,
    # displacement prescribed exactly), p=15.0 is 1.5x-4.9x WORSE in RMS
    # error on every field with real spatial structure, and better only on a
    # uniform shift -- which is free to over-smooth by construction:
    #     field                     p=0.75    p=15.0
    #     turbulence k<=6           0.167 px  0.750 px   (4.5x)
    #     turbulence k<=4           0.131 px  0.173 px   (1.3x)
    #     Lamb-Oseen r_c=15         0.035 px  0.056 px   (1.6x)
    #     Lamb-Oseen r_c=30         0.022 px  0.035 px   (1.6x)
    #     uniform 3.37 px           0.021 px  0.021 px   (1.0x)
    # smoothn's `s` enters as a transfer function 1/(1 + s*lambda^2); at
    # s=15 the mid-band gain is ~1/16 and Nyquist ~1/1000, i.e. near-total
    # removal of everything but the largest scales -- then fed forward as the
    # next pass's deformation predictor. The 0.3-1.0 plateau is flat; keep
    # smoothn ON (enabling it WAS right), just not at that strength.
    smoothn_p: float = 0.75

    # CPU-only; no effect on GPU. Restores openpiv's own real per-pass
    # validation (Westerweel & Scarano's "universal outlier detection"
    # local-median test) between each window-deformation pass, rejecting
    # and locally-mean-replacing a spurious vector BEFORE it gets deformed
    # into the next, finer pass -- matching LaVision DaVis's per-pass
    # "multi-pass postprocessing" scheme (its median-based removal
    # factor). ON by default: confirmed via real DaVis-dataset comparison
    # to raise U/V correlation from ~0.6 to ~0.8 and drop spurious-vector
    # rejections from ~30% to ~3%, at negligible extra cost (its
    # generic_filter-based cost is fast-pathed -- see
    # engines/_openpiv_speedups.fast_local_norm_median_val). See
    # engines/cpu_engine.py's CPUPIVProcess for how this is wired in.
    per_pass_validation: bool = True
    per_pass_median_threshold: float = 2.0   # DaVis's default removal factor
    per_pass_median_size: int = 1            # 1 -> 3x3 neighborhood (DaVis's filter length 1)

    # CPU-only; bundled with per_pass_validation (same on/off switch, see
    # above) rather than a separate toggle -- rejects a vector whose
    # correlation peak2mean signal-to-noise ratio falls below this
    # threshold -- a low-confidence/no-real-peak correlation, distinct
    # from the local-median (UOD) test above, which only catches a vector
    # that disagrees with its neighbors. 1.0 is openpiv's own PIVSettings
    # default. Affordable at negligible extra cost because
    # engines/_openpiv_speedups.py's fast correlation path computes this
    # ratio from data it already gathers per window.
    #
    # A REAL ATTEMPT WAS MADE (and reverted) to replace this with DaVis's
    # own real per-pass scheme instead, confirmed via a real JobHistory.xml
    # to pair a peak-RATIO (peak2peak) threshold with an independent
    # minimum-correlation-value floor (useScalarfieldThreshold=true,
    # peakRatioThreshold=1.5, correlationThreshold=0.5) -- see
    # engines._openpiv_speedups.py's sig2noise_method="davis_combined" and
    # _fast_peak2peak_sig2noise (both still present, still bit-exact-
    # verified against openpiv's own peak2peak formula on synthetic data)
    # for the implementation. It is NOT wired in as this app's default:
    # real-dataset verification found peak2peak's `width=2` exclusion zone
    # (openpiv's own default, used unmodified) is too narrow for this
    # app's real, wide correlation peaks at real window sizes -- confirmed
    # 99.16% of ALL first-pass (64px) windows on a real image rejected the
    # peak-ratio test at threshold 1.5, collapsing DaVis-agreement from
    # corr~0.95 to corr~0.36 on real data despite passing every synthetic
    # unit test. See docs/IMPROVEMENT_PLAN.md for the full finding and
    # what would need to change (a wider exclusion zone, and/or deriving
    # the threshold from THIS app's own real peak-ratio distribution
    # rather than copying DaVis's number unmodified) before this could be
    # tried again.
    per_pass_sig2noise_threshold: float = 1.0


@dataclass
class RangeFilterSettings:
    """Config surface for processing.postprocess.range_filter -- universal
    outlier detection (Westerweel & Scarano): rejects a vector whose
    deviation from its local neighbourhood median, NORMALIZED by that
    neighbourhood's median absolute deviation, exceeds residual_max. This
    is the sole "remove if residual..." detection method (no
    magnitude/component range option -- see PostProcessSettings'
    docstring for why).

    residual_max IS A DIMENSIONLESS RATIO, NOT A PIXEL DISTANCE, and that
    changed: range_filter used to threshold the raw px/frame deviation
    while being documented and labelled as universal outlier detection.
    See processing.postprocess.range_filter's docstring for the full
    account and the real-data measurements (the absolute form missed
    11.26% of genuine outliers on a real DaVis-compared pair). The
    default below is 2.0 to match LaVision DaVis's own
    medianUniversalOutlierRemovalFactor for the reference dataset, and it
    is NOT interchangeable with the old default of 3.0: a residual_max
    loaded from a `.pivproj` written before this change means "3 px from
    the local median", not "3x the local MAD". Re-tune against your own
    data rather than assuming the number carries over."""
    enabled: bool = True
    residual_max: Optional[float] = 2.0
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
    """The SOLE source of vector validation (see ValidationSettings'
    docstring -- the engines themselves never reject anything). Two
    detection methods, both ON by default: range_filter (local-window
    universal outlier detection, matching LaVision DaVis's real final-stage
    local median-UOD check almost exactly -- a real recording's
    JobHistory.xml `finalPostProcessingParameter` shows
    medianUniversalOutlierRemovalFactor=2, filterLength=1 -> 3x3, matching
    range_filter's own default below) and global_outlier_std (a
    FIELD-WIDE, not local, std-dev cutoff -- see below for why this one's
    right default turned out to depend on WHERE in the pipeline it runs).

    global_outlier_std's story: DaVis's own real final stage has NO
    field-wide/global check at all (useAllowedVectorRange=false in that
    same JobHistory.xml) -- and on this app's PER-CAMERA planar/stereo 2D
    fields, applying one BEFORE triangulation was measured to reject
    exactly 0 vectors on a real turbulent "Swirl" stereo recording. That
    makes physical sense: a bad 2D correlation still returns SOME plausible
    small displacement within the search window, rarely a true field-wide
    statistical outlier. But `pipeline.process_stereo_pair` (see its own
    docstring) validates the COMBINED/TRIANGULATED field instead, and
    triangulation is exactly where a small per-camera disagreement gets
    AMPLIFIED into a genuinely extreme value -- confirmed on the same real
    data: without a global check, max|velocity| reached ~1200-1900 mm/s
    against DaVis's own real ~320-380 max, surviving even a local (W-aware)
    range_filter pass because a small, internally self-consistent cluster
    of bad vectors looks locally fine to itself. Restoring the ON-by-default
    3.0 threshold at the COMBINED-field stage (where it now runs, for
    stereo) closed that gap: real max|velocity| dropped to physically
    plausible levels in every tested pair, alongside a real density and
    outlier-rate improvement -- see process_stereo_pair's own docstring for
    the exact numbers. replace_invalid/smooth_field are a separate, later
    step (filling gaps in what's left after removal), not a third
    detection method."""
    global_outlier_std: Optional[float] = 3.0   # std-dev spurious-vector filter; None disables (see docstring)
    range_filter: RangeFilterSettings = field(default_factory=RangeFilterSettings)
    # ON by default, alongside range_filter's corrected (normalized) UOD
    # statistic: that correction rejects substantially MORE vectors than
    # the old absolute-distance form did (measured on a real pair: 12.0%
    # vs 1.0%), because it now catches outliers the old form waved
    # through. Left unfilled, those rejections would show up as holes in
    # the field -- LaVision DaVis holds ~98% vector density on the same
    # data while being essentially outlier-free, and it does that by
    # substituting a replacement rather than leaving a gap. Interpolating
    # here is the closest equivalent this pipeline has. Note what that
    # means for downstream use: a filled vector is INTERPOLATED FROM ITS
    # NEIGHBOURS, not measured, so it carries no independent information
    # -- turn this off if an analysis needs strictly measured vectors
    # only (the `valid` mask returned alongside u/v still marks exactly
    # which vectors were filled, either way).
    replace_invalid: bool = True
    smooth_field: bool = False
    smooth_sigma: float = 1.0
    # Drop connected groups of valid vectors smaller than this many vectors
    # (4-connectivity), matching LaVision's "remove groups" final
    # post-processing step -- see processing.postprocess.remove_small_groups.
    # ON by default (threshold 5, DaVis's own default); None disables it.
    # Only applies to a regular (ny, nx) grid, skipped for tiled GPU output
    # same as range_filter/smooth_field.
    remove_small_groups_threshold: Optional[int] = 5

    def for_pipeline(self):
        """A tiny namespace matching what processing.pipeline.process_frames
        expects for its `post` argument (range_filter as a kwargs dict or
        None, not a RangeFilterSettings object)."""
        class _Post:
            pass
        p = _Post()
        p.global_outlier_std = self.global_outlier_std
        p.range_filter = self.range_filter.to_kwargs()
        p.replace_invalid = self.replace_invalid
        p.smooth_field = self.smooth_field
        p.smooth_sigma = self.smooth_sigma
        p.remove_small_groups_threshold = self.remove_small_groups_threshold
        return p


@dataclass
class PreprocessSettings:
    """LaVision-style min/max intensity filter (processing.preprocess.
    min_max_filter), applied to RAW camera frames before any correlation
    -- for stereo, applied per-camera BEFORE dewarping (each camera's own
    raw pixel grid, not the calibrated/mapped one). Removes local
    background intensity level and normalizes local contrast over a
    window of min_max_filter_length pixels (see processing/preprocess.py
    for the exact 5-step formula). Off by default -- unlike the
    validation filters, this changes the input images themselves, so
    it's opt-in."""
    min_max_filter_enabled: bool = False
    min_max_filter_length: int = 5   # L, in pixels


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
    # Which real-world Z (mm) this mapping was fit at -- None for a manually-
    # entered single-plane mapping. Set (alongside StereoSettings' matching
    # `cam*_mapping_plane2`) when a camera has two DaVis-calibrated Z-planes;
    # see calibration.camera_mapping.interpolate_camera_mapping.
    z_mm: Optional[float] = None
    # This camera's real raw sensor size (DaVis Calibration.xml's own
    # OriginalImageSize, sibling to CorrectedImageSize/PixelPerMmFactor in
    # the same CommonParameters block -- see io.davis_set.
    # _exact_camera_mapping_from_calibration_xml) -- a fixed per-camera
    # constant, identical across every plane of that camera, unlike x0/
    # x_span/y0/y_span/dx_coefs/dy_coefs which are per-Z-plane. Used by
    # calibration.camera_mapping.CameraMapping.raw_domain_valid to reject
    # correlation-grid points that world_to_raw maps OUTSIDE this camera's
    # real sensor -- dewarp_image already zero-fills those pixels (map_
    # coordinates' cval=0.0), but a correlation window straddling that
    # zero-padding can still return a spurious "valid-looking" vector that
    # post-processing's statistical filters don't reliably catch (see
    # CameraMapping.raw_domain_valid's docstring). 0 (the default) means
    # "unknown, no masking possible" -- matches this schema's existing
    # 0/None-means-not-available convention (e.g. CalibrationSettings'
    # pixel_pitch_mm/frame_dt_s) -- set for the exact-decode calibration
    # path (which reads it straight off Calibration.xml) but NOT the
    # marks-fit path (_fit_camera_mapping_planes has no OriginalImageSize
    # to read), so raw_domain_valid must treat 0 as "always valid" rather
    # than assuming it's populated.
    raw_width: int = 0
    raw_height: int = 0


@dataclass
class PinholeMappingSettings:
    """One camera's DaVis `PinholeOpenCV` calibration, stored as plain scalars
    so the whole ProjectConfig stays dataclasses.asdict/from_dict round-trippable
    (config.io) -- the live object with the actual projection maths is
    calibration.pinhole.PinholeCameraMapping, built from this by
    calibration.camera_mapping.build_camera_mapping.

    This is the SECOND of DaVis's two internal calibration models. Unlike
    CameraMappingSettings (Polynomial3rdOrder), which stores one coefficient set
    per calibrated Z-plane and interpolates between them, a pinhole camera is
    valid at any Z by construction -- so there is no plane2 counterpart, and
    sheet_z_mm simply selects the plane the mapping is evaluated on.

    f_px is already converted to PIXELS (Calibration.xml stores FocalLengthPixel
    in millimetres despite its name -- see calibration.pinhole's module
    docstring, which documents every decoded convention and its validation).
    """
    f_px: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    # Extrinsics as DaVis stores them: Euler angles (radians, applied Rz@Ry@Rx)
    # and a translation in mm, rather than a pre-multiplied 3x3 -- keeps the
    # serialized form identical to the source file's own parameterization.
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    # Corrected/dewarped canvas <-> world mm, straight off this camera's <Scales>.
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    name: str = ""
    raw_width: int = 0
    raw_height: int = 0
    # DaVis's own declared <FitError RMS> for this camera. Kept because it is a
    # real, checkable invariant: reprojecting the snapshot's own
    # MarkPositionTable.xml must reproduce it (see io.davis_set's staleness
    # check), which is what makes a stale or correction-field-backed snapshot
    # detectable instead of silently wrong.
    fit_rms: Optional[float] = None


@dataclass
class StereoSettings:
    cam0_mapping: CameraMappingSettings = field(default_factory=CameraMappingSettings)
    cam1_mapping: CameraMappingSettings = field(default_factory=CameraMappingSettings)
    # DaVis 'PinholeOpenCV' calibration, when that is what the project has.
    # Mutually exclusive with cam*_mapping/cam*_mapping_plane2 (which carry the
    # 'Polynomial3rdOrder' model): whichever is populated is the one
    # build_camera_mapping uses. None (the default) keeps the polynomial path.
    cam0_pinhole: Optional[PinholeMappingSettings] = None
    cam1_pinhole: Optional[PinholeMappingSettings] = None
    # Second calibrated Z-plane per camera -- None (the default) means a
    # single-plane mapping, unchanged from before this field existed.
    # Populated together with sheet_z_mm when DaVis auto-extraction finds two
    # calibrated planes; calibration.camera_mapping.build_camera_mapping
    # interpolates cam*_mapping/cam*_mapping_plane2 at sheet_z_mm.
    cam0_mapping_plane2: Optional[CameraMappingSettings] = None
    cam1_mapping_plane2: Optional[CameraMappingSettings] = None
    world_shape: Tuple[int, int] = (0, 0)
    world_scale_px_per_mm: float = 1.0
    dewarp_order: int = 1
    # OPTIONAL MANUAL OVERRIDE. Normally leave all four None: the triangulation
    # angles are now derived PER PIXEL and exactly from the calibration itself,
    # by calibration.camera_mapping.stereo_view_angles, at each processing entry
    # point. Setting these forces one single global angle per camera instead,
    # for a rig whose calibration genuinely can't supply the geometry.
    #
    # Why per-pixel replaced a scalar. reconstruct_stereo solves
    # dx1 = dX - dZ*tan(alpha1), and the real viewing angle is not constant: on
    # the reference rig it varies 8.4deg (cam0) / 8.8deg (cam1) across the field
    # of view. The two cameras' errors largely cancel in the DIFFERENCE, so W is
    # barely affected (~0.6%) -- but the in-plane components are not, because
    # nothing cancels there: U is off by ~13% of |W| at the FOV edges and V by
    # ~7% at top/bottom, the latter entirely from beta, whose near-zero MEAN
    # (-0.07deg) hides a +-4.3deg variation. That is why tuning a scalar angle
    # never closed the gap -- the quantity being tuned does not exist.
    #
    # The previous auto-derive (_estimate_stereo_angles, now deleted) returned
    # -34.5deg/+35.0deg where the truth is ~-44.1deg/+45.0deg. It measured
    # parallax on the RAW SENSOR, foreshortened by cos(alpha), and never
    # inverted the mapping: tan(34.53)/tan(44.08) = 0.710 ~= cos(44.1deg). One
    # missing factor, ~10deg of error, and the reason this field used to be a
    # required manual entry at all.
    alpha1_deg: Optional[float] = None
    alpha2_deg: Optional[float] = None
    beta1_deg: Optional[float] = None
    beta2_deg: Optional[float] = None
    # Real Z (mm) of the laser sheet for the CURRENT recording -- an
    # acquisition-time quantity, never derivable from a calibration file.
    # Only meaningful/required when a camera has a second calibrated plane.
    sheet_z_mm: Optional[float] = None


@dataclass
class DualPlanarCameraSettings:
    """One camera's placement within a DaVis "SideBySide2D" project's
    shared combined canvas -- read straight off Calibration.xml's
    RegionWithinCorrectedImage/OriginalImageSize for this camera's own
    CoordinateMapper, NOT fit or computed. region_x/region_y are this
    camera's raw (undewarped) frame's placement origin within the shared
    canvas, in CANVAS pixels (row-down/column-right, matching DaVis's own
    RegionWithinCorrectedImage convention); region_width/region_height is
    the SIZE that raw frame occupies there -- DaVis's own lens-corrected
    footprint, close to but not identical to raw_width/raw_height (e.g. a
    real project: 4144x3041 vs a 4096x3008 raw sensor -- the ~1% gap is
    exactly the lens distortion this feature's flat-scale approach
    approximates rather than fully removing). raw_width/raw_height is
    this camera's own OriginalImageSize, i.e. the actual raw sensor pixel
    grid iter_dual_planar_from_set's frames come in as. The ratio
    region_width/raw_width (and the y equivalent) is the flat per-axis
    scale pipeline.combine_dual_planar_pair uses to place a RAW
    (undewarped) frame's PIV grid onto the shared canvas -- see that
    function's docstring for why a flat scale, not a full per-camera
    polynomial lens dewarp (the same CoordinateMapper data could in
    principle feed calibration.camera_mapping like the stereo path does),
    is this feature's deliberate starting point."""
    region_x: float = 0.0
    region_y: float = 0.0
    region_width: float = 1.0
    region_height: float = 1.0
    raw_width: int = 1
    raw_height: int = 1


@dataclass
class DualPlanarSettings:
    """Calibration for a DaVis "SideBySide2D" dual-camera planar project
    (see ProjectSettings.dual_camera) -- auto-extracted by
    davis_set.read_dual_planar_calibration_from_set, never hand-entered
    (unlike StereoSettings, there's no manual-entry GUI form for this:
    every field here is a direct read off Calibration.xml, nothing fit or
    chosen).

    canvas_width/canvas_height and scale_x_*/scale_y_* describe the
    shared "corrected" canvas both cameras' RegionWithinCorrectedImage
    placements (cam0/cam1 above) are defined against, and the shared
    LinearScaleX/LinearScaleY converting a canvas pixel coordinate
    straight to real-world mm -- both confirmed IDENTICAL across cam0/
    cam1 on real project data (same physical plane, same canvas), so
    read once (off cam0's CoordinateMapper) rather than duplicated
    per-camera. scale_y_mm_per_px is signed (negative on real data,
    DaVis's own "world Y increases upward, canvas row increases
    downward" convention) -- pipeline.combine_dual_planar_pair relies on
    that sign, don't abs() it."""
    enabled: bool = False
    cam0: DualPlanarCameraSettings = field(default_factory=DualPlanarCameraSettings)
    cam1: DualPlanarCameraSettings = field(default_factory=DualPlanarCameraSettings)
    canvas_width: int = 0
    canvas_height: int = 0
    scale_x_mm_per_px: float = 1.0
    scale_x_offset_mm: float = 0.0
    scale_y_mm_per_px: float = 1.0
    scale_y_offset_mm: float = 0.0


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
class PerformanceSettings:
    """Hardware-tuning knobs -- see perf/autotune.py's module docstring
    for the design principle: everything that makes the CPU planar
    pipeline faster is an unconditional patch (engines/_openpiv_speedups.py),
    applied regardless of machine. The ONE thing that legitimately varies
    by hardware and is left to the user is how many pairs to process
    concurrently (Tier 3's ProcessPoolExecutor, wired in
    piv_suite_gui/workers/pipeline_worker.py and cli/main.py's planar
    batch loops) -- everything else (correlation chunk size) is derived
    automatically from available RAM, never user-facing."""
    n_workers: Optional[int] = None   # None = auto (perf.autotune.recommended_workers())


@dataclass
class ProjectConfig:
    """The full, canonical settings tree for one PIV project -- what
    gets saved to/loaded from a `.pivproj` JSON file (config.io)."""
    project: ProjectSettings = field(default_factory=ProjectSettings)
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    correlation: CorrelationSettings = field(default_factory=CorrelationSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    postprocess: PostProcessSettings = field(default_factory=PostProcessSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    stereo: StereoSettings = field(default_factory=StereoSettings)
    dual_planar: DualPlanarSettings = field(default_factory=DualPlanarSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
