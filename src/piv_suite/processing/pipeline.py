"""Per-pair PIV processing pipeline: run an engine on a frame pair, apply
the shared post-processing chain, and (for stereo) combine two cameras'
results into a 3-component field.

`process_frames` is migrated from piv_common.py, generalized to the
engines.base.PIVEngine Protocol instead of assuming a specific backend
(confirmed byte-identical between the GPU and CPU repos already, so this
is a low-risk, mostly-mechanical migration). It's also extended with the
new range/residual filter (processing.postprocess.range_filter),
slotted in just before the existing standard-deviation filter.
"""

import time

import numpy as np

from . import postprocess
from ..calibration.reconstruction import reconstruct_stereo


def process_frames(engine, frame_a, frame_b, post, report_gpu_mem=False, on_gpu_report=None,
                    cancel_check=None):
    """Run one engine (from engines.registry.get_engine_factory(...)) on a
    frame pair and apply the shared post-processing pipeline. Returns
    (u, v, valid, elapsed) in px/frame -- calibration to physical units and
    stereo combination happen in the caller.

    `post` is a PostProcessSettings-like object (duck-typed, not required
    to be config.schema.PostProcessSettings specifically) with attributes:
    global_outlier_std, range_filter (a dict of range_filter() kwargs, or
    None to skip), replace_invalid, smooth_field, smooth_sigma.

    cancel_check, if given, is forwarded straight to the engine call --
    see engines.base.PIVEngine.__call__'s docstring for what each backend
    actually does with it (CPUPIVProcess checks it between multi-pass
    iterations; the GPU backend's non-tiled path ignores it). May raise
    engines.base.EngineCancelled instead of returning, if the engine
    fires it -- callers that pass cancel_check must be prepared to catch
    that (see pipeline_worker.py).

    Post-processing order: validity mask -> range/residual filter ->
    std-dev outlier mask -> NaN-fill -> replace_invalid -> smooth -- same
    position global_outlier_mask already occupied in the original
    pipeline, with the new range/residual filter running just before it.
    Returns reject counts for each filter alongside the field via
    `last_reject_counts` set on this function's return isn't done here
    (kept a pure return tuple) -- see run_planar_pair/run_stereo_pair for
    the CSV-facing reject-count bookkeeping.
    """
    t0 = time.time()
    u, v = engine(frame_a, frame_b, cancel_check=cancel_check)
    if report_gpu_mem and on_gpu_report is not None:
        on_gpu_report()
    elapsed = time.time() - t0

    # `engine.val_locations` is always all-False on both backends (see
    # engines/cpu_engine.py and config/legacy.py's to_gpu_settings) --
    # validation is no longer decided during calculation, so `valid`
    # below is determined ENTIRELY by the filters that follow, driven by
    # `post` (PostProcessSettings). This invariant is load-bearing: don't
    # reintroduce a rejecting engine-side validation without updating it
    # here too.
    valid = ~engine.val_locations

    range_cfg = getattr(post, "range_filter", None)
    n_range_rejected = 0
    if range_cfg:
        range_invalid = postprocess.range_filter(u, v, **range_cfg)
        n_range_rejected = int((range_invalid & valid).sum())
        valid = valid & ~range_invalid

    n_std_rejected = 0
    if post.global_outlier_std is not None:
        std_invalid = postprocess.global_outlier_mask(u, v, post.global_outlier_std)
        n_std_rejected = int((std_invalid & valid).sum())
        valid = valid & ~std_invalid

    n_group_rejected = 0
    group_threshold = getattr(post, "remove_small_groups_threshold", None)
    if group_threshold:
        grouped_valid = postprocess.remove_small_groups(valid, group_threshold)
        n_group_rejected = int((valid & ~grouped_valid).sum())
        valid = grouped_valid

    u_out, v_out = u.copy(), v.copy()
    u_out[~valid] = np.nan
    v_out[~valid] = np.nan

    if post.replace_invalid:
        x, y = engine.coords
        u_out, v_out = postprocess.replace_invalid_vectors(x, y, u_out, v_out, valid)

    if post.smooth_field:
        u_out, v_out = postprocess.smooth_vector_field(u_out, v_out, post.smooth_sigma)

    reject_counts = {
        "range_residual": n_range_rejected,
        "std_dev": n_std_rejected,
        "small_groups": n_group_rejected,
    }
    return u_out, v_out, valid, elapsed, reject_counts


def process_frames_tiled(frame_a, frame_b, post, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
                          report_gpu_mem=False, free_pools_fn=None, verbose=False, cancel_check=None):
    """Tiled counterpart to process_frames() -- runs the GPU engine tile
    by tile (engines.gpu_engine.run_tiled) to bound peak GPU memory on
    very large frames, then applies the same post-processing chain.
    range_filter and smooth_field are intentionally skipped (with a
    warning) since tiled output is an unstructured point set, not a
    regular (ny, nx) grid -- range_filter's local-window median needs a
    grid to define "local," and smooth_field's Gaussian blur does too.
    global_outlier_mask has no such requirement (field-wide mean/std over
    a flat array works the same regardless of shape), so it still runs.

    cancel_check, if given, is forwarded to run_tiled(), which polls it
    once per tile -- see run_tiled's docstring. May raise
    engines.base.EngineCancelled instead of returning."""
    from ..engines.gpu_engine import run_tiled

    class _Ctrl:  # minimal shim -- run_tiled only reads .verbose
        pass
    _ctrl = _Ctrl()
    _ctrl.verbose = verbose

    x, y, u, v, valid_raw, elapsed = run_tiled(
        frame_a, frame_b, _ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
        report_gpu_mem=report_gpu_mem, free_pools_fn=free_pools_fn, cancel_check=cancel_check,
    )

    valid = valid_raw
    range_cfg = getattr(post, "range_filter", None)
    n_range_rejected = 0
    if range_cfg:
        print("[warn] range_filter (universal outlier detection) is "
              "ignored for tiled output -- its local-window median needs "
              "a regular (ny, nx) grid, and tiled results are an "
              "unstructured point set stitched from multiple tiles' own "
              "local grids instead")

    n_std_rejected = 0
    if post.global_outlier_std is not None:
        std_invalid = postprocess.global_outlier_mask(u, v, post.global_outlier_std)
        n_std_rejected = int((std_invalid & valid).sum())
        valid = valid & ~std_invalid

    if getattr(post, "remove_small_groups_threshold", None):
        print("[warn] remove_small_groups_threshold is ignored for tiled "
              "output -- it needs a regular (ny, nx) grid to define "
              "connectivity, and tiled results are an unstructured point "
              "set stitched from multiple tiles' own local grids instead")

    u_out, v_out = u.copy(), v.copy()
    u_out[~valid] = np.nan
    v_out[~valid] = np.nan

    if post.replace_invalid:
        u_out, v_out = postprocess.replace_invalid_vectors(x, y, u_out, v_out, valid)

    if post.smooth_field:
        print("[warn] smooth_field is ignored for tiled output -- Gaussian "
              "smoothing needs a regular (ny, nx) grid, and tiled results "
              "are an unstructured point set stitched from multiple tiles' "
              "own local grids instead")

    reject_counts = {"range_residual": n_range_rejected, "std_dev": n_std_rejected, "small_groups": 0}
    return x, y, u_out, v_out, valid, elapsed, reject_counts


def combine_dual_planar_pair(cam0, cam1, dual_planar, frame_dt_s=None):
    """Stitch two coplanar cameras' independently-processed planar PIV
    fields into one combined field on DaVis's own shared canvas (see
    config.schema.DualPlanarSettings) -- the dual-camera-planar
    counterpart to combine_stereo_pair, but no triangulation involved:
    both cameras see the SAME flat plane, just an offset (overlapping)
    region of it, so this only needs to PLACE each camera's field in the
    right spot and average the overlap, not solve for a 3rd velocity
    component.

    cam0/cam1 are (u, v, x, y, valid) 5-tuples straight from
    processing.pipeline.process_frames (u/v the postprocessed px/frame
    field, valid its mask) plus that camera's coordinate grid -- x, y
    MUST be that camera's own RAW (undewarped) sensor pixel grid in
    ROW-DOWN convention, i.e. engine.coords, NOT the display-flipped
    (x, y) engines.registry's factory/cpu_engine.init_cpu_processor
    hands back for plotting. Using the flipped version here would place
    cam0/cam1 upside-down relative to RegionWithinCorrectedImage's own
    row-down convention (and DaVis's LinearScaleY, whose NEGATIVE slope
    already encodes "canvas row increases downward, world Y increases
    upward" -- see DualPlanarSettings' docstring).

    Placement is a flat per-axis affine scale (region_width/raw_width,
    region_height/raw_height -- see DualPlanarCameraSettings' docstring),
    not a full per-camera polynomial lens dewarp: this feature's
    deliberate starting point (see davis_set.read_dual_planar_
    calibration_from_set's docstring) -- validate against real DaVis
    output (a .vc7 from the same recording) before trusting it near the
    overlap seam, where lens distortion is largest. Velocity is scaled by
    the SAME per-axis factor a position displacement would be (a
    displacement of N raw pixels covers N*scale canvas-mm, exactly like a
    static position does), then by frame_dt_s to reach m/s -- generalizing
    postprocess.apply_calibration's isotropic scalar pixel_pitch_mm to
    this feature's real (anisotropic-SIGN) per-axis mm/px scale. Unlike
    apply_calibration, frame_dt_s=None does NOT skip the position/mm
    conversion (that conversion is mandatory just to place two different
    cameras' pixel grids in one shared coordinate system, independent of
    time) -- it only leaves velocity in mm/frame instead of m/s.

    The two cameras' own PIV grids don't line up cell-for-cell on the
    shared canvas (their region_width/raw_width ratios differ slightly --
    confirmed on real data: 1.012 vs 1.004, since DaVis's own lens
    correction stretches each camera's footprint by a slightly different
    amount), so both are resampled (scipy.interpolate.griddata, linear)
    onto ONE shared regular mm grid before combining. Only ORIGINALLY-
    VALID points (per camera's own `valid` mask) are used as griddata's
    source scatter -- an already-interpolated (replace_invalid-filled)
    cell carries no independent information, so letting it feed this
    interpolation too would double-smooth rather than genuinely combine
    two cameras' measurements. The overlap strip (both cameras' resampled
    footprints covering the same mm range) is averaged (nanmean) rather
    than either camera arbitrarily overwriting the other.

    Returns (X_mm, Y_mm, U, V, valid) on one shared regular grid -- valid
    True wherever EITHER camera's resampled footprint reaches that cell."""
    from scipy.interpolate import griddata

    layers = []
    for (u, v, x, y, valid), cam in ((cam0, dual_planar.cam0), (cam1, dual_planar.cam1)):
        scale_x = cam.region_width / cam.raw_width
        scale_y = cam.region_height / cam.raw_height
        canvas_x = cam.region_x + x * scale_x
        canvas_y = cam.region_y + y * scale_y
        x_mm = canvas_x * dual_planar.scale_x_mm_per_px + dual_planar.scale_x_offset_mm
        y_mm = canvas_y * dual_planar.scale_y_mm_per_px + dual_planar.scale_y_offset_mm

        u_mm = u * scale_x * dual_planar.scale_x_mm_per_px
        v_mm = v * scale_y * dual_planar.scale_y_mm_per_px
        if frame_dt_s is not None:
            u_mm, v_mm = u_mm / frame_dt_s / 1000.0, v_mm / frame_dt_s / 1000.0
        layers.append((x_mm, y_mm, u_mm, v_mm, np.asarray(valid)))

    # Shared output grid: spans the union of both cameras' placed extent,
    # at cam0's own median mm spacing (both cameras use the same PIV
    # window/overlap on the same-sized raw sensor, so their native
    # spacings are already close -- this doesn't need to be exact, just a
    # reasonable common resolution to resample both onto).
    x0_mm, y0_mm = layers[0][0], layers[0][1]
    step_x_mm = abs(float(np.median(np.diff(np.sort(np.unique(x0_mm[0, :]))))))
    step_y_mm = abs(float(np.median(np.diff(np.sort(np.unique(y0_mm[:, 0]))))))

    all_x = np.concatenate([lay[0].ravel() for lay in layers])
    all_y = np.concatenate([lay[1].ravel() for lay in layers])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    nx = max(2, int(round((x_max - x_min) / step_x_mm)) + 1)
    ny = max(2, int(round((y_max - y_min) / step_y_mm)) + 1)
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xi, yi)

    u_layers, v_layers = [], []
    for x_mm, y_mm, u_mm, v_mm, valid in layers:
        pts = np.column_stack([x_mm[valid].ravel(), y_mm[valid].ravel()])
        u_layers.append(griddata(pts, u_mm[valid].ravel(), (X, Y), method="linear"))
        v_layers.append(griddata(pts, v_mm[valid].ravel(), (X, Y), method="linear"))

    with np.errstate(invalid="ignore"):
        U = np.nanmean(np.stack(u_layers), axis=0)
        V = np.nanmean(np.stack(v_layers), axis=0)
    valid_out = ~np.isnan(U)
    return X, Y, U, V, valid_out


def combine_stereo_pair(u1, v1, u2, v2, angles, world_scale_px_per_mm, frame_dt_s=None):
    """Combine two cameras' per-camera (u, v) fields (already dewarped
    onto a shared world grid, in px/frame, at world_scale_px_per_mm
    px/mm) into 3-component (U, V, W), matching Stereo-PIV.py's
    handle_pair(). angles = (alpha1, alpha2, beta1, beta2) in radians.

    frame_dt_s=None (no real time base available) leaves the result in
    mm/frame, matching apply_calibration's/combine_dual_planar_pair's own
    "no dt = stay in native displacement units, don't silently unit-shift"
    convention. When frame_dt_s IS given, the result is m/s -- dividing
    by frame_dt_s alone (mm/frame / s) only reaches mm/s, an extra /1000
    is required to reach m/s, same as combine_dual_planar_pair already
    does. THIS WAS MISSING (a real bug, not by design): every stereo
    U/V/W this app has ever produced with frame_dt_s set -- CLI, GUI Run,
    and Preview alike, all three call this same function -- was silently
    1000x too large (mm/s reported as if it were m/s), never caught
    because no test asserted absolute physical units, only relative
    reconstruction accuracy against a synthetic displacement (see
    test_stereo_pipeline.py, which uses frame_dt_s=None and so never
    exercised this branch at all)."""
    alpha1, alpha2, beta1, beta2 = angles
    u1_mm, v1_mm, u2_mm, v2_mm = (a / world_scale_px_per_mm for a in (u1, v1, u2, v2))
    U, V, W = reconstruct_stereo(u1_mm, v1_mm, u2_mm, v2_mm, alpha1, alpha2, beta1, beta2)
    if frame_dt_s is not None:
        U, V, W = (a / frame_dt_s / 1000.0 for a in (U, V, W))
    return U, V, W
