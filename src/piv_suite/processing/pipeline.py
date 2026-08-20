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


def process_frames(engine, frame_a, frame_b, post, report_gpu_mem=False, on_gpu_report=None):
    """Run one engine (from engines.registry.get_engine_factory(...)) on a
    frame pair and apply the shared post-processing pipeline. Returns
    (u, v, valid, elapsed) in px/frame -- calibration to physical units and
    stereo combination happen in the caller.

    `post` is a PostProcessSettings-like object (duck-typed, not required
    to be config.schema.PostProcessSettings specifically) with attributes:
    global_outlier_std, range_filter (a dict of range_filter() kwargs, or
    None to skip), replace_invalid, smooth_field, smooth_sigma.

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
    u, v = engine(frame_a, frame_b)
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
                          report_gpu_mem=False, free_pools_fn=None, verbose=False):
    """Tiled counterpart to process_frames() -- runs the GPU engine tile
    by tile (engines.gpu_engine.run_tiled) to bound peak GPU memory on
    very large frames, then applies the same post-processing chain.
    range_filter and smooth_field are intentionally skipped (with a
    warning) since tiled output is an unstructured point set, not a
    regular (ny, nx) grid -- range_filter's local-window median needs a
    grid to define "local," and smooth_field's Gaussian blur does too.
    global_outlier_mask has no such requirement (field-wide mean/std over
    a flat array works the same regardless of shape), so it still runs."""
    from ..engines.gpu_engine import run_tiled

    class _Ctrl:  # minimal shim -- run_tiled only reads .verbose
        pass
    _ctrl = _Ctrl()
    _ctrl.verbose = verbose

    x, y, u, v, valid_raw, elapsed = run_tiled(
        frame_a, frame_b, _ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
        report_gpu_mem=report_gpu_mem, free_pools_fn=free_pools_fn,
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


def combine_stereo_pair(u1, v1, u2, v2, angles, world_scale_px_per_mm, frame_dt_s=None):
    """Combine two cameras' per-camera (u, v) fields (already dewarped
    onto a shared world grid, in px/frame, at world_scale_px_per_mm
    px/mm) into 3-component (U, V, W), matching Stereo-PIV.py's
    handle_pair(). angles = (alpha1, alpha2, beta1, beta2) in radians."""
    alpha1, alpha2, beta1, beta2 = angles
    u1_mm, v1_mm, u2_mm, v2_mm = (a / world_scale_px_per_mm for a in (u1, v1, u2, v2))
    U, V, W = reconstruct_stereo(u1_mm, v1_mm, u2_mm, v2_mm, alpha1, alpha2, beta1, beta2)
    if frame_dt_s is not None:
        U, V, W = (a / frame_dt_s for a in (U, V, W))
    return U, V, W
