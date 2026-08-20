"""Displacement post-processing: outlier/spurious-vector rejection,
range/residual filtering, invalid-vector interpolation, smoothing, and
calibration to physical units.

`global_outlier_mask`, `replace_invalid_vectors`, `smooth_vector_field`,
and `apply_calibration` are migrated unchanged from piv_common.py
(identical across all four source repos). `range_filter` is new -- a
displacement-range/local-residual rejection filter, complementary to the
existing standard-deviation-based `global_outlier_mask`.
"""

import numpy as np


def global_outlier_mask(u, v, n_std):
    """Reject vectors whose u or v component is more than n_std standard
    deviations from the FIELD-WIDE mean (a single global threshold, not a
    local/windowed one) -- the standard-deviation-based spurious-vector
    filter. Returns a bool array, True = rejected."""
    if n_std is None:
        return np.zeros_like(u, dtype=bool)
    u_mean, u_std = np.nanmean(u), np.nanstd(u)
    v_mean, v_std = np.nanmean(v), np.nanstd(v)
    return (np.abs(u - u_mean) > n_std * u_std) | (np.abs(v - v_mean) > n_std * v_std)


def range_filter(u, v, residual_max=None, window_size=3):
    """Reject vectors whose residual from their LOCAL window median
    exceeds a threshold ("remove if residual...").

    This is complementary to global_outlier_mask: that one rejects vectors
    far from the FIELD-WIDE mean (a single global threshold); this one
    rejects vectors far from their immediate spatial neighbors, catching
    spatially-localized spurious vectors -- a spike surrounded by
    consistent neighbors -- that a global mean/std check can miss in a
    large field.

    Parameters
    ----------
    u, v : ndarray, same shape, a regular (ny, nx) grid (not applicable to
        flat/tiled output).
    residual_max : float or None -- if set, rejects vectors whose distance
        from the local window median (magnitude of (u - median_u,
        v - median_v)) exceeds this value. None disables the filter
        (returns an all-False mask).
    window_size : odd int, the local median filter's window size (in
        vectors, not pixels).

    Returns a bool array (same shape as u), True = rejected.
    """
    if residual_max is None:
        return np.zeros_like(u, dtype=bool)
    if u.ndim != 2:
        raise ValueError(
            "range_filter requires u, v to be a regular 2-D (ny, nx) "
            "grid -- not applicable to flat/tiled output"
        )
    med_u = _nan_median_filter(u, window_size)
    med_v = _nan_median_filter(v, window_size)
    residual = np.hypot(u - med_u, v - med_v)
    return residual > residual_max


def _nan_median_filter(a, size):
    """Local median filter, NaN-aware only when needed. In the current
    architecture u/v reaching range_filter are always NaN-free (the
    engine fills residual NaN before returning -- see cpu_engine.py's
    module docstring), so the compiled scipy.ndimage.median_filter (not
    NaN-aware, but exact and orders of magnitude faster than a per-pixel
    Python callback -- measured ~6.7s vs well under 1s on a ~185k-cell
    finest-pass grid) is used whenever there's nothing for it to trip on.
    Falls back to the slower NaN-aware generic_filter(nanmedian) path if
    NaN is actually present, so this stays correct even if a future
    caller doesn't guarantee NaN-free input."""
    if not np.isnan(a).any():
        from scipy.ndimage import median_filter
        return median_filter(a, size=size, mode="nearest")
    from scipy.ndimage import generic_filter
    return generic_filter(a, np.nanmedian, size=size, mode="nearest")


def replace_invalid_vectors(x, y, u, v, valid_mask):
    from scipy.interpolate import griddata
    invalid = ~valid_mask
    if not invalid.any():
        return u, v
    if not valid_mask.any():
        print("[warn] replace_invalid_vectors: every vector was rejected -- "
              "nothing left to interpolate from, so invalid vectors stay "
              "NaN. Check your validation/outlier/range-filter thresholds "
              "against this data.")
        return u, v
    pts_valid = np.column_stack([x[valid_mask], y[valid_mask]])
    u_out, v_out = u.copy(), v.copy()
    u_out[invalid] = griddata(pts_valid, u[valid_mask], (x[invalid], y[invalid]), method="linear")
    v_out[invalid] = griddata(pts_valid, v[valid_mask], (x[invalid], y[invalid]), method="linear")
    still_bad = np.isnan(u_out)
    if still_bad.any():
        u_out[still_bad] = griddata(pts_valid, u[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
        v_out[still_bad] = griddata(pts_valid, v[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
    return u_out, v_out


def smooth_vector_field(u, v, sigma):
    from scipy.ndimage import gaussian_filter
    mask = (~np.isnan(u)).astype(float)
    u0, v0 = np.nan_to_num(u), np.nan_to_num(v)
    wsum = np.clip(gaussian_filter(mask, sigma), 1e-8, None)
    return gaussian_filter(u0, sigma) / wsum, gaussian_filter(v0, sigma) / wsum


def apply_calibration(u, v, pixel_pitch_mm=None, frame_dt_s=None):
    """Planar px/frame -> physical units, if pixel_pitch_mm and frame_dt_s
    are both given; otherwise a no-op (stays px/frame)."""
    if pixel_pitch_mm is None or frame_dt_s is None:
        return u, v
    scale = (pixel_pitch_mm / 1000.0) / frame_dt_s
    return u * scale, v * scale
