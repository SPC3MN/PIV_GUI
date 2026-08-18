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


def range_filter(u, v, u_range=None, v_range=None, magnitude_range=None,
                  residual_max=None, neighborhood_size=3):
    """Reject vectors that fall outside a configured displacement range,
    and/or whose residual from their LOCAL neighborhood median exceeds a
    threshold ("remove residuals above a certain range").

    This is complementary to global_outlier_mask: that one rejects vectors
    far from the FIELD-WIDE mean (a single global threshold); this one
    rejects vectors either outside a fixed, physically-motivated range
    (e.g. "no vector should exceed 20 px/frame in this experiment") or far
    from their immediate spatial neighbors (catches spatially-localized
    spurious vectors -- a spike surrounded by consistent neighbors -- that
    a global mean/std check can miss in a large field).

    Parameters
    ----------
    u, v : ndarray, same shape (regular (ny, nx) grid expected for the
        residual check; range checks work on any shape).
    u_range, v_range : (min, max) or None -- absolute component bounds.
    magnitude_range : (min, max) or None -- bounds on sqrt(u**2 + v**2).
    residual_max : float or None -- if set, rejects vectors whose distance
        from the local neighborhood median (magnitude of
        (u - median_u, v - median_v)) exceeds this value. Requires u, v to
        be a regular 2-D grid.
    neighborhood_size : odd int, the local median filter's window size
        (in vectors, not pixels) for the residual check.

    Returns a bool array (same shape as u), True = rejected.
    """
    invalid = np.zeros_like(u, dtype=bool)

    if u_range is not None:
        lo, hi = u_range
        invalid |= (u < lo) | (u > hi)
    if v_range is not None:
        lo, hi = v_range
        invalid |= (v < lo) | (v > hi)
    if magnitude_range is not None:
        lo, hi = magnitude_range
        mag = np.hypot(u, v)
        invalid |= (mag < lo) | (mag > hi)

    if residual_max is not None:
        if u.ndim != 2:
            raise ValueError(
                "residual_max requires u, v to be a regular 2-D (ny, nx) "
                "grid -- not applicable to flat/tiled output"
            )
        med_u = _nan_median_filter(u, neighborhood_size)
        med_v = _nan_median_filter(v, neighborhood_size)
        residual = np.hypot(u - med_u, v - med_v)
        invalid |= residual > residual_max

    return invalid


def _nan_median_filter(a, size):
    """NaN-aware local median filter -- scipy.ndimage.median_filter isn't
    NaN-aware, so this uses generic_filter with nanmedian instead (slower,
    but correct on fields that already have NaN gaps from prior validation
    steps)."""
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
