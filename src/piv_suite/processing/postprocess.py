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


def global_outlier_mask(u, v, n_std, w=None):
    """Reject vectors whose u, v, or (if given) w component is more than
    n_std standard deviations from the FIELD-WIDE mean (a single global
    threshold, not a local/windowed one) -- the standard-deviation-based
    spurious-vector filter. Returns a bool array, True = rejected.

    `w` exists for the stereo combined-field case (see
    pipeline.process_stereo_pair): a local-neighbourhood check (range_filter
    below) can never catch a small, internally-self-consistent CLUSTER of
    bad vectors -- by construction they look locally fine to each other.
    Confirmed on real data: after a local-only UOD pass on a triangulated
    stereo field, max|velocity| was still ~1479 mm/s against a real
    DaVis comparison's own ~323 max -- a physically implausible survivor
    that only a field-wide check catches. Planar/dual-planar callers
    never pass w -- the two-component form is unchanged."""
    if n_std is None:
        return np.zeros_like(u, dtype=bool)
    u_mean, u_std = np.nanmean(u), np.nanstd(u)
    v_mean, v_std = np.nanmean(v), np.nanstd(v)
    invalid = (np.abs(u - u_mean) > n_std * u_std) | (np.abs(v - v_mean) > n_std * v_std)
    if w is not None:
        w_mean, w_std = np.nanmean(w), np.nanstd(w)
        invalid = invalid | (np.abs(w - w_mean) > n_std * w_std)
    return invalid


UOD_EPS_PX = 0.1


def _valid_neighbour_count(valid, window_size):
    """How many non-NaN neighbours each cell has, excluding itself."""
    from scipy.ndimage import uniform_filter
    k = int(window_size)
    total = uniform_filter(valid.astype(np.float64), size=k, mode="constant", cval=0.0) * k * k
    return np.rint(total).astype(int) - valid.astype(int)


def range_filter(u, v, residual_max=None, window_size=3, eps=UOD_EPS_PX, w=None,
                 insertion_max=None, min_neighbours=None):
    """Universal outlier detection (Westerweel & Scarano 2005): reject a
    vector whose deviation from its local neighbourhood median, NORMALIZED
    by that neighbourhood's median absolute deviation, exceeds
    residual_max.

    Complementary to global_outlier_mask: that one rejects vectors far
    from the FIELD-WIDE mean (a single global threshold); this one rejects
    vectors far from their immediate spatial neighbours, catching
    spatially-localized spurious vectors -- a spike surrounded by
    consistent neighbours -- that a global mean/std check can miss in a
    large field.

    THE NORMALIZATION IS THE WHOLE POINT, and this function did not always
    have it. It previously thresholded the RAW deviation
    `hypot(u - med_u, v - med_v) > residual_max`, i.e. an absolute
    distance in px/frame, while being labelled "Universal outlier
    detection" in the GUI, documented as universal outlier detection in
    config.schema.PostProcessSettings, and configured against DaVis's
    `medianUniversalOutlierRemovalFactor` (a dimensionless ratio) by
    scripts/compare_davis_lavision.py. Those are different quantities and
    the mismatch was measurable on real data: against a real DaVis export
    of the same frame pair, the absolute form flagged 1.05% of vectors
    while true UOD at the same threshold flagged 12.04% -- it MISSED
    11.26% of genuine outliers and rejected 0.27% of good vectors. The
    reason is that an absolute threshold cannot be right everywhere at
    once in a field with varying local dynamics: in a calm region (small
    MAD) a 1.5 px spike is a blatant outlier that an absolute 2.0 px
    threshold waves through, while in a high-shear region (large MAD) a
    perfectly legitimate 2.5 px difference from neighbours gets cut. The
    normalized statistic is scale-free precisely so one threshold can
    apply across the whole field.

    THRESHOLD SEMANTICS CHANGED WITH THAT FIX. residual_max is now a
    dimensionless ratio (DaVis's own removal factor is 2.0, which is this
    schema's new default), NOT a pixel distance. A residual_max carried
    over from an older `.pivproj` file means something different than it
    used to -- 3.0 formerly meant "3 px from the local median", now it
    means "3x the local MAD". Re-tune it against your own data rather
    than assuming the old number transfers.

    `eps` is an absolute floor added to the MAD, in PIXELS (the
    conventional W&S value is 0.1 px, representing expected measurement
    noise). It stops a locally-uniform neighbourhood, whose MAD is
    near zero, from turning a negligible deviation into an enormous
    ratio. Because eps is absolute, u/v must be in px/frame here --
    which they are: pipeline.process_frames runs this filter BEFORE
    apply_calibration converts to physical units.

    Parameters
    ----------
    u, v : ndarray, same shape, a regular (ny, nx) grid (not applicable to
        flat/tiled output), in px/frame.
    residual_max : float or None -- normalized-residual threshold above
        which a vector is rejected. None disables the filter (returns an
        all-False mask).
    window_size : odd int, the local neighbourhood's size (in vectors, not
        pixels). The vector being tested is EXCLUDED from its own
        neighbourhood statistics (per W&S) so a strong outlier cannot drag
        the median it is being judged against toward itself.
    w : optional 3rd component (same shape as u, v). Passed by
        pipeline.process_stereo_pair to judge a triangulated stereo
        vector's local consistency in all 3 physical components at once,
        not just its in-plane U,V -- see normalized_median_residual's own
        docstring for why W matters here specifically. Planar/dual-planar
        callers never pass this; the 2-component form is unchanged.

    insertion_max : float or None -- DaVis's "remove and iteratively
        replace" second threshold (its medianUniversalOutlierInsertion
        Factor, default 3, against a removal factor of 2). When given, a
        rejected vector is RE-INSERTED if, once the other rejects are taken
        out of its neighbourhood, its residual falls below this. Iterated to
        convergence. None keeps the plain single-shot removal.

        This is a genuine density mechanism, not a leniency knob. A good
        vector sitting next to a cluster of bad ones is judged against a
        neighbourhood median those bad ones have dragged around; removing
        them first and re-testing is what tells the two cases apart. The
        threshold is deliberately LOOSER than residual_max, because a
        vector that has already survived one round of cleaning has more
        evidence in its favour than one being judged for the first time.
    min_neighbours : int or None -- refuse to reject a vector with fewer
        than this many valid neighbours (DaVis's medianFilterMinNoNeighbours,
        default 3). A vector at the edge of the field or beside a large hole
        is otherwise judged on almost no evidence, and rejecting it there
        erodes the field's border a little more on every pass.

    Returns a bool array (same shape as u), True = rejected.
    """
    if residual_max is None:
        return np.zeros_like(u, dtype=bool)
    if u.ndim != 2:
        raise ValueError(
            "range_filter requires u, v to be a regular 2-D (ny, nx) "
            "grid -- not applicable to flat/tiled output"
        )
    residual = normalized_median_residual(u, v, window_size=window_size, eps=eps, w=w)
    rejected = residual > residual_max

    present = np.isfinite(u)
    if min_neighbours:
        rejected &= _valid_neighbour_count(present, window_size) >= min_neighbours
    if not insertion_max or not rejected.any():
        return rejected

    # Iterate to convergence. The rejected set only ever shrinks (a vector is
    # re-inserted, never re-rejected, because each pass judges against a
    # neighbourhood that has only gained members), so this terminates; the
    # cap is belt-and-braces against a pathological float case.
    for _ in range(10):
        kept = present & ~rejected
        # The rejected cell keeps ITS OWN value (it is what's being judged);
        # only its NEIGHBOURS are cleaned. NaN residual means it has no
        # surviving neighbours at all -- no evidence to re-admit it on, so it
        # stays out.
        resid = normalized_median_residual(u, v, window_size=window_size, eps=eps,
                                           w=w, contributing=kept)
        reinstate = rejected & np.isfinite(resid) & (resid <= insertion_max)
        if min_neighbours:
            reinstate &= _valid_neighbour_count(kept, window_size) >= min_neighbours
        if not reinstate.any():
            break
        rejected = rejected & ~reinstate
    return rejected


def normalized_median_residual(u, v, window_size=3, eps=UOD_EPS_PX, w=None,
                                contributing=None, _max_bytes=256 * 1024**2):
    """The Westerweel & Scarano universal-outlier-detection statistic per
    vector: sqrt(r_u^2 + r_v^2) (or sqrt(r_u^2 + r_v^2 + r_w^2) when `w` is
    given), where r = |value - neighbourhood median| / (neighbourhood MAD +
    eps), with the vector itself excluded from its own neighbourhood.
    Exposed separately from range_filter so the same quantity can be
    reported/plotted (see scripts/compare_velocity_fields.py) without
    duplicating the formula.

    `w`: the stereo out-of-plane component (see
    pipeline.process_stereo_pair). W is the component most sensitive to
    inter-camera disagreement in a triangulated stereo vector, so folding
    it into the same local-neighbourhood statistic catches a bad
    triangulation that a U,V-only local check would miss even though the
    underlying 2D camera measurements individually looked locally fine.
    A LOCAL check like this one still can't catch a small, internally
    self-consistent CLUSTER of bad vectors, though -- by construction
    they look locally fine to their own (also bad) neighbours -- see
    global_outlier_mask's own docstring for the companion field-wide
    check that exists specifically for that residual case.

    Neighbourhoods are built with a NaN-padded sliding-window view rather
    than scipy.ndimage.median_filter, because the MAD term needs, for each
    cell, the median of |neighbour - THAT CELL's median| -- each neighbour
    measured against the centre cell's median, not against its own. A
    plain median_filter over a precomputed |a - median| array would
    silently compute the latter. NaN neighbours (already-rejected or
    absent vectors) drop out of both medians instead of counting as
    zeros, so a vector beside a hole is not penalized for the hole.

    Processed in row chunks bounded by _max_bytes: the window array is
    (n_cells, window_size**2) floats, which at the maximum window size the
    GUI offers (21 -> 441 per cell) would be ~650MB on a full finest-pass
    grid if built in one piece. Chunking keeps peak memory flat regardless
    of grid size and window size, at no numerical cost -- each cell's
    statistics depend only on its own neighbourhood, so chunk boundaries
    cannot change any result.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings

    k = int(window_size)
    if k % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    pad = k // 2
    centre = k * k // 2

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ny, nx = u.shape
    if w is not None:
        w = np.asarray(w, dtype=np.float64)

    bytes_per_row = nx * k * k * 8 * (3 if w is not None else 2)
    rows_per_chunk = max(1, int(_max_bytes // max(1, bytes_per_row)))

    # `contributing` separates the two roles this array plays. The centre
    # value being JUDGED always comes from u/v/w; which cells are allowed to
    # take part in a neighbourhood is `contributing`. They coincide unless a
    # caller is re-testing already-rejected vectors against a cleaned
    # neighbourhood (see range_filter's insertion_max), where the rejected
    # cell must still be judged even though it may no longer vote.
    if contributing is None:
        u_src, v_src, w_src = u, v, w
    else:
        u_src = np.where(contributing, u, np.nan)
        v_src = np.where(contributing, v, np.nan)
        w_src = np.where(contributing, w, np.nan) if w is not None else None

    u_pad = np.pad(u_src, pad, mode="constant", constant_values=np.nan)
    v_pad = np.pad(v_src, pad, mode="constant", constant_values=np.nan)
    w_pad = np.pad(w_src, pad, mode="constant", constant_values=np.nan) if w is not None else None
    out = np.empty((ny, nx), dtype=np.float64)

    with warnings.catch_warnings():
        # A cell whose entire neighbourhood is NaN legitimately has no
        # median -- it comes out NaN (never > threshold, so never
        # rejected on no evidence). Same warning the equivalent
        # generic_filter(nanmedian) path already emits.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")

        for start in range(0, ny, rows_per_chunk):
            end = min(start + rows_per_chunk, ny)
            wu = sliding_window_view(u_pad[start:end + 2 * pad], (k, k)).reshape(end - start, nx, k * k).copy()
            wv = sliding_window_view(v_pad[start:end + 2 * pad], (k, k)).reshape(end - start, nx, k * k).copy()
            wu[..., centre] = np.nan   # exclude each vector from judging itself
            wv[..., centre] = np.nan

            med_u = np.nanmedian(wu, axis=-1)
            med_v = np.nanmedian(wv, axis=-1)
            mad_u = np.nanmedian(np.abs(wu - med_u[..., None]), axis=-1)
            mad_v = np.nanmedian(np.abs(wv - med_v[..., None]), axis=-1)

            if w_pad is not None:
                w_win = sliding_window_view(w_pad[start:end + 2 * pad], (k, k)).reshape(end - start, nx, k * k).copy()
                w_win[..., centre] = np.nan
                med_w = np.nanmedian(w_win, axis=-1)
                mad_w = np.nanmedian(np.abs(w_win - med_w[..., None]), axis=-1)

            with np.errstate(invalid="ignore", divide="ignore"):
                r_u = np.abs(u[start:end] - med_u) / (mad_u + eps)
                r_v = np.abs(v[start:end] - med_v) / (mad_v + eps)
                if w_pad is not None:
                    r_w = np.abs(w[start:end] - med_w) / (mad_w + eps)
                total_sq = r_u ** 2 + r_v ** 2
                if w_pad is not None:
                    total_sq = total_sq + r_w ** 2
                out[start:end] = np.sqrt(total_sq)

    return out


def remove_small_groups(valid_mask, min_group_size):
    """Reject vectors belonging to a connected group of valid vectors
    smaller than min_group_size (4-connectivity), matching LaVision's
    "remove groups" final post-processing step -- catches small isolated
    islands of spuriously-agreeing vectors that neither the field-wide
    std-dev filter nor the local-median residual filter reject (each
    vector in a small cluster can look locally consistent with its few
    neighbors while still being wrong). Returns a new bool mask (True =
    still valid); does not mutate the input."""
    from scipy.ndimage import label
    if min_group_size is None or min_group_size <= 1:
        return valid_mask
    labels, n = label(valid_mask)
    if n == 0:
        return valid_mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_group_size
    keep[0] = False  # background label
    return keep[labels]


def replace_invalid_vectors(x, y, u, v, valid_mask, w=None):
    """Interpolate invalid (valid_mask False) cells of u, v from the
    valid cells around them (scipy.interpolate.griddata, linear, falling
    back to nearest for any cell the linear pass couldn't reach). w:
    optional 3rd component (see pipeline.process_stereo_pair) -- filled
    the same way, from its own values at the valid cells. Returns (u, v)
    when w is None (unchanged 2-component contract for planar/dual-planar
    callers), else (u, v, w)."""
    from scipy.interpolate import griddata
    invalid = ~valid_mask
    if not invalid.any():
        return (u, v) if w is None else (u, v, w)
    if not valid_mask.any():
        print("[warn] replace_invalid_vectors: every vector was rejected -- "
              "nothing left to interpolate from, so invalid vectors stay "
              "NaN. Check your validation/outlier/range-filter thresholds "
              "against this data.")
        return (u, v) if w is None else (u, v, w)
    pts_valid = np.column_stack([x[valid_mask], y[valid_mask]])
    u_out, v_out = u.copy(), v.copy()
    u_out[invalid] = griddata(pts_valid, u[valid_mask], (x[invalid], y[invalid]), method="linear")
    v_out[invalid] = griddata(pts_valid, v[valid_mask], (x[invalid], y[invalid]), method="linear")
    still_bad = np.isnan(u_out)
    if still_bad.any():
        u_out[still_bad] = griddata(pts_valid, u[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
        v_out[still_bad] = griddata(pts_valid, v[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
    if w is None:
        return u_out, v_out
    w_out = w.copy()
    w_out[invalid] = griddata(pts_valid, w[valid_mask], (x[invalid], y[invalid]), method="linear")
    still_bad_w = np.isnan(w_out)
    if still_bad_w.any():
        w_out[still_bad_w] = griddata(pts_valid, w[valid_mask], (x[still_bad_w], y[still_bad_w]), method="nearest")
    return u_out, v_out, w_out


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
