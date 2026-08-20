"""Drop-in vectorized replacements for openpiv-python functions used
during CPU-backend PIV calculation. Two kinds of patch live here, and
they are NOT the same kind of change:

FAITHFUL SPEEDUPS (fast_replace_nans, fast_correlation_to_displacement):
each is a faithful re-expression of the SAME formula as the original in
vectorized numpy -- not openpiv's own alternate "vectorized_*" functions
(which were checked and found to differ subtly: a `<= 0` vs `< 0`
gaussian-fallback threshold, and a forced float32 cast, either of which
would silently change results for a science-critical PIV run -- not
acceptable here, hence writing faithful equivalents instead of flipping
`use_vectorized=True`). Confirmed via cProfile against a real
3008x4096 frame pair (see the perf investigation this module came out
of): a single 32px/75%-overlap pass took ~113s, of which ~49% was
`openpiv.lib.replace_nans` and ~35% was `openpiv.pyprocess.
correlation_to_displacement` (a pure-Python loop over every window).

BEHAVIOR-CHANGING PATCH (loose_typical_validation): unlike the two
above, this one deliberately does NOT reproduce openpiv's own
validation.typical_validation. Vector validation has been moved
entirely to processing.postprocess (driven by PostProcessSettings, one
explicit step after the engine returns) -- see cpu_engine.py's module
docstring for the full rationale. openpiv's own multi-pass loop
(windef.multipass_img_deform) still calls `validation.typical_validation`
unconditionally with no settings flag to disable it, so this patch
replaces it with a version that flags a vector invalid ONLY when it is
literal NaN -- never for any global-range/std/median/sig2noise
criterion. This keeps the per-pass replace_outliers call (also
unconditional in openpiv) as a pure numerical-stability mechanism (see
cpu_engine._fill_residual_nan for why residual NaN between passes is
dangerous, not just a validation artifact) without it ever causing a
vector to be counted "invalid" in the final output.

fast_local_median_val exists but is NOT wired into apply_speedups() --
see its own docstring; nothing calls openpiv.validation.local_median_val
any more now that typical_validation itself is replaced wholesale.

apply_speedups() monkeypatches these onto the installed openpiv package
at runtime -- openpiv's own files are never touched on disk, so this is
trivially reversible (don't call apply_speedups()) and survives an
openpiv version upgrade without a vendored fork to maintain. Called once
from cpu_engine.py before any processing happens.

Every function here has a paired regression test in
tests/unit/test_openpiv_speedups.py: the two faithful speedups assert
near-exact numerical equality against the ORIGINAL unpatched openpiv
function on the same inputs (including real-data-derived edge cases:
all-NaN neighborhoods, correlation-map borders, negative correlation
values); loose_typical_validation asserts it flags literal NaN only,
never a finite outlier -- run that suite again after any openpiv
version bump, since the faithful re-implementations assume the current
(0.25.4) algorithms exactly.
"""

import numpy as np


def fast_replace_nans(array, max_iter, tol, kernel_size=2, method="disk"):
    """Faithful vectorized re-implementation of openpiv.lib.replace_nans.

    The original iterates every NaN cell in a Python for-loop, building a
    fresh np.meshgrid + boundary mask on every single cell on every
    iteration -- the dominant CPU cost in a validated PIV pass. The
    replacement performs the exact same Jacobi-style iterative diffusion
    (each iteration's replacement values are computed from the PREVIOUS
    iteration's filled array, matching the original's `win = filled[...]`
    read followed by a single bulk `filled[nan_indices] = replaced_new`
    write only after the full sweep) via one whole-array convolution per
    iteration instead of one Python-level window extraction per NaN cell.

    THE STOPPING RULE IS NOT WHAT THE DOCSTRING/PARAMETER NAME SAYS.
    openpiv.lib.replace_nans has a real bug: `replaced_old = replaced_new`
    (no `.copy()`) aliases the two names to the SAME array object from
    iteration 2 onward, so `replaced_new - replaced_old` in every later
    convergence check is an array minus ITSELF -- exactly 0.0 wherever
    both are real numbers, NaN wherever a cell still has no valid
    neighbour (NaN - NaN = NaN, and `NaN < tol` is always False). So
    `tol` does NOT control anything for any positive value: confirmed
    empirically that tol=1e-3 and tol=1e10 produce byte-identical output
    on real data, while tol=-1 (never satisfiable) produces a genuinely
    different result. The ACTUAL rule, from iteration index 1 onward, is
    "stop as soon as every originally-NaN cell has a non-NaN value this
    iteration" -- a discrete fill/no-fill condition, not a numerical
    threshold, which is exactly what's replicated below (deliberately,
    not "fixed" -- changing this would change every PIV run's output).

    Faithfulness details that matter for exact equivalence, preserved
    here:
    - The SET of cells being replaced is fixed to the array's ORIGINAL
      NaN positions for the entire iteration loop (a cell successfully
      filled in iteration 1 is still recomputed in iteration 2, using its
      own now-filled value as one of its neighbours, exactly as the
      original does -- once a cell has any value it can never become
      unfillable again, since its own value now participates in its
      own kernel window every subsequent iteration, which is also why
      "all filled" is a one-way, monotonic event safe to detect once).
    - The kernel window is centered ON the cell being replaced (kernel
      size 2*kernel_size+1 spans the cell itself) -- scipy.ndimage.
      convolve's default centering matches this for these odd-sized
      kernels.
    - Out-of-array-bounds kernel positions are excluded from BOTH the
      weighted sum and its normalizing weight total (the original's
      `in_mask` filtering) -- exactly reproduced by convolving with
      `mode="constant", cval=0.0` on both the (NaN-zeroed) data and the
      (0/1) validity mask, since a zero contributes nothing to either
      sum.
    - A cell with zero valid weighted neighbours stays NaN for that
      iteration, matching `else: replaced_new[k] = np.nan`.
    - All three kernel shapes (localmean/disk/distance) are radially
      symmetric about their center, so scipy.ndimage.convolve's kernel
      flip (true convolution, not correlation) is a no-op -- confirmed
      by construction, not assumed.
    - The kernel array dtype is `int`, matching the original exactly --
      NOT a style choice: for method="distance" this silently TRUNCATES
      the fractional distance-based weights to integers (another real
      openpiv bug -- e.g. every weight below 1.0 truncates to exactly
      0), and reproducing that truncation exactly is required for
      identical output, not just "close" output.
    """
    if not np.any(np.isnan(array)):
        return array.copy()

    from scipy.ndimage import convolve as _ndimage_convolve

    kernel_size = int(kernel_size)
    filled = np.asarray(array, dtype=np.float64).copy()
    is_original_nan = np.isnan(filled)
    n_nans = int(is_original_nan.sum())

    kernel_int = np.zeros([2 * kernel_size + 1] * filled.ndim, dtype=int)  # dtype=int matches the original -- see docstring
    if method == "localmean":
        kernel_int += 1
    elif method == "disk":
        dist, dist_inv = _get_dist(kernel_int, kernel_size)
        kernel_int[dist <= kernel_size] = 1
    elif method == "distance":
        dist, dist_inv = _get_dist(kernel_int, kernel_size)
        kernel_int[dist <= kernel_size] = dist_inv[dist <= kernel_size]  # truncates to int, matching the original's bug
    else:
        raise ValueError("Known methods are: `localmean`, `disk` or `distance`.")
    kernel = kernel_int.astype(np.float64)

    replaced_old_all_real = np.zeros(n_nans)  # only meaningful for iteration 0's genuine (non-aliased) check

    for it in range(max_iter):
        valid = ~np.isnan(filled)
        data_for_conv = np.where(valid, filled, 0.0)
        numerator = _ndimage_convolve(data_for_conv, kernel, mode="constant", cval=0.0)
        denominator = _ndimage_convolve(valid.astype(np.float64), kernel, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            new_vals = np.where(denominator > 0, numerator / denominator, np.nan)

        replaced_new = new_vals[is_original_nan]
        filled = np.where(is_original_nan, new_vals, filled)

        if it == 0:
            # the one iteration where replaced_old is NOT yet aliased in
            # the original -- a genuine tol comparison against zeros.
            if np.mean((replaced_new - replaced_old_all_real) ** 2) < tol:
                break
        else:
            # every later iteration: replaced_old IS replaced_new in the
            # original (aliased), so the check is exactly "no NaN left".
            if not np.isnan(replaced_new).any():
                break

    return filled.astype(array.dtype, copy=False)


def _get_dist(kernel, kernel_size):
    """Unchanged from openpiv.lib.get_dist -- kept local so this module
    has no import-time dependency on openpiv beyond what's patched."""
    if kernel.ndim == 2:
        xs, ys = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist
        return dist, dist_inv
    if kernel.ndim == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2 + (zs - kernel_size) ** 2)
        dist_inv = np.sqrt(3) * kernel_size - dist
        return dist, dist_inv
    raise ValueError(f"get_dist only supports 2D/3D kernels, got {kernel.ndim}D")


def fast_fft_correlate_images(image_a, image_b, correlation_method="circular",
                               normalized_correlation=True, conj=np.conj,
                               rfft2=None, irfft2=None, fftshift=None):
    """Faithful re-implementation of openpiv.pyprocess.fft_correlate_images,
    swapping numpy.fft (single-threaded) for scipy.fft (supports
    `workers=`, multi-threaded across all CPU cores) -- confirmed the
    second-largest chunk of a validated PIV pass after replace_nans.

    Cannot be done by monkeypatching openpiv.pyprocess.rfft2_/irfft2_/
    fftshift_ (the module-level numpy.fft names fft_correlate_images
    defaults to): Python default-argument values are bound ONCE at
    function-definition time, so reassigning those module names after
    the fact would silently do nothing to the already-defined function
    -- the whole function has to be replaced instead, same reasoning as
    filters.replace_nans in this module's apply_speedups().

    Numerically safe to swap: since NumPy 1.17, numpy.fft and scipy.fft
    are both backed by the same pocketfft C library -- this isn't two
    different algorithms, it's the same one, with scipy.fft additionally
    exposing the `workers` parallelism knob numpy.fft doesn't. Confirmed
    empirically (not just assumed) against real correlation arrays: see
    tests/unit/test_openpiv_speedups.py.

    correlation_method="linear" is user-selectable in the GUI (settings_
    panel's Correlation method combo) and intentionally NOT reimplemented
    here -- its zero-padding branch is a different, more involved
    computation than "circular"'s, and "circular" is both the default
    and, per openpiv's own docs, the normal/faster choice. Falls back to
    calling the original (slow but correct) function for "linear" rather
    than risking a subtly-wrong fast path for a branch this investigation
    didn't need to touch to fix the reported slowness.

    NOT USED -- deliberately excluded from apply_speedups(). Every
    isolated check passed (peak location identical on 2000/2000 real
    correlation windows; the value difference sits exactly at float32's
    precision floor, ~1e-7 relative, and vanishes to exactly 0.0 given
    float64 inputs -- this is genuine floating-point rounding noise from
    a different but equally valid summation order through the SAME
    underlying pocketfft algorithm, not a bug). But a full 4-pass
    end-to-end run on real data (A/B tested: identical settings, only
    this one patch toggled) showed that noise COMPOUNDS across passes --
    each pass's sub-pixel displacement estimate repositions where the
    NEXT (finer) pass samples its deformed interrogation windows, so a
    ~1e-7-relative perturbation in an early pass's correlation can shift
    which actual image content a later pass ends up correlating against.
    Result: up to 0.13 px of divergence on 126,162 of 189,857 vectors --
    utterly unacceptable for a pipeline whose whole point is exact
    reproducibility. The other three patches in this module were
    confirmed NOT to have this compounding effect (see
    apply_speedups()'s docstring and tests/unit/test_openpiv_speedups.py
    for the full 4-pass comparison). Kept defined, not wired up, so a
    future attempt at this specific optimization doesn't have to
    rediscover why it's unsafe from scratch -- don't re-enable this
    without re-running that full end-to-end comparison, not just the
    single-call check that looked fine here.
    """
    if correlation_method != "circular":
        from openpiv.pyprocess import fft_correlate_images as _orig_fft_correlate_images
        return _orig_fft_correlate_images(
            image_a, image_b, correlation_method=correlation_method,
            normalized_correlation=normalized_correlation, conj=conj)

    import scipy.fft as _scipy_fft
    from openpiv.pyprocess import normalize_intensity
    import os

    if normalized_correlation:
        image_a = normalize_intensity(image_a)
        image_b = normalize_intensity(image_b)

    s2 = np.array(image_b.shape[-2:])
    workers = os.cpu_count() or 1
    f2a = conj(_scipy_fft.rfft2(image_a, workers=workers))
    f2b = _scipy_fft.rfft2(image_b, workers=workers)
    corr = _scipy_fft.fftshift(_scipy_fft.irfft2(f2a * f2b, workers=workers).real, axes=(-2, -1))

    if normalized_correlation:
        corr = corr / (s2[0] * s2[1])
        corr = np.clip(corr, 0, 1)
    return corr


def fast_correlation_to_displacement(corr, n_rows, n_cols, subpixel_method="gaussian"):
    """Faithful vectorized re-implementation of
    openpiv.pyprocess.correlation_to_displacement, which loops over every
    window in pure Python calling find_subpixel_peak_position() on each
    one individually -- confirmed the dominant cost inside
    extended_search_area_piv for a high-window-count pass.

    Reproduces find_subpixel_peak_position's exact per-window logic,
    vectorized across all N windows at once:
    - peak location via argmax (numpy's argmax and the original's
      `np.unravel_index(np.argmax(corr), corr.shape)` both resolve ties
      to the first occurrence in C-order flattened traversal -- same
      convention, vectorized or not).
    - a window is invalid (NaN, NaN) if its peak sits exactly on the
      correlation map's border, checked BEFORE eps is added (matching
      the original's early return).
    - the 5-point neighbourhood (center + 4-neighbors) is read with the
      SAME `+= eps` bias as the original (`corr += eps` mutates the
      window in place before reading, so the 5 points read here are
      computed as `point + eps`, identically).
    - gaussian falls back to parabolic per-window using a strict `< 0`
      threshold on any of the 5 points -- NOT openpiv's own
      `vectorized_correlation_to_displacements`, which uses `<= 0` and a
      forced float32 cast (checked, found different, deliberately not
      used here).
    - no dtype is forced; the input's own float precision is preserved.
    """
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")

    n_windows, h, w = corr.shape
    flat = corr.reshape(n_windows, -1)
    peak_flat = np.argmax(flat, axis=1)
    peak_i = peak_flat // w
    peak_j = peak_flat % w

    on_border = (peak_i == 0) | (peak_i == h - 1) | (peak_j == 0) | (peak_j == w - 1)

    # Clamp border windows' indices to something safe (e.g. the center) so
    # the neighbour-index arithmetic below never goes out of bounds; their
    # result is overwritten with NaN afterward regardless of what's
    # computed here, matching the original's early `return (nan, nan)`
    # before ever touching neighbour values for those windows.
    safe_i = np.where(on_border, h // 2, peak_i)
    safe_j = np.where(on_border, w // 2, peak_j)

    eps = 1e-7
    idx = np.arange(n_windows)
    c = corr[idx, safe_i, safe_j] + eps
    cl = corr[idx, safe_i - 1, safe_j] + eps
    cr = corr[idx, safe_i + 1, safe_j] + eps
    cd = corr[idx, safe_i, safe_j - 1] + eps
    cu = corr[idx, safe_i, safe_j + 1] + eps

    use_parabolic = np.zeros(n_windows, dtype=bool)
    if subpixel_method == "gaussian":
        any_negative = (c < 0) | (cl < 0) | (cr < 0) | (cd < 0) | (cu < 0)
        use_parabolic = any_negative

    shift_i = np.zeros(n_windows, dtype=np.float64)
    shift_j = np.zeros(n_windows, dtype=np.float64)

    if subpixel_method == "centroid":
        shift_i = ((safe_i - 1) * cl + safe_i * c + (safe_i + 1) * cr) / (cl + c + cr)
        shift_j = ((safe_j - 1) * cd + safe_j * c + (safe_j + 1) * cu) / (cd + c + cu)
        subp_i = shift_i
        subp_j = shift_j
    else:
        gaussian_mask = ~use_parabolic if subpixel_method == "gaussian" else np.zeros(n_windows, dtype=bool)
        parabolic_mask = use_parabolic if subpixel_method == "gaussian" else np.ones(n_windows, dtype=bool)

        if np.any(gaussian_mask):
            with np.errstate(invalid="ignore", divide="ignore"):
                nom1 = np.log(cl) - np.log(cr)
                den1 = 2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr)
                nom2 = np.log(cd) - np.log(cu)
                den2 = 2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu)
                g_shift_i = np.divide(nom1, den1, out=np.zeros_like(nom1), where=(den1 != 0.0))
                g_shift_j = np.divide(nom2, den2, out=np.zeros_like(nom2), where=(den2 != 0.0))
            shift_i = np.where(gaussian_mask, g_shift_i, shift_i)
            shift_j = np.where(gaussian_mask, g_shift_j, shift_j)

        if np.any(parabolic_mask):
            with np.errstate(invalid="ignore", divide="ignore"):
                p_shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
                p_shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)
            shift_i = np.where(parabolic_mask, p_shift_i, shift_i)
            shift_j = np.where(parabolic_mask, p_shift_j, shift_j)

        subp_i = safe_i + shift_i
        subp_j = safe_j + shift_j

    subp_i = np.where(on_border, np.nan, subp_i)
    subp_j = np.where(on_border, np.nan, subp_j)

    default_i = np.floor(h / 2)
    default_j = np.floor(w / 2)
    v = subp_i - default_i
    u = subp_j - default_j

    return u.reshape(n_rows, n_cols), v.reshape(n_rows, n_cols)


def fast_local_median_val(u, v, u_threshold, v_threshold, size=1):
    """Faithful vectorized re-implementation of
    openpiv.validation.local_median_val, which uses
    scipy.ndimage.generic_filter(masked_u, np.nanmedian, ...) -- a Python
    callback invoked once per grid cell, confirmed a major cost of
    validation on a large grid.

    generic_filter's `mode="constant", cval=np.nan` sliding-window
    nanmedian is exactly reproduced by NaN-padding the array by `size` on
    each side and taking a strided sliding-window view, then a single
    vectorized np.nanmedian call over the window axes -- same 5 (or
    (2*size+1)^2) values fed to the same nanmedian function per cell,
    just without a quarter-million individual Python calls to get there.

    Wired in by apply_speedups() (patches openpiv.validation.local_median_val)
    -- reached only when CPUPIVProcess's opt-in per_pass_validation is on
    AND median_normalized is left False (the default is True, matching
    DaVis's own per-pass scheme -- see fast_local_norm_median_val below
    for the one actually exercised by that default); kept active for the
    non-normalized case too since it's the same cost problem either way.
    """
    if np.ma.is_masked(u):
        masked_u = np.where(np.ma.getmask(u), np.nan, np.ma.getdata(u))
        masked_v = np.where(np.ma.getmask(v), np.nan, np.ma.getdata(v))
    else:
        masked_u = np.asarray(u)
        masked_v = np.asarray(v)

    um = _sliding_nanmedian(masked_u, size)
    vm = _sliding_nanmedian(masked_v, size)

    ind = (np.abs(u - um) > u_threshold) | (np.abs(v - vm) > v_threshold)
    return ind


def _sliding_nanmedian(a, size):
    from numpy.lib.stride_tricks import sliding_window_view

    k = 2 * size + 1
    padded = np.pad(a.astype(np.float64, copy=False), size, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (k, k))
    import warnings
    with warnings.catch_warnings():
        # generic_filter(np.nanmedian, ...) emits the same "All-NaN slice"
        # RuntimeWarning for these same cells -- not a new behavior, just
        # keeping this path's console output no noisier than the original.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(windows, axis=(-2, -1))


def fast_local_norm_median_val(u, v, ε, threshold, size=1):
    # Parameter literally named `ε` (not `eps`) -- openpiv's own
    # typical_validation calls local_norm_median_val(u, v, ε=0.2,
    # threshold=..., size=...) with that exact unicode keyword, and
    # apply_speedups() patches this function in as a drop-in replacement
    # for that same call site.
    """Faithful vectorized re-implementation of
    openpiv.validation.local_norm_median_val (the Westerweel & Scarano
    "universal outlier detection" test -- DaVis's own per-pass validation
    method, hence median_normalized=True's default in CPUPIVProcess's
    opt-in per_pass_validation mode). The original calls
    scipy.ndimage.generic_filter FOUR TIMES per pass (um, vm, rm_u, rm_v),
    each a Python callback per grid cell -- confirmed the dominant new
    cost once per_pass_validation is enabled (measured ~+80s per frame
    pair on a real ~4000x3000 job, 4 passes x 4 generic_filter calls each
    on grids up to ~195k cells).

    Reproduces the original's exact per-window arithmetic vectorized
    across all cells at once via a single NaN-padded sliding-window view
    per array (shape (ny, nx, k*k), k=2*size+1) instead of one Python
    callback invocation per cell per generic_filter call:
    - um/vm: nanmedian over the FULL k*k window (center included),
      matching the original's plain generic_filter(nanmedian, ...) calls.
    - rm_u/rm_v: the original's rfunc sets the window's OWN center to NaN
      before taking its median (ym) and then the median of |window - ym|
      (also excluding center, since NaN propagates through the
      subtraction and nanmedian ignores it) -- reproduced by building a
      second window array with the middle column of the flattened k*k
      window set to NaN before both nanmedian calls.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings

    if np.ma.is_masked(u):
        masked_u = np.where(np.ma.getmask(u), np.nan, np.ma.getdata(u))
        masked_v = np.where(np.ma.getmask(v), np.nan, np.ma.getdata(v))
    else:
        masked_u = np.asarray(u, dtype=np.float64)
        masked_v = np.asarray(v, dtype=np.float64)

    k = 2 * size + 1
    center = k * k // 2

    def windows_of(a):
        padded = np.pad(a.astype(np.float64, copy=False), size, mode="constant", constant_values=np.nan)
        w = sliding_window_view(padded, (k, k)).reshape(a.shape[0], a.shape[1], k * k)
        return w

    wu, wv = windows_of(masked_u), windows_of(masked_v)
    wu_excl, wv_excl = wu.copy(), wv.copy()
    wu_excl[..., center] = np.nan
    wv_excl[..., center] = np.nan

    with warnings.catch_warnings():
        # Same "All-NaN slice"/"Mean of empty slice" RuntimeWarnings the
        # original's generic_filter(nanmedian, ...) calls already emit at
        # the frame's border cells -- not a new behavior.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        um = np.nanmedian(wu, axis=-1)
        vm = np.nanmedian(wv, axis=-1)
        ym_u = np.nanmedian(wu_excl, axis=-1)
        ym_v = np.nanmedian(wv_excl, axis=-1)
        rm_u = np.nanmedian(np.abs(wu_excl - ym_u[..., None]), axis=-1)
        rm_v = np.nanmedian(np.abs(wv_excl - ym_v[..., None]), axis=-1)

    r0ast_u = np.abs(masked_u - um) / (rm_u + ε)
    r0ast_v = np.abs(masked_v - vm) / (rm_v + ε)
    return np.sqrt(r0ast_u ** 2 + r0ast_v ** 2) > threshold


def loose_typical_validation(u, v, s2n, settings):
    """Replacement for openpiv.validation.typical_validation, wired in by
    apply_speedups(). Flags a vector invalid ONLY if it is literal NaN --
    never for any global-range, global-std, local-median, or sig2noise
    criterion. This is a deliberate BEHAVIOR CHANGE, not a faithful
    speedup: see this module's docstring for why (vector validation now
    lives entirely in processing.postprocess/PostProcessSettings, run
    once after the engine returns, not mid-calculation).

    Feeds directly into the SAME filters.replace_outliers() call that
    openpiv.windef.multipass_img_deform already makes unconditionally
    every pass -- so NaN cells (numerical instability, e.g. a window
    with no correlation peak) still get a bounded local-mean fill,
    protecting the next pass's RectBivariateSpline deformation (see
    cpu_engine.CPUPIVProcess/_fill_residual_nan for the documented crash
    this guards against), but no vector is ever treated as "invalid" for
    a value-based reason during calculation.

    `s2n` and `settings` are accepted (matching typical_validation's
    signature, since windef.py calls this by position/keyword) but
    unused -- sig2noise-based rejection is dropped entirely from this
    path; see cpu_engine.CPUPIVProcess forcing settings.sig2noise_validate
    = False, which also skips the extra correlation peak-search cost
    typical_validation's sig2noise branch would otherwise still pay
    for."""
    u_data = np.ma.getdata(u) if np.ma.is_masked(u) else np.asarray(u)
    v_data = np.ma.getdata(v) if np.ma.is_masked(v) else np.asarray(v)
    return np.isnan(u_data) | np.isnan(v_data)


_PATCHED = False
_REAL_TYPICAL_VALIDATION = None


def use_loose_validation():
    """Switch openpiv.validation.typical_validation to the NaN-only patch
    (this codebase's default calculation-time behavior -- see
    loose_typical_validation's docstring). Requires apply_speedups() to
    have run at least once."""
    import openpiv.validation
    openpiv.validation.typical_validation = loose_typical_validation


def use_real_validation():
    """Switch openpiv.validation.typical_validation to openpiv's OWN
    unpatched implementation (global range + std-dev + local-median/UOD,
    driven by whatever settings.min_max_u_disp/std_threshold/
    median_threshold/median_size/median_normalized the caller has set) --
    for CPUPIVProcess's opt-in per_pass_validation mode, see
    cpu_engine.py. Requires apply_speedups() to have run at least once
    (that's what captures the real function before patching it away)."""
    import openpiv.validation
    if _REAL_TYPICAL_VALIDATION is None:
        raise RuntimeError("apply_speedups() must run before use_real_validation()")
    openpiv.validation.typical_validation = _REAL_TYPICAL_VALIDATION


def apply_speedups():
    """Monkeypatch the functions above onto the installed openpiv
    package. Idempotent -- safe to call more than once. Never touches
    openpiv's own files on disk.

    Patches the NAME each caller actually resolves at call time, which
    isn't always the module the function is defined in:
    openpiv.filters does `from openpiv.lib import replace_nans` at its
    own import time, binding a private copy into openpiv.filters'
    namespace -- patching only openpiv.lib.replace_nans would silently
    do nothing, since openpiv.filters.replace_outliers (the actual
    caller) already holds its own reference to the original function.
    correlation_to_displacement and typical_validation are both called
    from within their OWN defining module (pyprocess.py's
    extended_search_area_piv, windef.py's multipass_img_deform calling
    `validation.typical_validation`), so patching the module attribute
    there is sufficient."""
    global _PATCHED, _REAL_TYPICAL_VALIDATION
    if _PATCHED:
        return
    import openpiv.filters
    import openpiv.lib
    import openpiv.pyprocess
    import openpiv.validation

    openpiv.lib.replace_nans = fast_replace_nans
    openpiv.filters.replace_nans = fast_replace_nans
    openpiv.pyprocess.correlation_to_displacement = fast_correlation_to_displacement
    _REAL_TYPICAL_VALIDATION = openpiv.validation.typical_validation
    openpiv.validation.typical_validation = loose_typical_validation
    # Only reached when a caller opts into per_pass_validation and
    # switches the module attribute above back to _REAL_TYPICAL_VALIDATION
    # (see use_real_validation()) -- patched unconditionally here anyway
    # since it's harmless when unreached and this is the only place
    # apply_speedups() ever runs.
    openpiv.validation.local_median_val = fast_local_median_val
    openpiv.validation.local_norm_median_val = fast_local_norm_median_val
    # fast_fft_correlate_images is intentionally NOT wired in here -- see
    # its docstring's final paragraph. It passed every isolated check
    # (peak location unchanged in 2000/2000 real windows, diff exactly at
    # float32's precision floor, vanishing to 0.0 with float64 inputs),
    # but a full 4-pass end-to-end run on real data showed the per-window
    # float32 rounding noise it introduces compounds across passes (each
    # pass's sub-pixel estimate re-positions the NEXT pass's deformed
    # sampling window) into real divergence: up to 0.13 px, on 126,162 of
    # 189,857 vectors -- confirmed by A/B-testing the exact same 4-pass
    # run with and without just this one patch. The two faithful-speedup
    # patches above (replace_nans, correlation_to_displacement) were
    # separately confirmed NOT to have this compounding effect (a
    # combined end-to-end run with only those two active, validation
    # held constant, matched the unpatched baseline to ~1e-15 on every
    # vector) -- keeping fast_fft_correlate_images defined but unused so
    # a future attempt doesn't have to rediscover why it's unsafe from
    # scratch. typical_validation's replacement above is a separate,
    # deliberate behavior change (not a faithful speedup) and is NOT
    # expected to match the unpatched baseline -- see
    # loose_typical_validation's docstring.
    _PATCHED = True
