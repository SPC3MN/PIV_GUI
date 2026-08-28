"""Drop-in vectorized replacements for openpiv-python functions used
during CPU-backend PIV calculation. Three kinds of patch live here, and
they are NOT the same kind of change:

FAITHFUL SPEEDUPS (fast_sliding_window_array, fast_replace_nans,
fast_correlation_to_displacement, fast_extended_search_area_piv): each is
a faithful re-expression of
the SAME formula as the original in vectorized numpy -- not openpiv's
own alternate "vectorized_*" functions
(which were checked and found to differ subtly: a `<= 0` vs `< 0`
gaussian-fallback threshold, and a forced float32 cast, either of which
would silently change results for a science-critical PIV run -- not
acceptable here, hence writing faithful equivalents instead of flipping
`use_vectorized=True`). Confirmed via cProfile against a real
3008x4096 frame pair (see the perf investigation this module came out
of): a single 32px/75%-overlap pass took ~113s, of which ~49% was
`openpiv.lib.replace_nans` and ~35% was `openpiv.pyprocess.
correlation_to_displacement` (a pure-Python loop over every window).
These are bit-exact vs. the unpatched originals.

PRECISION-UPGRADE SPEEDUP (fast_fft_correlate_images): NOT bit-exact vs.
today's output -- windows enter this function at float32 (frames are
cast to float32 once, in CPUPIVProcess.__call__, and openpiv's own
fft_correlate_images never widens that), and this patch upcasts to
float64 before correlating. That's the OTHER kind of change this
project's numerical bar allows ("bit-exact OR provably more accurate"):
the upcast is lossless and every downstream FFT/multiply/inverse-FFT
step then carries strictly less rounding error than doing the same chain
at float32, which is provable from the arithmetic, not just measured.
The backend swap bundled into the same patch (numpy.fft -> scipy.fft, for
speed) WAS separately verified empirically, not just assumed, to not
compound into anything meaningful across a real 4-pass run -- see the
function's own docstring and apply_speedups()'s comment at the
fft_correlate_images wiring line for the numbers.

BEHAVIOR-CHANGING PATCH (loose_typical_validation): unlike the two kinds
above, this one deliberately does NOT reproduce openpiv's own
validation.typical_validation. It is the path active when
per_pass_validation is False (config.schema.ValidationSettings; True by
default). openpiv's own multi-pass loop (windef.multipass_img_deform)
still calls `validation.typical_validation` unconditionally with no
settings flag to disable it, so this patch replaces it with a version
that flags a vector invalid ONLY when it is literal NaN -- never for any
global-range/std/median/sig2noise criterion. This keeps the per-pass
replace_outliers call (also unconditional in openpiv) as a pure
numerical-stability mechanism (see cpu_engine._fill_residual_nan for why
residual NaN between passes is dangerous, not just a validation
artifact) without it ever causing a vector to be counted "invalid" in
the final output. By default (per_pass_validation=True), use_real_
validation() is active instead, restoring openpiv's own real
typical_validation -- see cpu_engine.py's module docstring for the full
rationale and why that's still consistent with "the final valid/invalid
DECISION lives in processing.postprocess."

FAITHFUL ADDITION (peak2mean sig2noise in fast_extended_search_area_piv /
_correlation_to_displacement_flat): computes openpiv's own peak2mean
formula (peak correlation value / |mean of the correlation plane|,
zeroed for border/weak peaks -- see openpiv.validation.sig2noise_ratio)
from data the fast path already gathers per window, so requesting
sig2noise_method="peak2mean" no longer falls back to openpiv's slow,
unchunked correlation path the way any non-None sig2noise_method
originally did. Bit-exact vs. openpiv's own peak2mean output; see
tests/unit/test_openpiv_speedups.py.

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


def fast_sliding_window_array(image, window_size=(64, 64), overlap=(32, 32)):
    """Faithful vectorized re-implementation of
    openpiv.pyprocess.sliding_window_array, which builds two full-size
    int index arrays (win_x, win_y, each shape (n_windows, wy, wx)) and
    gathers via fancy indexing `image[win_y, win_x]` -- at this app's real
    workload (189,857 windows) that's 1.55 GB *each* just for the index
    arrays, to produce a 777 MB result. Confirmed the dominant cost of
    window extraction (5.98s -> 0.28s replacing it, np.array_equal exact)
    in the perf investigation this module came out of.

    openpiv's own construction (via get_rect_coordinates/get_coordinates,
    center_on_field=False) places window k's top-left corner at
    (row(k) * (window_size[0]-overlap[0]), col(k) * (window_size[1]-overlap[1])),
    row(k)/col(k) running 0..field_shape-1 in C order (row-major) -- i.e.
    a plain regular grid starting at the image's (0, 0) corner, stepping
    by (window_size - overlap) each direction, with never an out-of-bounds
    window (get_field_shape's floor division guarantees the last window's
    far edge never exceeds the image). That is EXACTLY what
    `sliding_window_view(image, window_size)[::step_y, ::step_x]` produces:
    sliding_window_view's [a, b, i, j] element is image[a+i, b+j], so after
    step-slicing by (step_y, step_x), position [row, col, i, j] is
    image[row*step_y + i, col*step_x + j] -- the same pixel openpiv's
    fancy-indexed gather reads for window (row, col)'s local offset (i, j).
    The row-major flatten below (via ascontiguousarray + reshape) matches
    openpiv's own `np.reshape(x, (-1, 1, 1))` C-order flatten of the same
    (n_rows, n_cols) grid, so windows[k] here is the same window as
    windows[k] there for every k, not just the same multiset.

    ascontiguousarray is required (not optional): sliding_window_view's
    output is a genuinely strided view, sharing memory with `image` and
    overlapping between adjacent windows -- reshape alone would raise on
    a non-contiguous array, and returning a view (rather than a copy)
    would silently break any caller that mutates its result in place
    (openpiv's own does, in the search_area_size > window_size branch of
    extended_search_area_piv), unlike the original's fancy-indexing gather
    which always allocates a fresh, independent array. ascontiguousarray
    forces exactly that same fresh-copy semantics.
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    wy, wx = window_size
    step_y = wy - overlap[0]
    step_x = wx - overlap[1]

    all_windows = np.lib.stride_tricks.sliding_window_view(image, (wy, wx))
    strided = all_windows[::step_y, ::step_x]
    n_rows, n_cols = strided.shape[0], strided.shape[1]
    return np.ascontiguousarray(strided).reshape(n_rows * n_cols, wy, wx)


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
    """Faithful-at-float64 re-implementation of
    openpiv.pyprocess.fft_correlate_images's "circular" branch, run at
    float64 instead of the original's float32 (frames enter CPUPIVProcess
    cast to float32, and openpiv's own fft_correlate_images never widens
    that), with scipy.fft in place of numpy.fft, chunked via
    perf.autotune.recommended_chunk_size() for cache locality.

    THIS IS NOT A BIT-EXACT PATCH relative to today's (float32) output --
    it's the OTHER kind this project's numerical bar allows ("bit-exact
    OR provably more accurate", see this module's docstring's opening).
    Upcasting float32 pixel data to float64 is lossless (every float32
    value has an exact float64 representation) and every arithmetic step
    of the FFT/multiply/inverse-FFT chain then carries strictly less
    rounding error than the same chain at float32 -- that's the "provably"
    part, not an empirical claim. It reproduces the investigation this
    module came out of finding ~1.35x more speed available from float32
    alone, beyond what float64 gets here, and documents that the project
    owner explicitly chose float64 over that extra speed.

    `workers=` is DELIBERATELY NOT PASSED to scipy.fft here. An earlier
    version of this docstring claimed scipy.fft's workers parameter gives
    "multi-threaded across all CPU cores -- confirmed". That was false
    and has been corrected: measured directly on this machine (scipy
    1.17.1, DUCC-backed pocketfft on Windows) -- timing rfft2 at
    workers=1/4/24/48 gives identical wall-clock every time, and
    scipy.fft.get_workers() reports 1 regardless of what's passed.
    In-process FFT threading is unavailable on this build; the actual
    parallelism route is process-level (see engines Tier 3 /
    ProcessPoolExecutor across frame pairs), not this parameter.

    Backend-equivalence (scipy.fft vs numpy.fft, held at the SAME float64
    precision) was measured here to NOT be exactly 0.000e+00 the way an
    earlier investigation (different machine) found -- at this dataset's
    actual window sizes (32px/64px), ~2-3% of correlation-array elements
    differ from a float64 numpy.fft reference by up to ~5e-13. That's
    ~6 orders of magnitude below the float32-vs-float32 backend
    divergence that was separately measured (see below) to compound into
    0.13 px across 4 passes, so it was verified negligible via the SAME
    kind of full 4-pass real-data A/B before this function was wired into
    apply_speedups() -- see apply_speedups()'s docstring for that result.
    If you bump scipy/numpy versions, re-run that comparison; don't
    assume the diff stays this small.

    correlation_method="linear" is user-selectable in the GUI (settings_
    panel's Correlation method combo) and intentionally NOT reimplemented
    here -- its zero-padding branch is a different, more involved
    computation than "circular"'s, and "circular" is both the default
    and, per openpiv's own docs, the normal/faster choice. Falls back to
    calling the original (slow but correct) function for "linear".

    Chunking (recommended_chunk_size()) processes `image_a`/`image_b` in
    windows-axis slices rather than one big batched FFT call -- verified
    this is exactly equivalent to one unchunked call regardless of chunk
    boundary (each window's FFT is independent; pocketfft has no
    cross-window state), so this is purely a cache-locality optimization
    (measured ~19% faster FFT time from a chunk's working set fitting in
    cache), never a numerical concern.
    """
    if correlation_method != "circular":
        # Must NOT re-`from openpiv.pyprocess import fft_correlate_images`
        # here once apply_speedups() has run: that name has been
        # monkeypatched to THIS function, so a fresh import would bind
        # _orig to itself and recurse forever. _REAL_FFT_CORRELATE_IMAGES
        # is captured by apply_speedups() before it patches the name
        # away -- same gotcha, same fix, as _REAL_TYPICAL_VALIDATION
        # below. Falls back to a fresh import only for the (test-only)
        # case of calling this function directly before apply_speedups()
        # has ever run.
        _orig = _REAL_FFT_CORRELATE_IMAGES
        if _orig is None:
            from openpiv.pyprocess import fft_correlate_images as _orig
        return _orig(
            image_a, image_b, correlation_method=correlation_method,
            normalized_correlation=normalized_correlation, conj=conj)

    import scipy.fft as _scipy_fft
    from openpiv.pyprocess import normalize_intensity

    from ..perf.autotune import recommended_chunk_size

    if normalized_correlation:
        image_a = normalize_intensity(image_a)
        image_b = normalize_intensity(image_b)

    s2 = np.array(image_b.shape[-2:])
    n_windows = image_a.shape[0]
    wy, wx = image_a.shape[-2], image_a.shape[-1]
    chunk = recommended_chunk_size((wy, wx))

    corr = np.empty((n_windows, wy, wx), dtype=np.float64)
    for start in range(0, n_windows, chunk):
        end = min(start + chunk, n_windows)
        a64 = image_a[start:end].astype(np.float64, copy=False)
        b64 = image_b[start:end].astype(np.float64, copy=False)
        f2a = conj(_scipy_fft.rfft2(a64))
        f2b = _scipy_fft.rfft2(b64)
        corr[start:end] = _scipy_fft.fftshift(_scipy_fft.irfft2(f2a * f2b).real, axes=(-2, -1))

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
    n_rows_arg, n_cols_arg = n_rows, n_cols
    # sig2noise discarded here -- this wrapper matches fast_correlation_
    # to_displacement's own (u, v)-only contract (its caller, whoever
    # that is, never asked for a 3rd value); fast_extended_search_area_piv
    # below is the one caller that actually wants sig2noise, and it calls
    # _correlation_to_displacement_flat directly, not through here.
    u, v, _sig2noise = _correlation_to_displacement_flat(corr, subpixel_method=subpixel_method)
    return u.reshape(n_rows_arg, n_cols_arg), v.reshape(n_rows_arg, n_cols_arg)


def _correlation_to_displacement_flat(corr, subpixel_method="gaussian"):
    """The same per-window math as fast_correlation_to_displacement, minus
    the final reshape to (n_rows, n_cols) -- factored out so
    fast_extended_search_area_piv can call it per chunk (a chunk covers a
    subset of the flattened window index range, which doesn't have its
    own meaningful n_rows/n_cols) and reshape the FULL grid only once,
    after every chunk has been written into it.

    ALSO returns sig2noise (peak2mean) as a 3rd value -- confirmed exact
    match to openpiv.pyprocess.sig2noise_ratio's own "peak2mean" formula
    (sig2noise = peak_value / |mean(correlation_plane)|, zeroed wherever
    the peak sits on the correlation plane's border or is below 1e-3):
    `flat` (the already-reshaped (n_windows, h*w) correlation data) and
    the peak location are BOTH already computed just above for the u/v
    displacement itself -- computing peak2mean here is one more mean()
    reduction over data already fully in memory, not a second
    correlation or extra FFT work. This is what makes real sig2noise-
    based rejection affordable on the fast/chunked path at all (see
    fast_extended_search_area_piv's docstring for why the ORIGINAL
    approach -- falling back to openpiv's own slow, unchunked
    implementation whenever sig2noise was requested -- was a much bigger
    cost than computing sig2noise itself ever was)."""
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")

    n_windows, h, w = corr.shape
    flat = corr.reshape(n_windows, -1)
    peak_flat = np.argmax(flat, axis=1)
    peak_i = peak_flat // w
    peak_j = peak_flat % w

    on_border = (peak_i == 0) | (peak_i == h - 1) | (peak_j == 0) | (peak_j == w - 1)

    # peak2mean sig2noise -- computed from the RAW peak value (no +eps,
    # matching sig2noise_ratio's own corr_max1) and flat's own mean, same
    # array `flat` the displacement math below also uses. corr_max1 is
    # zeroed (not just left small) for a border peak or one below 1e-3,
    # exactly mirroring sig2noise_ratio's own `condition` check -- both
    # mean "no usable signal here", scored as sig2noise=0 either way
    # (sig2noise_ratio's own final `sig2noise[isnan] = 0.0` line covers
    # the mean-is-zero case; replicated here via the same nan-out-then-
    # zero two-step for an identical result, not an approximation of it).
    raw_peak = flat[np.arange(n_windows), peak_flat]
    corr_max1 = np.where((raw_peak < 1e-3) | on_border, 0.0, raw_peak)
    corr_max2 = np.abs(flat.mean(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        sig2noise = corr_max1 / corr_max2
    sig2noise = np.where(np.isnan(sig2noise) | (corr_max2 == 0), 0.0, sig2noise)

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

    return u, v, sig2noise


def fast_extended_search_area_piv(frame_a, frame_b, window_size, overlap=(0, 0), dt=1.0,
                                   search_area_size=None, correlation_method="circular",
                                   subpixel_method="gaussian", sig2noise_method="peak2mean",
                                   width=2, normalized_correlation=False, use_vectorized=False):
    """Streaming re-implementation of
    openpiv.pyprocess.extended_search_area_piv: processes the interrogation
    grid a few ROWS at a time (window extraction -> correlate -> subpixel
    peak -> write into the u/v grid -> discard) instead of the original's
    three full-array passes (materialize ALL windows for both frames via
    sliding_window_array, correlate the WHOLE batch, THEN find peaks).

    Exact for the windows it processes: correlation windows are
    independent of each other (confirmed by the same reasoning as
    fast_fft_correlate_images's chunking -- no cross-window state), and
    this delegates the actual per-window math to fast_fft_correlate_images
    and _correlation_to_displacement_flat, the same functions the
    unchunked path already uses -- this is a different loop structure
    around identical arithmetic, not a different computation.

    Memory, not speed, is the point here: sliding_window_array is already
    fast after fast_sliding_window_array (Tier 1 item 1), and
    fast_fft_correlate_images already chunks its OWN FFT calls for cache
    locality -- but it still receives (and fast_sliding_window_array still
    allocates) the FULL per-frame window stack up front, and the full
    correlation output array. On this machine that's fine for ONE
    pipeline running alone (~3 GB peak for a 190k-window pass), but Tier
    3 runs many worker PROCESSES concurrently, each with its own peak --
    at 24-48 workers that stops being fine. Streaming by grid-row keeps
    each worker's peak footprint down to a few chunk rows' worth of
    windows, independent of image size or worker count.

    The outer (grid-row) chunk size comes from
    perf.autotune.recommended_pipeline_chunk_size(), NOT the same
    recommended_chunk_size() fast_fft_correlate_images uses one level
    down -- measured directly (not assumed) that reusing the smaller
    cache-locality target here is a real regression: a real image's grid
    has more columns per row than that target's window count, so every
    grid row became its own outer chunk (~370 outer iterations for one
    fine pass on this dataset), each paying fixed per-call numpy/Python
    overhead -- a measured 12% SLOWER wall-clock than not chunking this
    level at all, for a memory benefit the coarser, RAM/worker-count-
    derived target below already delivers without that overhead.

    ONLY fast-paths what this app actually exercises --
    search_area_size == window_size (this app never uses the "extended
    search area" case where the search region in frame B is intentionally
    larger than the window in frame A), sig2noise_method is None (forced
    by CPUPIVProcess -- see cpu_engine.py and Tier 1 item 3),
    use_vectorized=False (this app's default, and openpiv's own
    "vectorized_*" functions were already found to differ subtly -- see
    this module's opening docstring), correlation_method="circular" (the
    default; "linear" is user-selectable but rare). Falls back to the
    original (faithful, slower) implementation for everything else,
    rather than reimplementing branches this app doesn't exercise and
    can't verify against real data -- same pattern as
    fast_fft_correlate_images's "linear" fallback.
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    if search_area_size is None:
        search_area_size = window_size
    elif isinstance(search_area_size, int):
        search_area_size = (search_area_size, search_area_size)

    # sig2noise_method == "peak2mean" (this app's only real usage --
    # cpu_engine.py hard-codes it, PIVSettings' own default too) no
    # longer forces the slow fallback: _correlation_to_displacement_flat
    # computes peak2mean sig2noise "for free" from the SAME correlation-
    # plane data already needed for the displacement itself (see that
    # function's docstring) -- there's no real extra correlation/peak-
    # search cost to fall back to the slow path FOR. "peak2peak" still
    # falls back (needs a genuine second-peak search this fast path
    # doesn't implement), and so does sig2noise_method=None with
    # anything ELSE non-default below -- this only narrows the fallback
    # condition, doesn't remove it.
    if (search_area_size != window_size or use_vectorized or correlation_method != "circular"
            or sig2noise_method not in (None, "peak2mean")):
        _orig = _REAL_EXTENDED_SEARCH_AREA_PIV
        if _orig is None:
            from openpiv.pyprocess import extended_search_area_piv as _orig
        return _orig(
            frame_a, frame_b, window_size, overlap=overlap, dt=dt,
            search_area_size=search_area_size, correlation_method=correlation_method,
            subpixel_method=subpixel_method, sig2noise_method=sig2noise_method,
            width=width, normalized_correlation=normalized_correlation,
            use_vectorized=use_vectorized,
        )

    from openpiv.pyprocess import get_field_shape
    from ..perf.autotune import recommended_pipeline_chunk_size

    # Same validation as the original -- reproduced faithfully so a bad
    # call still fails the same way, not a chunking-path-specific error.
    if overlap[0] >= search_area_size[0] or overlap[1] >= search_area_size[1]:
        raise ValueError("Overlap has to be smaller than the search_area_size")
    if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
        raise ValueError("Search size cannot be smaller than the window_size")
    if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
        raise ValueError("window size cannot be larger than the image")

    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)
    n_windows = n_rows * n_cols
    wy, wx = search_area_size
    step_y = wy - overlap[0]
    step_x = wx - overlap[1]

    # Strided VIEWS over the whole image -- no per-window data copied yet
    # (same construction as fast_sliding_window_array, kept as a view here
    # instead of immediately materializing it).
    view_a = np.lib.stride_tricks.sliding_window_view(frame_a, (wy, wx))[::step_y, ::step_x]
    view_b = np.lib.stride_tricks.sliding_window_view(frame_b, (wy, wx))[::step_y, ::step_x]

    chunk_windows = recommended_pipeline_chunk_size((wy, wx))
    rows_per_chunk = max(1, chunk_windows // max(1, n_cols))

    u_flat = np.empty(n_windows, dtype=np.float64)
    v_flat = np.empty(n_windows, dtype=np.float64)
    s2n_flat = np.empty(n_windows, dtype=np.float64)

    for row_start in range(0, n_rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, n_rows)
        # Materialize (copy) only THIS chunk's rows of windows -- the same
        # ascontiguousarray+reshape fast_sliding_window_array uses, just
        # applied to a row slice of the grid instead of the whole thing.
        a_chunk = np.ascontiguousarray(view_a[row_start:row_end]).reshape(-1, wy, wx)
        b_chunk = np.ascontiguousarray(view_b[row_start:row_end]).reshape(-1, wy, wx)

        corr_chunk = fast_fft_correlate_images(
            a_chunk, b_chunk, correlation_method="circular",
            normalized_correlation=normalized_correlation,
        )
        u_chunk, v_chunk, s2n_chunk = _correlation_to_displacement_flat(
            corr_chunk, subpixel_method=subpixel_method)

        flat_start, flat_end = row_start * n_cols, row_end * n_cols
        u_flat[flat_start:flat_end] = u_chunk
        s2n_flat[flat_start:flat_end] = s2n_chunk
        v_flat[flat_start:flat_end] = v_chunk

    u = u_flat.reshape(n_rows, n_cols)
    v = v_flat.reshape(n_rows, n_cols)
    # sig2noise_method is None -> the original's own np.nan placeholder
    # (nothing asked for sig2noise, so nothing was computed above either
    # -- _correlation_to_displacement_flat always computes it, but this
    # branch simply isn't the one a None request should surface it from).
    # "peak2mean" -> the real, cheaply-computed values collected above.
    sig2noise = s2n_flat.reshape(n_rows, n_cols) if sig2noise_method == "peak2mean" \
        else np.full((n_rows, n_cols), np.nan)
    return u / dt, v / dt, sig2noise


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
    path. This is the path active when per_pass_validation is False; see
    cpu_engine.CPUPIVProcess forcing settings.sig2noise_validate = False
    (and settings.sig2noise_method = None) in that case, which also skips
    the extra correlation peak-search cost a real sig2noise ratio would
    otherwise pay for. When per_pass_validation is True instead,
    use_real_validation() (below) is active and DOES use a real sig2noise
    ratio -- see cpu_engine.CPUPIVProcess's __init__ and this module's top
    docstring."""
    u_data = np.ma.getdata(u) if np.ma.is_masked(u) else np.asarray(u)
    v_data = np.ma.getdata(v) if np.ma.is_masked(v) else np.asarray(v)
    return np.isnan(u_data) | np.isnan(v_data)


_PATCHED = False
_REAL_TYPICAL_VALIDATION = None
_REAL_FFT_CORRELATE_IMAGES = None
_REAL_EXTENDED_SEARCH_AREA_PIV = None


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
    sliding_window_array, fft_correlate_images, correlation_to_displacement,
    and typical_validation are all called from within their OWN defining
    module (pyprocess.py's extended_search_area_piv calling
    sliding_window_array, fft_correlate_images, and
    correlation_to_displacement, windef.py's multipass_img_deform calling
    `validation.typical_validation`), so patching the module attribute
    there is sufficient."""
    global _PATCHED, _REAL_TYPICAL_VALIDATION, _REAL_FFT_CORRELATE_IMAGES, _REAL_EXTENDED_SEARCH_AREA_PIV
    if _PATCHED:
        return
    import openpiv.filters
    import openpiv.lib
    import openpiv.pyprocess
    import openpiv.validation
    import openpiv.windef

    openpiv.pyprocess.sliding_window_array = fast_sliding_window_array
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

    # fast_fft_correlate_images IS wired in, unlike the other three
    # faithful speedups above it's NOT bit-exact vs. today's float32
    # output by design (see its own docstring: float32->float64 is a
    # lossless-upcast, provably-more-accurate change, the other kind this
    # project's numerical bar allows alongside bit-exact). What had to be
    # separately verified, on THIS machine, before wiring it in: does the
    # scipy.fft-vs-numpy.fft BACKEND swap (held at the same float64
    # precision) compound across passes the way the float32 backend swap
    # was found to. It does not -- a full 4-pass real-data A/B (scipy.fft
    # float64 vs. a numpy.fft float64 reference, everything else held
    # identical) gave a worst-case 2.3e-9 px divergence across all 189,857
    # vectors on the real dataset, ~8 orders of magnitude under the 0.13 px
    # threshold that got the float32 version rejected -- see
    # fast_fft_correlate_images's own docstring for the numbers. If you
    # bump scipy/numpy versions, re-run that A/B (not just an isolated
    # per-call check) before trusting this stays wired in.
    _REAL_FFT_CORRELATE_IMAGES = openpiv.pyprocess.fft_correlate_images
    openpiv.pyprocess.fft_correlate_images = fast_fft_correlate_images

    # fast_extended_search_area_piv only fast-paths the branches this app
    # exercises (search_area_size == window_size, sig2noise_method=None,
    # use_vectorized=False, correlation_method="circular") and falls back
    # to the real original -- captured here, before patching, for the
    # same reason _REAL_FFT_CORRELATE_IMAGES is -- for everything else.
    # windef.py does `from openpiv.pyprocess import extended_search_area_piv`
    # at ITS OWN import time (the exact same private-copy gotcha as
    # openpiv.filters/replace_nans above) -- windef.first_pass and
    # multipass_img_deform call THEIR OWN bound name, so patching only
    # openpiv.pyprocess.extended_search_area_piv would silently never
    # reach them. Both names have to be patched.
    _REAL_EXTENDED_SEARCH_AREA_PIV = openpiv.pyprocess.extended_search_area_piv
    openpiv.pyprocess.extended_search_area_piv = fast_extended_search_area_piv
    openpiv.windef.extended_search_area_piv = fast_extended_search_area_piv

    _PATCHED = True
