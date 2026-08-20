"""Adapters: canonical ProjectConfig sections <-> each backend's native
settings vocabulary (GPU's flat `piv_settings` dict / CPU's
`openpiv.settings.PIVSettings` field names). Neither engine needs to know
about the other's key names -- engines.registry's factories receive
already-adapted settings dicts from these functions.

Verified against the original repos' real defaults: canonical passes
[(64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75)] expand to EXACTLY the
original CPU default (windowsizes=[64,32,32,32], overlap=[32,24,24,24])
and the original GPU default (min_search_size=32,
search_size_iters=(1, 3), overlap_ratio=(0.5, 0.75)) -- i.e. both original
pipelines' defaults were already the same multi-pass schedule, just
expressed in two different vocabularies.
"""

from .schema import CorrelationSettings, PassSettings, ValidationSettings


def _grouped_passes(passes):
    """Group consecutive passes sharing the same window_size into
    (window_size, overlap_fraction, count) tuples, coarse-to-fine, same
    order as given. Warns (doesn't raise) if overlap_fraction varies
    within a group -- the GPU vocabulary only has one overlap_ratio per
    window-size level, so the group's first pass's fraction wins."""
    groups = []
    for p in passes:
        if groups and groups[-1][0] == p.window_size:
            size, frac, count = groups[-1]
            if frac != p.overlap_fraction:
                print(f"[warn] passes at window_size={p.window_size} have "
                      f"differing overlap_fraction ({frac} vs "
                      f"{p.overlap_fraction}) -- GPU's overlap_ratio is one "
                      f"value per window-size level; using {frac}")
            groups[-1] = (size, frac, count + 1)
        else:
            groups.append((p.window_size, p.overlap_fraction, 1))
    return groups


def passes_to_cpu(passes):
    """-> (windowsizes: list[int], overlap: list[int] in PIXELS) -- CPU's
    vocabulary lists every pass explicitly (no implicit repeat-count)."""
    windowsizes = [p.window_size for p in passes]
    overlap = [round(p.window_size * p.overlap_fraction) for p in passes]
    return windowsizes, overlap


def passes_to_gpu(passes):
    """-> (min_search_size: int, search_size_iters: tuple[int, ...],
    overlap_ratio: tuple[float, ...]) -- GPU's vocabulary compresses
    consecutive same-window-size passes into one level with an iteration
    count. min_search_size is the FINEST (last) pass's window size, and
    must be the smallest across all passes (validated)."""
    groups = _grouped_passes(passes)
    min_search_size = groups[-1][0]
    if any(size < min_search_size for size, _, _ in groups):
        raise ValueError(
            f"passes_to_gpu: the finest (last) pass's window_size "
            f"({min_search_size}) must be the SMALLEST across all passes "
            f"-- GPU's min_search_size convention doubles window size "
            f"going coarse; got sizes {[g[0] for g in groups]}"
        )
    search_size_iters = tuple(count for _, _, count in groups)
    overlap_ratio = tuple(frac for _, frac, _ in groups)
    return min_search_size, search_size_iters, overlap_ratio


def passes_from_cpu(windowsizes, overlap):
    """Inverse of passes_to_cpu -- reconstructs canonical PassSettings from
    a legacy cpu_settings' windowsizes/overlap-in-pixels lists. Used by
    scripts/migrate_legacy_config.py."""
    return [PassSettings(w, o / w) for w, o in zip(windowsizes, overlap)]


def passes_from_gpu(min_search_size, search_size_iters, overlap_ratio):
    """Inverse of passes_to_gpu -- expands a legacy piv_settings'
    (min_search_size, search_size_iters, overlap_ratio) level-compressed
    form back into an explicit per-pass list, coarse-to-fine. Used by
    scripts/migrate_legacy_config.py."""
    num_levels = len(search_size_iters)
    passes = []
    for level, (count, frac) in enumerate(zip(search_size_iters, overlap_ratio)):
        size = min_search_size * (2 ** (num_levels - 1 - level))
        passes.extend(PassSettings(size, frac) for _ in range(count))
    return passes


def to_cpu_settings(correlation: CorrelationSettings, validation: ValidationSettings) -> dict:
    """-> cpu_settings dict, ready for engines.cpu_engine.CPUPIVProcess /
    init_cpu_processor (openpiv.settings.PIVSettings field names).

    No sig2noise_*/validation_first_pass/replace_vectors keys -- those
    ValidationSettings fields were removed (validation now lives entirely
    in PostProcessSettings; CPUPIVProcess itself force-disables
    sig2noise_validate and always runs its NaN-only per-pass fill, see
    engines/cpu_engine.py)."""
    windowsizes, overlap = passes_to_cpu(correlation.passes)
    return {
        "windowsizes": windowsizes,
        "overlap": overlap,
        "dt": correlation.dt,
        "correlation_method": correlation.correlation_method,
        "subpixel_method": correlation.subpixel_method,
        "deformation_method": correlation.deformation_method,
        "interpolation_order": correlation.interpolation_order,
        "filter_method": validation.filter_method,
        "max_filter_iteration": validation.max_filter_iteration,
        "filter_kernel_size": validation.filter_kernel_size,
        "smoothn": validation.smoothn,
        "smoothn_p": validation.smoothn_p,
        "per_pass_validation": validation.per_pass_validation,
        "per_pass_median_threshold": validation.per_pass_median_threshold,
        "per_pass_median_size": validation.per_pass_median_size,
    }


def to_gpu_settings(correlation: CorrelationSettings, validation: ValidationSettings):
    """-> (min_search_size: int, piv_settings: dict), ready for
    engines.gpu_engine.init_gpu_processor (piv_gpu's own kwarg names).

    Only the fields with a clear canonical-schema equivalent are set
    here -- GPU-only knobs with no canonical-schema field yet (shrink_ratio,
    center, normalize, mask_zero, n_fft, deforming_par, s2n_size,
    validation_size, median_tol, mad_tol, mean_tol, rms_tol, scaling_par,
    mask, dtype_f) are left unset, so piv_gpu falls back to its own
    internal defaults for those -- they're not modeled in the canonical
    schema yet (a real gap to close once GPU-specific tuning needs are
    clearer, not a silent behavior change: piv_gpu's own defaults apply
    exactly as if a hand-written piv_settings dict had omitted them too).

    replacing_method is deliberately NOT validation.filter_method passed
    straight through -- confirmed against a real error from piv_gpu itself
    (openpiv_gpu.gpu_process.piv_gpu.__init__'s own assertion): it must be
    a TUPLE with one entry per pass (length == len(search_size_iters)),
    drawn from {'median', 'spring', 'mean'} -- a different vocabulary AND
    shape than CPU's scalar filter_method (which allows 'localmean',
    'disk', 'distance' -- none of which piv_gpu accepts). 'median' per
    pass is the closest equivalent to CPU's 'localmean' default.

    s2n_tol/median_tol/mad_tol/mean_tol/rms_tol are ALL hard-coded to
    None (validation now lives entirely in PostProcessSettings, not
    here): ValidationGPU only computes/ORs a criterion when its tol is
    not None, so with all five None, piv_gpu.val_locations is always
    all-False -- confirmed by reading openpiv_gpu/gpu_validation.py and
    gpu_process.py directly, no vendored-repo changes needed.
    num_replacing_iters is hard-coded to 0 (replace_outliers short-
    circuits immediately once val_locations is always empty, so any
    other value would be inert -- 0 says so honestly) and revalidate is
    hard-coded to False (nothing to revalidate against)."""
    min_search_size, search_size_iters, overlap_ratio = passes_to_gpu(correlation.passes)
    num_passes = len(search_size_iters)
    piv_settings = {
        "search_size_iters": search_size_iters,
        "overlap_ratio": overlap_ratio,
        "dt": correlation.dt,
        "subpixel_method": correlation.subpixel_method,
        "s2n_tol": None,
        "median_tol": None,
        "mad_tol": None,
        "mean_tol": None,
        "rms_tol": None,
        "replacing_method": ("median",) * num_passes,
        "num_replacing_iters": 0,
        "replacing_size": validation.filter_kernel_size,
        "revalidate": False,
        "smooth": validation.smoothn,
        "smoothing_par": validation.smoothn_p,
    }
    if correlation.batch_size is not None:
        piv_settings["batch_size"] = correlation.batch_size
    return min_search_size, piv_settings
