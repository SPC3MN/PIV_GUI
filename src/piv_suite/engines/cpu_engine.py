"""CPU engine -- multi-pass, window-deformation openpiv-python processing.

Migrated unchanged from piv_common.CPUPIVProcess/init_cpu_processor
(identical across Planar_PIV_CPU and Stereo_PIV_CPU). No cupy/GPU imports
anywhere in this module -- this engine works on any machine with plain
openpiv-python installed.

THE FINAL VALID/INVALID DECISION LIVES IN POST-PROCESSING, NOT HERE --
but by default (per_pass_validation=True) this engine now also REPLACES
(never permanently rejects) a spurious vector mid-calculation, between
passes, before it gets deformed into the next, finer pass. The openpiv
internals this engine drives (windef.first_pass / multipass_img_deform)
call validation.typical_validation and filters.replace_outliers
unconditionally every pass -- openpiv gives no flag to disable that --
and _openpiv_speedups.apply_speedups()/use_real_validation() (see
__init__) swaps typical_validation for openpiv's OWN real
implementation when per_pass_validation is True, so a vector failing
either the per-pass local-median (UOD) test OR a per-pass correlation
peak2mean signal-to-noise threshold gets locally-mean-replaced by
replace_outliers right there, before deformation -- matching DaVis's own
multi-pass postprocessing, which pairs a median removal factor with a
peak-ratio ("scalarfield") threshold (confirmed via a real DaVis
JobHistory.xml). When per_pass_validation is False,
use_loose_validation() is active instead, and _openpiv_speedups.
loose_typical_validation flags a vector invalid ONLY when it is literal
NaN -- replace_outliers is then a pure numerical-stability mechanism
(preventing a residual-NaN cell from poisoning the NEXT pass's
RectBivariateSpline deformation -- see _fill_residual_nan below for the
documented crash this guards against), and no real vector is ever
replaced mid-calculation. Either way, self.val_locations is always
all-False (all-valid): a vector replaced here was already fixed up, not
marked invalid, so the ENTIRE final valid/invalid DECISION is still made
once, downstream, in processing.postprocess, driven by
PostProcessSettings -- this engine's replacement pass only changes what
value reaches that decision, matching how DaVis's own multi-pass
postprocessing works (its final pass's useScalarfieldThreshold is
separately false; the threshold only applies mid-calculation there too).

CPUPIVProcess.__init__ also applies the two remaining faithful,
numerically-verified-equivalent speedups from the same module for
functions that otherwise dominate wall-clock time on large,
high-overlap, multi-pass runs (confirmed on a real 3008x4096 frame pair:
~49% of a single 32px/75%-overlap pass was replace_nans, ~35% was a
pure-Python per-window loop in correlation_to_displacement). See
_openpiv_speedups.py for exactly what's patched, what was tried and
rejected (an FFT-backend swap that looked safe in isolation but
compounded into real divergence across passes), and why.
"""

import dataclasses

import numpy as np


def _fill_residual_nan(u, v):
    """openpiv's own filters.replace_outliers (method=localmean, bounded
    by max_filter_iteration/filter_kernel_size) can't always fully repair
    a pass's literal-NaN cells (flagged by loose_typical_validation, see
    this module's docstring -- "flagged" here means NaN, not a
    value-based rejection) -- e.g. a no-signal region sitting right at
    the frame edge has no far-side valid neighbor for a local mean to
    converge from, no matter how many iterations run. Confirmed on a
    real image pair (Windows/CUDA hardware): 310 vectors survived
    pass 1's own replace_outliers call as literal NaN, all clustered in
    the bottom few grid rows.

    Left alone, that NaN doesn't stay local -- the NEXT pass builds a
    RectBivariateSpline over the whole field to interpolate the
    deformation grid, and a spline fit poisons ENTIRELY (the whole
    interpolated surface goes NaN, not just cells near the bad region) if
    even one input point is NaN. That all-NaN deformation field then
    reaches scipy.ndimage.map_coordinates, whose native bounds-clamping
    doesn't safely handle NaN coordinates -- confirmed via a real
    Windows fatal exception (access violation), not a Python exception.

    Filling with this pass's own valid-vector median is a safe, bounded
    stand-in -- these cells are already reflected as invalid in the
    flags/val_locations mask this function's caller already tracks, so
    filling here only prevents literal NaN from reaching the next pass,
    it doesn't change what counts as a valid vector in the output."""
    u_data = np.ma.getdata(u) if isinstance(u, np.ma.MaskedArray) else u
    v_data = np.ma.getdata(v) if isinstance(v, np.ma.MaskedArray) else v
    nan_u = np.isnan(u_data)
    nan_v = np.isnan(v_data)
    if nan_u.any():
        u_data[nan_u] = np.nanmedian(u_data)
    if nan_v.any():
        v_data[nan_v] = np.nanmedian(v_data)
    return u, v


class CPUPIVProcess:
    """Adapter around openpiv-python's own multi-pass pipeline
    (`openpiv.windef.first_pass` / `multipass_img_deform`, driven by an
    `openpiv.settings.PIVSettings` object), shaped to match engines.base's
    PIVEngine Protocol (.coords, .val_locations, .scaling_par, and being
    callable as process(frame_a, frame_b) -> (u, v)) so
    processing.pipeline.process_frames() doesn't need to branch on
    backend.

    This replicates the per-pair body of `openpiv.windef.piv()` (coarse
    grid, decreasing window size per pass, image deformation between
    passes, a NaN-only safety fill between passes, optional smoothn)
    directly on in-memory frame arrays. cpu_settings keys are
    `PIVSettings` field names (e.g. windowsizes, overlap, filter_method,
    smoothn, ...) -- see openpiv.settings.PIVSettings for the full list;
    unknown keys are warned about, not silently dropped. sig2noise_validate/
    sig2noise_method/validation_first_pass/replace_vectors are accepted if
    present in cpu_settings (for backward compatibility with older settings
    dicts) but are always overridden below, driven by per_pass_validation
    instead -- see this module's docstring.

    per_pass_validation/per_pass_median_threshold/per_pass_median_size/
    per_pass_sig2noise_threshold (not PIVSettings fields; popped before the
    above) opt into openpiv's own real per-pass local-median (UOD) AND
    peak2mean signal-to-noise validation between passes instead of the
    NaN-only default -- see config.schema.ValidationSettings' docstring."""

    def __init__(self, frame_shape, **cpu_settings):
        from openpiv.settings import PIVSettings
        from openpiv.pyprocess import get_rect_coordinates

        from ._openpiv_speedups import apply_speedups, use_loose_validation, use_real_validation
        apply_speedups()

        # Opt-in per-pass validation (see config.schema.ValidationSettings'
        # docstring) -- popped before the unknown-fields check below since
        # these aren't PIVSettings fields themselves, they control WHICH
        # validation function windef's per-pass calls resolve to.
        per_pass_validation = cpu_settings.pop("per_pass_validation", False)
        per_pass_median_threshold = cpu_settings.pop("per_pass_median_threshold", 2.0)
        per_pass_median_size = cpu_settings.pop("per_pass_median_size", 1)
        per_pass_sig2noise_threshold = cpu_settings.pop("per_pass_sig2noise_threshold", 1.0)

        settings = PIVSettings()
        valid_fields = {f.name for f in dataclasses.fields(settings)}
        unknown = sorted(set(cpu_settings) - valid_fields)
        if unknown:
            print(f"[warn] cpu_settings has keys PIVSettings won't recognize: "
                  f"{unknown} -- check spelling against "
                  "openpiv.settings.PIVSettings's fields")
        for key, val in cpu_settings.items():
            if key in valid_fields:
                setattr(settings, key, val)

        settings.windowsizes = tuple(settings.windowsizes)
        settings.overlap = tuple(settings.overlap)
        if len(settings.overlap) != len(settings.windowsizes):
            raise ValueError(
                f"cpu_settings.overlap (length {len(settings.overlap)}) must "
                f"have the same length as windowsizes (length "
                f"{len(settings.windowsizes)}) -- one entry per pass"
            )
        settings.num_iterations = len(settings.windowsizes)

        # Global module-level toggle, set once per engine construction --
        # safe because this app runs one CPUPIVProcess per project/batch
        # sequentially (never multiple engines with different validation
        # settings concurrently in the same process).
        self._per_pass_validation = per_pass_validation
        if per_pass_validation:
            settings.median_normalized = True
            settings.median_threshold = per_pass_median_threshold
            settings.median_size = per_pass_median_size
            # DaVis's per-pass "multi-pass postprocessing" step pairs a
            # local-median/UOD test with a peak-ratio ("scalarfield")
            # threshold -- no global range or std-dev criterion -- so
            # widen openpiv's other typical_validation checks to
            # effectively never trigger, isolating the local-median and
            # sig2noise tests as the only ones that can actually flag (and
            # get locally-mean-replaced by replace_outliers) a vector here.
            settings.min_max_u_disp = (-1e6, 1e6)
            settings.min_max_v_disp = (-1e6, 1e6)
            settings.std_threshold = 1e6
            # peak2mean sig2noise, computed cheaply on the fast correlation
            # path from data it already gathers per window -- see
            # _openpiv_speedups.py's fast_extended_search_area_piv /
            # _correlation_to_displacement_flat docstrings for how the
            # naive version of this (before that module's fast path
            # accounted for "peak2mean" specifically) used to fall back to
            # openpiv's slow, unchunked correlation entirely. Bundled with
            # per_pass_validation rather than a separate flag since a
            # real sig2noise ratio is only ever produced when
            # use_real_validation() (below) is also active -- see this
            # module's top docstring.
            settings.sig2noise_validate = True
            settings.sig2noise_method = "peak2mean"
            settings.sig2noise_threshold = per_pass_sig2noise_threshold
            use_real_validation()
        else:
            # loose_typical_validation ignores its s2n argument entirely
            # (see this module's docstring), so there's no reason to pay
            # for computing one -- keep it off for the loose/NaN-only path.
            settings.sig2noise_validate = False
            settings.sig2noise_method = None
            use_loose_validation()

        self._settings = settings
        self.scaling_par = 1.0
        self.coords = get_rect_coordinates(frame_shape, settings.windowsizes[-1], settings.overlap[-1])
        self.val_locations = None

        # Autotune sanity: log what perf.autotune actually chose for this
        # engine's window sizes -- printed once per engine build (once
        # per worker process/batch, not per pair or per pass), so a user
        # can confirm the auto-tuner picked something sensible for their
        # machine without needing to read perf/autotune.py's source.
        from ..perf.autotune import recommended_chunk_size, recommended_pipeline_chunk_size
        chunk_info = ", ".join(
            f"{w}px->{recommended_chunk_size((w, w))}/chunk (fft), "
            f"{recommended_pipeline_chunk_size((w, w))}/chunk (pipeline)"
            for w in sorted(set(settings.windowsizes))
        )
        print(f"[info] perf autotune: {chunk_info}")

    def __call__(self, frame_a, frame_b, cancel_check=None):
        """cancel_check (see engines.base.PIVEngine.__call__'s docstring)
        is polled once per pass boundary, below -- NOT mid-pass, since a
        single pass's own correlation is one blocking openpiv call this
        engine has no hook into. A typical 3-4 pass schedule still means
        cancellation latency is bounded by ONE pass's wall-clock time
        instead of the whole pair's, which is the actual, honest win
        available here without either interrupting a blocking numpy/FFT
        call (unsafe, not attempted -- see pipeline_worker.py) or
        rewriting openpiv's own windef internals."""
        from openpiv import windef, validation, filters
        from .base import EngineCancelled

        settings = self._settings
        frame_a = np.asarray(frame_a, dtype=np.float32)
        frame_b = np.asarray(frame_b, dtype=np.float32)

        # -- pass 0 (coarsest window) --
        x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, settings)
        grid_mask = np.zeros_like(u, dtype=bool)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)

        # loose_typical_validation (patched in via apply_speedups) only
        # ever flags literal NaN -- always run it and the paired
        # replace_outliers call, unconditionally, as a pure numerical-
        # stability fill. No real vector is ever rejected here; see this
        # module's docstring.
        flags = validation.typical_validation(u, v, s2n, settings)
        u, v = filters.replace_outliers(
            u, v, flags, method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )
        u, v = _fill_residual_nan(u, v)

        if settings.smoothn:
            from openpiv import smoothn as _smoothn
            u, *_ = _smoothn.smoothn(u, s=settings.smoothn_p)
            v, *_ = _smoothn.smoothn(v, s=settings.smoothn_p)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)

        # -- passes 1..N-1 (decreasing window size, image deformation) --
        for i in range(1, settings.num_iterations):
            if cancel_check is not None and cancel_check():
                raise EngineCancelled(f"cancelled before pass {i}/{settings.num_iterations - 1}")
            x, y, u, v, grid_mask, flags = windef.multipass_img_deform(
                frame_a, frame_b, i, x, y, u, v, settings)
            u, v = _fill_residual_nan(u, v)
            if settings.smoothn and i < settings.num_iterations - 1:
                from openpiv import smoothn as _smoothn
                u, *_ = _smoothn.smoothn(u, s=settings.smoothn_p)
                v, *_ = _smoothn.smoothn(v, s=settings.smoothn_p)
            u = np.ma.masked_array(u, np.ma.nomask)
            v = np.ma.masked_array(v, np.ma.nomask)

        u = np.ma.filled(u, 0.0)
        v = np.ma.filled(v, 0.0)
        u = u / settings.dt
        v = v / settings.dt

        # Always all-valid: validation is no longer decided during
        # calculation -- pipeline.process_frames' `valid = ~val_locations`
        # is meant to be entirely determined by processing.postprocess
        # (PostProcessSettings) instead. See this module's docstring.
        self.val_locations = np.zeros_like(u, dtype=bool)
        return u, v


def init_cpu_processor(frame_shape, cpu_settings):
    process = CPUPIVProcess(frame_shape, **cpu_settings)
    x, y = process.coords
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y
