"""CPU engine -- multi-pass, window-deformation openpiv-python processing.

Migrated unchanged from piv_common.CPUPIVProcess/init_cpu_processor
(identical across Planar_PIV_CPU and Stereo_PIV_CPU). No cupy/GPU imports
anywhere in this module -- this engine works on any machine with plain
openpiv-python installed.

CPUPIVProcess.__init__ applies _openpiv_speedups.apply_speedups() before
any processing -- three drop-in, numerically-verified-equivalent
vectorized replacements for openpiv functions that otherwise dominate
wall-clock time on large, high-overlap, multi-pass runs (confirmed on a
real 3008x4096 frame pair: ~49% of a single 32px/75%-overlap pass was
replace_nans, ~35% was a pure-Python per-window loop in
correlation_to_displacement, ~13% was local_median_val's generic_filter
callback). Full multi-pass output matches the unpatched baseline to
~1e-15 per vector on real data -- see _openpiv_speedups.py for exactly
what's patched, what was tried and rejected (an FFT-backend swap that
looked safe in isolation but compounded into real divergence across
passes), and why.
"""

import dataclasses

import numpy as np


def _fill_residual_nan(u, v):
    """openpiv's own filters.replace_outliers (method=localmean, bounded
    by max_filter_iteration/filter_kernel_size) can't always fully repair
    a pass's flagged-invalid vectors -- e.g. a no-signal region sitting
    right at the frame edge has no far-side valid neighbor for a local
    mean to converge from, no matter how many iterations run. Confirmed
    on a real image pair (Windows/CUDA hardware): 310 vectors survived
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
    passes, sig2noise/global/median validation, iterative outlier
    replacement, optional smoothn) directly on in-memory frame arrays --
    i.e. the same multi-pass + validation + replacement feature set as
    piv_gpu, just via openpiv-python's implementation of it instead of
    piv_gpu's own. cpu_settings keys are `PIVSettings` field names (e.g.
    windowsizes, overlap, sig2noise_threshold, filter_method, smoothn,
    ...) -- see openpiv.settings.PIVSettings for the full list; unknown
    keys are warned about, not silently dropped."""

    def __init__(self, frame_shape, **cpu_settings):
        from openpiv.settings import PIVSettings
        from openpiv.pyprocess import get_rect_coordinates

        from ._openpiv_speedups import apply_speedups
        apply_speedups()

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

        self._settings = settings
        self.scaling_par = 1.0
        self.coords = get_rect_coordinates(frame_shape, settings.windowsizes[-1], settings.overlap[-1])
        self.val_locations = None

    def __call__(self, frame_a, frame_b):
        from openpiv import windef, validation, filters

        settings = self._settings
        frame_a = np.asarray(frame_a, dtype=np.float32)
        frame_b = np.asarray(frame_b, dtype=np.float32)

        # -- pass 0 (coarsest window) --
        x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, settings)
        grid_mask = np.zeros_like(u, dtype=bool)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)

        if settings.validation_first_pass:
            flags = validation.typical_validation(u, v, s2n, settings)
        else:
            flags = np.zeros_like(u, dtype=bool)

        if (settings.num_iterations == 1 and settings.replace_vectors) or settings.num_iterations > 1:
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

        # flags from the final (finest) pass -- True = invalid, same
        # convention as piv_gpu's val_locations
        self.val_locations = np.asarray(flags, dtype=bool)
        return u, v


def init_cpu_processor(frame_shape, cpu_settings):
    process = CPUPIVProcess(frame_shape, **cpu_settings)
    x, y = process.coords
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y
