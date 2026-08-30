import warnings

import numpy as np

from piv_suite.engines.cpu_engine import CPUPIVProcess, _fill_residual_nan


def test_fill_residual_nan_replaces_nan_with_valid_median():
    # Regression test: confirmed on a real image pair (Windows/CUDA
    # hardware) that openpiv's own filters.replace_outliers can leave
    # residual NaN when a literal-NaN cell (flagged by
    # loose_typical_validation -- not a value-based rejection, see
    # cpu_engine.py's module docstring) sits at the frame edge with no
    # far-side valid neighbor to converge from -- e.g. a no-signal band
    # outside the laser sheet. Left as NaN, that poisons the NEXT pass's
    # RectBivariateSpline fit entirely (the whole interpolated surface
    # goes NaN, not just the bad region), which then crashes
    # scipy.ndimage.map_coordinates with a Windows access violation
    # instead of a Python exception.
    u = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 5.0]])
    v = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, np.nan]])

    filled_u, filled_v = _fill_residual_nan(u, v)

    assert not np.isnan(filled_u).any()
    assert not np.isnan(filled_v).any()
    # non-NaN entries are untouched
    assert filled_u[0, 0] == 1.0 and filled_u[0, 1] == 2.0 and filled_u[1, 0] == 3.0 and filled_u[1, 2] == 5.0
    # NaN entries filled with THAT array's own valid-vector median
    assert filled_u[0, 2] == np.nanmedian(u)
    assert filled_u[1, 1] == np.nanmedian(u)
    assert filled_v[1, 2] == np.nanmedian(v)


def test_fill_residual_nan_no_op_when_no_nan():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    filled_u, filled_v = _fill_residual_nan(u, v)
    assert np.array_equal(filled_u, u)
    assert np.array_equal(filled_v, v)


def test_fill_residual_nan_falls_back_to_zero_when_entire_field_is_nan():
    # A WHOLE pass can come back entirely NaN -- confirmed for real once
    # davis_combined's peak-ratio test was wired in (see engines/
    # _openpiv_speedups.py): on genuinely uncorrelated/no-signal image
    # data, EVERY window can legitimately fail. np.nanmedian of an
    # all-NaN array is itself NaN (with a RuntimeWarning, not an
    # exception) -- this used to leave the field entirely NaN right where
    # this function's whole job is to prevent that, which then reached
    # the NEXT pass's spline deformation and crashed with a real Windows
    # access violation (see this function's own docstring). Must fall
    # back to a benign literal 0.0 instead, with no warning surfacing to
    # the caller.
    u = np.full((3, 3), np.nan)
    v = np.full((3, 3), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        filled_u, filled_v = _fill_residual_nan(u, v)

    assert not np.isnan(filled_u).any()
    assert not np.isnan(filled_v).any()
    assert (filled_u == 0.0).all()
    assert (filled_v == 0.0).all()


def test_fill_residual_nan_zero_fallback_is_per_component_independent():
    # Only the array that's ENTIRELY NaN falls back to 0.0 -- the other
    # component's own genuine median is used if it has any real data,
    # confirming the two components are handled independently, not
    # coupled through a single shared fallback decision.
    u = np.full((2, 2), np.nan)
    v = np.array([[1.0, np.nan], [3.0, 5.0]])

    filled_u, filled_v = _fill_residual_nan(u, v)

    assert (filled_u == 0.0).all()
    assert filled_v[0, 1] == np.nanmedian(v)
    assert filled_v[0, 0] == 1.0 and filled_v[1, 0] == 3.0 and filled_v[1, 1] == 5.0


def test_cpu_engine_always_forces_use_vectorized_false():
    # REAL BUG, found and fixed: openpiv.settings.PIVSettings' own default
    # is use_vectorized=True, which nothing in this app ever overrode --
    # meaning every real correlation this app ever ran silently used
    # openpiv's own "vectorized_*" functions (checked and found to differ
    # subtly from this app's carefully verified fast_* reimplementations
    # -- see _openpiv_speedups.py's top docstring) instead of the
    # faithful fast path the rest of this module assumes is active.
    # CPUPIVProcess must force this False unconditionally, including
    # against a caller that tries to pass it through cpu_settings
    # (confirming the force happens AFTER the general cpu_settings
    # override loop, not before it).
    default_engine = CPUPIVProcess((64, 64), windowsizes=[32], overlap=[16])
    assert default_engine._settings.use_vectorized is False

    overridden_engine = CPUPIVProcess((64, 64), windowsizes=[32], overlap=[16], use_vectorized=True)
    assert overridden_engine._settings.use_vectorized is False


def test_fill_residual_nan_handles_masked_arrays():
    # multipass_img_deform/replace_outliers hand back np.ma.MaskedArray
    # in the real pipeline, not plain ndarrays -- confirm the fill
    # operates on the underlying data (mask is a SEPARATE concept, e.g.
    # a static image mask, from the residual-NaN-after-replacement bug
    # this function guards against).
    u = np.ma.masked_array([1.0, np.nan, 3.0], mask=[False, False, False])
    v = np.ma.masked_array([1.0, 2.0, 3.0], mask=[True, False, False])
    filled_u, filled_v = _fill_residual_nan(u, v)
    assert not np.isnan(np.ma.getdata(filled_u)).any()
    assert np.ma.getdata(filled_u)[1] == np.nanmedian([1.0, 3.0])


def test_cpu_engine_val_locations_always_all_false():
    # Validation now lives entirely in processing.postprocess -- no vector
    # should ever be counted invalid because of anything the engine itself
    # decides during calculation (see cpu_engine.py's module docstring).
    # A mostly-random-noise pair is engineered to trigger heavy rejection
    # under openpiv's OLD (real) typical_validation -- confirming
    # val_locations stays all-False even under conditions that used to
    # produce many rejected vectors.
    rng = np.random.RandomState(42)
    size = 128
    frame_a = (rng.rand(size, size) * 255).astype(np.float32)
    frame_b = (rng.rand(size, size) * 255).astype(np.float32)  # uncorrelated with frame_a

    process = CPUPIVProcess(
        frame_a.shape,
        windowsizes=[32, 16],
        overlap=[16, 8],
    )
    u, v = process(frame_a, frame_b)

    assert process.val_locations.shape == u.shape
    assert process.val_locations.dtype == bool
    assert not process.val_locations.any()
