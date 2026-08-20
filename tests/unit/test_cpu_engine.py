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
