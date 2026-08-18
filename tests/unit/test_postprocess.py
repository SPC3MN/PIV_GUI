import numpy as np
import pytest

from piv_suite.processing.postprocess import (
    apply_calibration, global_outlier_mask, range_filter, replace_invalid_vectors,
    smooth_vector_field,
)


def test_global_outlier_mask_none_disables():
    u = np.array([1.0, 2.0, 100.0])
    v = np.zeros(3)
    mask = global_outlier_mask(u, v, None)
    assert not mask.any()


def test_global_outlier_mask_flags_known_spike():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, size=1000)
    v = rng.normal(0, 1, size=1000)
    u[500] = 1000.0  # obvious spike
    mask = global_outlier_mask(u, v, n_std=4.0)
    assert mask[500]
    # most of the well-behaved normal samples should NOT be flagged
    assert mask.sum() < 10


def test_range_filter_u_range():
    u = np.array([-5.0, 0.0, 5.0, 50.0])
    v = np.zeros(4)
    invalid = range_filter(u, v, u_range=(-10, 10))
    np.testing.assert_array_equal(invalid, [False, False, False, True])


def test_range_filter_magnitude_range():
    u = np.array([3.0, 0.0])
    v = np.array([4.0, 0.0])  # magnitude 5.0 and 0.0
    invalid = range_filter(u, v, magnitude_range=(1.0, 10.0))
    np.testing.assert_array_equal(invalid, [False, True])


def test_range_filter_residual_flags_spatial_spike():
    ny, nx = 9, 9
    u = np.ones((ny, nx)) * 2.0
    v = np.zeros((ny, nx))
    u[4, 4] = 50.0  # a single vector wildly different from its neighbors
    invalid = range_filter(u, v, residual_max=5.0, neighborhood_size=3)
    assert invalid[4, 4]
    assert invalid.sum() == 1


def test_range_filter_residual_requires_2d():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        range_filter(u, v, residual_max=1.0)


def test_range_filter_no_bounds_rejects_nothing():
    u = np.array([1.0, 1000.0])
    v = np.array([1.0, 1000.0])
    invalid = range_filter(u, v)
    assert not invalid.any()


def test_replace_invalid_vectors_interpolates():
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    u = x.copy()
    v = y.copy()
    valid = np.ones_like(u, dtype=bool)
    valid[2, 2] = False
    u_out, v_out = replace_invalid_vectors(x.ravel(), y.ravel(), u.ravel(), v.ravel(), valid.ravel())
    # linear interpolation of a linear field should recover the true value
    assert np.isclose(u_out.reshape(5, 5)[2, 2], 2.0, atol=0.5)


def test_replace_invalid_vectors_no_crash_when_all_rejected():
    # Regression test: an overly strict filter combo (e.g. range_filter +
    # global_outlier_std) can reject every vector in a pair -- griddata
    # crashes with "No points given" if there's nothing left to interpolate
    # from; this must degrade gracefully (stay NaN) instead of crashing the
    # whole batch run.
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    u = np.full((5, 5), np.nan)
    v = np.full((5, 5), np.nan)
    valid = np.zeros((5, 5), dtype=bool)
    u_out, v_out = replace_invalid_vectors(x.ravel(), y.ravel(), u.ravel(), v.ravel(), valid.ravel())
    assert np.isnan(u_out).all()
    assert np.isnan(v_out).all()


def test_replace_invalid_vectors_noop_when_all_valid():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    valid = np.ones(3, dtype=bool)
    u_out, v_out = replace_invalid_vectors(x, y, u, v, valid)
    np.testing.assert_array_equal(u_out, u)
    np.testing.assert_array_equal(v_out, v)


def test_smooth_vector_field_preserves_shape_and_handles_nan():
    u = np.ones((10, 10))
    v = np.zeros((10, 10))
    u[5, 5] = np.nan
    u_s, v_s = smooth_vector_field(u, v, sigma=1.0)
    assert u_s.shape == u.shape
    assert not np.isnan(u_s).any()


def test_apply_calibration_noop_without_both_params():
    u = np.array([1.0])
    v = np.array([1.0])
    u_out, v_out = apply_calibration(u, v, pixel_pitch_mm=None, frame_dt_s=0.001)
    np.testing.assert_array_equal(u_out, u)


def test_apply_calibration_scales_correctly():
    u = np.array([10.0])  # px/frame
    v = np.array([0.0])
    # 0.01 mm/px, 0.002 s/frame -> scale = (0.01/1000) / 0.002 = 0.005 m/s per px/frame
    u_out, v_out = apply_calibration(u, v, pixel_pitch_mm=0.01, frame_dt_s=0.002)
    expected_scale = (0.01 / 1000.0) / 0.002
    assert np.isclose(u_out[0], 10.0 * expected_scale)
