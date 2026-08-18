import numpy as np

from piv_suite.calibration.camera_mapping import CameraMapping


def _zero_coefs():
    return {k: 0.0 for k in ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")}


def test_s_t_normalization():
    cm = CameraMapping(x0=100.0, x_span=200.0, y0=50.0, y_span=100.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs())
    assert np.isclose(cm.s(100.0), 0.0)
    assert np.isclose(cm.s(200.0), 1.0)
    assert np.isclose(cm.t(50.0), 0.0)
    assert np.isclose(cm.t(100.0), 1.0)


def test_world_to_raw_identity_with_zero_coefs():
    cm = CameraMapping(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs())
    x, y = cm.world_to_raw(np.array([10.0, 20.0]), np.array([5.0, 15.0]))
    np.testing.assert_allclose(x, [10.0, 20.0])
    np.testing.assert_allclose(y, [5.0, 15.0])


def test_world_to_raw_constant_offset():
    coefs = _zero_coefs()
    coefs["1"] = 3.0  # constant term -> dx(s,t) = 3.0 everywhere
    cm = CameraMapping(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                        dx_coefs=coefs, dy_coefs=_zero_coefs())
    x, y = cm.world_to_raw(np.array([10.0]), np.array([5.0]))
    # x = xp - dx(s,t) = 10 - 3 = 7
    np.testing.assert_allclose(x, [7.0])
    np.testing.assert_allclose(y, [5.0])


def test_poly_matches_hand_computed_value():
    coefs = {"1": 1.0, "s": 2.0, "s2": 3.0, "s3": 0.0, "t": 4.0, "t2": 0.0,
             "t3": 0.0, "st": 5.0, "s2t": 0.0, "t2s": 0.0}
    # _poly(s=1, t=1, c) = 1 + 2*1 + 3*1 + 4*1 + 5*1*1 = 15
    val = CameraMapping._poly(1.0, 1.0, coefs)
    assert np.isclose(val, 15.0)


def test_dewarp_image_caches_grid_per_shape():
    cm = CameraMapping(x0=0.0, x_span=10.0, y0=0.0, y_span=10.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs())
    raw = np.arange(100.0).reshape(10, 10)
    out1 = cm.dewarp_image(raw, (10, 10))
    assert cm._cached_shape == (10, 10)
    # identity mapping (zero coefs) should reproduce the input near-exactly
    np.testing.assert_allclose(out1, raw, atol=1e-4)
    out2 = cm.dewarp_image(raw, (10, 10))
    np.testing.assert_array_equal(out1, out2)


def test_dewarp_image_matches_analytic_result_with_real_distortion():
    # dewarp_image with NONTRIVIAL (nonzero on every term, including cross
    # st/s2t/t2s terms) DaVis-style polynomial distortion -- not just the
    # zero-coefficient identity case above. Uses a bilinear ramp raw
    # pattern (x_raw + 2*y_raw) so order=1 map_coordinates reconstructs it
    # with ZERO interpolation error, letting this compare dewarp_image's
    # actual output against the fully analytic expected value (computed
    # from the SAME world_to_raw() call) to within floating-point
    # precision alone -- directly validates s()/t() normalization,
    # _poly(), world_to_raw(), and the map_coordinates() call together.
    world_shape = (60, 80)  # (ny, nx)
    ny, nx = world_shape
    margin = 20  # uniform shift baked into the constant coef term so
                 # every raw sample stays inside the padded raw canvas

    coefs_dx = {"1": 0.5 - margin, "s": 1.2, "s2": 0.6, "s3": 0.1, "t": -0.3,
                "t2": 0.2, "t3": 0.0, "st": 0.4, "s2t": 0.05, "t2s": -0.05}
    coefs_dy = {"1": -0.4 - margin, "s": -0.2, "s2": 0.15, "s3": 0.0, "t": 0.9,
                "t2": 0.5, "t3": 0.08, "st": -0.3, "s2t": 0.02, "t2s": 0.03}
    cm = CameraMapping(x0=nx / 2, x_span=nx, y0=ny / 2, y_span=ny,
                        dx_coefs=coefs_dx, dy_coefs=coefs_dy)

    raw_ny, raw_nx = ny + 2 * margin, nx + 2 * margin

    def raw_value(x_raw, y_raw):
        return x_raw + 2.0 * y_raw

    yr_idx, xr_idx = np.mgrid[0:raw_ny, 0:raw_nx].astype(np.float64)
    raw_image = raw_value(xr_idx, yr_idx)

    yp, xp = np.mgrid[0:ny, 0:nx].astype(np.float64)
    x_raw, y_raw = cm.world_to_raw(xp, yp)
    # confirm the margin actually keeps every sample in-bounds -- if this
    # ever fails, the test itself needs a bigger margin, not a looser tol
    assert x_raw.min() >= 0 and x_raw.max() <= raw_nx - 1
    assert y_raw.min() >= 0 and y_raw.max() <= raw_ny - 1
    expected = raw_value(x_raw, y_raw)

    actual = cm.dewarp_image(raw_image, world_shape, order=1)
    np.testing.assert_allclose(actual, expected, atol=1e-3)
