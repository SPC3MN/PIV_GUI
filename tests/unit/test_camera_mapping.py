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
