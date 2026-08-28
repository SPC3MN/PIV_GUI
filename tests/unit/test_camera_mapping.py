import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import (
    CameraMapping, build_camera_mapping, interpolate_camera_mapping,
)
from piv_suite.config.schema import CameraMappingSettings


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


def _plane(z_mm, x0=0.0, x_span=100.0, y0=0.0, y_span=100.0, coef_scale=1.0):
    coefs_a = {k: coef_scale * v for k, v in
               zip(("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s"),
                   (1.0, 2.0, 0.5, 0.1, -0.5, 0.2, 0.05, 0.3, 0.02, -0.02))}
    coefs_b = {k: v * 0.5 for k, v in coefs_a.items()}
    return CameraMappingSettings(x0=x0, x_span=x_span, y0=y0, y_span=y_span,
                                  dx_coefs=coefs_a, dy_coefs=coefs_b, z_mm=z_mm)


def test_build_camera_mapping_single_plane_matches_direct_construction():
    plane = _plane(z_mm=None, x0=10.0, x_span=50.0, y0=5.0, y_span=40.0)
    cm = build_camera_mapping(plane)
    assert cm.x0 == plane.x0 and cm.x_span == plane.x_span
    assert cm.dx_coefs == plane.dx_coefs


def test_build_camera_mapping_two_planes_requires_sheet_z_mm():
    p1, p2 = _plane(z_mm=1.0), _plane(z_mm=-2.0, coef_scale=2.0)
    with pytest.raises(ValueError):
        build_camera_mapping(p1, p2, sheet_z_mm=None)


def test_interpolate_camera_mapping_reduces_exactly_at_each_planes_own_z():
    p1, p2 = _plane(z_mm=1.0), _plane(z_mm=-2.0, coef_scale=2.0)
    cm_at_p1 = interpolate_camera_mapping(p1, p2, sheet_z_mm=1.0)
    assert cm_at_p1.x0 == p1.x0
    assert cm_at_p1.dx_coefs == p1.dx_coefs
    cm_at_p2 = interpolate_camera_mapping(p1, p2, sheet_z_mm=-2.0)
    assert cm_at_p2.dx_coefs == p2.dx_coefs


def test_interpolate_camera_mapping_midpoint_is_arithmetic_mean():
    p1, p2 = _plane(z_mm=0.0), _plane(z_mm=10.0, coef_scale=3.0)
    cm_mid = interpolate_camera_mapping(p1, p2, sheet_z_mm=5.0)
    for k in p1.dx_coefs:
        assert cm_mid.dx_coefs[k] == pytest.approx((p1.dx_coefs[k] + p2.dx_coefs[k]) / 2)
        assert cm_mid.dy_coefs[k] == pytest.approx((p1.dy_coefs[k] + p2.dy_coefs[k]) / 2)


def test_build_camera_mapping_interpolated_dewarp_is_sane():
    p1, p2 = _plane(z_mm=1.0, x0=50.0, x_span=100.0, y0=50.0, y_span=100.0), \
        _plane(z_mm=-2.0, x0=50.0, x_span=100.0, y0=50.0, y_span=100.0, coef_scale=1.5)
    cm = build_camera_mapping(p1, p2, sheet_z_mm=-0.5)
    raw = np.random.RandomState(0).rand(150, 150).astype(np.float32)
    dewarped = cm.dewarp_image(raw, (100, 100), order=1)
    assert dewarped.shape == (100, 100)
    assert not np.isnan(dewarped).any()
    assert np.any(dewarped > 0)


# ---- raw_domain_valid: geometric FOV mask (real raw sensor bounds) ----

def test_raw_domain_valid_true_inside_false_outside_sensor_bounds():
    # Identity mapping (zero coefs) -- world_to_raw(x, y) == (x, y), so
    # raw_domain_valid's own bounds check is exercised directly against a
    # known raw_width/raw_height with no polynomial distortion in the way.
    cm = CameraMapping(x0=50.0, x_span=100.0, y0=50.0, y_span=100.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs(),
                        raw_width=100, raw_height=80)
    x = np.array([10.0, 99.0, 100.0, 150.0, -5.0])
    y = np.array([10.0, 79.0, 80.0, 10.0, 10.0])
    # (10,10) and (99,79) are inside [0,100)x[0,80); (100,80) is exactly
    # on the exclusive upper bound (outside); (150,10) and (-5,10) are
    # outside in x either direction.
    expected = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(cm.raw_domain_valid(x, y), expected)


def test_raw_domain_valid_all_true_when_raw_size_unknown():
    # raw_width/raw_height default to None (no OriginalImageSize data --
    # the marks-fit calibration path, or any hand-built CameraMapping that
    # doesn't pass them, e.g. every other test in this file) -- must mean
    # "no masking possible", not "everything is out of view".
    cm = CameraMapping(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs())
    x = np.array([-1000.0, 0.0, 1e6])
    y = np.array([-1000.0, 0.0, 1e6])
    assert cm.raw_domain_valid(x, y).all()


def test_raw_domain_valid_all_true_when_raw_size_zero():
    # Same "unknown" contract as None, via config.schema.
    # CameraMappingSettings' own 0-default (not every caller passes None
    # explicitly -- build_camera_mapping always forwards settings.raw_
    # width/raw_height, which default to 0, not None).
    cm = CameraMapping(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                        dx_coefs=_zero_coefs(), dy_coefs=_zero_coefs(),
                        raw_width=0, raw_height=0)
    x = np.array([-1000.0, 1e6])
    y = np.array([-1000.0, 1e6])
    assert cm.raw_domain_valid(x, y).all()


def test_build_camera_mapping_threads_raw_width_height_single_plane():
    plane = _plane(z_mm=None, x0=10.0, x_span=50.0, y0=5.0, y_span=40.0)
    plane.raw_width, plane.raw_height = 4096, 3008
    cm = build_camera_mapping(plane)
    assert cm.raw_width == 4096
    assert cm.raw_height == 3008


def test_build_camera_mapping_threads_raw_width_height_two_planes():
    # raw_width/raw_height is a fixed per-camera constant (see
    # interpolate_camera_mapping's own comment) -- both planes carry the
    # SAME real value here, matching how _exact_camera_mapping_from_
    # calibration_xml actually populates them (one OriginalImageSize per
    # camera, stamped onto every plane).
    p1, p2 = _plane(z_mm=1.0), _plane(z_mm=-2.0, coef_scale=2.0)
    p1.raw_width = p2.raw_width = 4096
    p1.raw_height = p2.raw_height = 3008
    cm = build_camera_mapping(p1, p2, sheet_z_mm=-0.5)
    assert cm.raw_width == 4096
    assert cm.raw_height == 3008
