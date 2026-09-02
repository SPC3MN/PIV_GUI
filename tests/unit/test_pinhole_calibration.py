"""DaVis `PinholeOpenCV` calibration: exact decode, and the two silent-
corruption guards that came with it.

The headline test here (test_real_snapshot_reproduces_davis_fit_error) is a
REAL ground-truth test, not a self-consistency one: it reprojects a real
snapshot's own MarkPositionTable.xml through the decoded model and requires
the result to match DaVis's own independently-stored <FitError RMS>. Nothing
in the app produces that number -- DaVis wrote it -- so the test cannot pass
by agreeing with our own assumptions.

That distinction matters here specifically. The previous stereo-angle
estimator shipped with unit tests that built synthetic calibration data using
the same linear formula the function then inverted; they passed for the whole
life of a function that was ~10 degrees wrong on every real rig. Tests that
can only confirm their own premise are worse than no tests, because they read
as coverage.
"""

import os

import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import (build_camera_mapping, build_stereo_cameras,
                                                  stereo_view_angles)
from piv_suite.calibration.pinhole import (
    PinholeCameraMapping, euler_zyx, read_marks, read_pinhole_camera,
    weighted_reprojection_rms,
)
from piv_suite.config.schema import PinholeMappingSettings
from piv_suite.io.davis_set import read_stereo_calibration_from_set

# Real reference project. Read-only; skipped when it isn't mounted.
_REAL_PROJECT = r"D:\messy_data\Full_Tank(2026)"
_REAL_HISTORY = os.path.join(_REAL_PROJECT, "Properties", "Calibration History")
_PINHOLE_SNAPSHOTS = ("Calibration_260323_161023", "Calibration_260301_195254",
                      "Calibration_260303_162806")
_CORRECTION_FIELD_SNAPSHOT = "Calibration_260310_140344"

_has_real = pytest.mark.skipif(
    not os.path.isdir(_REAL_HISTORY),
    reason="real DaVis reference project not mounted on this machine")


def _snapshot(name):
    return os.path.join(_REAL_HISTORY, name)


# --------------------------------------------------------- exact decode --

@_has_real
@pytest.mark.parametrize("snapshot", _PINHOLE_SNAPSHOTS)
@pytest.mark.parametrize("camera", (1, 2))
def test_real_snapshot_reproduces_davis_fit_error(snapshot, camera):
    """The decode is exact when the stored parameters, used verbatim,
    reproduce DaVis's own <FitError RMS> against DaVis's own marks."""
    snap = _snapshot(snapshot)
    cam = read_pinhole_camera(os.path.join(snap, "Calibration.xml"), str(camera))
    assert cam is not None
    marks = read_marks(os.path.join(snap, "camera%d" % camera, "MarkPositionTable.xml"), camera)
    assert marks is not None and len(marks[0]) > 100
    rms = weighted_reprojection_rms(cam, marks)
    # 1e-6 relative is far looser than what is actually achieved (~1e-14);
    # it is set here to catch a real model regression, not float noise.
    assert rms == pytest.approx(cam.fit_rms, rel=1e-6)


@_has_real
def test_mark_table_must_be_selected_by_camera_number_not_folder():
    """DaVis writes the SAME mark table into camera1\\ and camera2\\, each
    containing BOTH cameras' blocks. Selecting by folder silently pairs one
    camera's marks with the other's parameters -- the mistake that made this
    model look undecodable. Both folders must yield identical marks for a
    given CameraNumber, and camera 1's marks must NOT fit camera 2."""
    snap = _snapshot(_PINHOLE_SNAPSHOTS[0])
    from_cam1_folder = read_marks(os.path.join(snap, "camera1", "MarkPositionTable.xml"), 2)
    from_cam2_folder = read_marks(os.path.join(snap, "camera2", "MarkPositionTable.xml"), 2)
    assert np.array_equal(from_cam1_folder[0], from_cam2_folder[0])

    cam1 = read_pinhole_camera(os.path.join(snap, "Calibration.xml"), "1")
    wrong_marks = read_marks(os.path.join(snap, "camera1", "MarkPositionTable.xml"), 2)
    wrong = weighted_reprojection_rms(cam1, wrong_marks)
    assert wrong > 10 * cam1.fit_rms


@_has_real
def test_focal_length_is_millimetres_not_pixels():
    """FocalLengthPixel is stored in mm despite its name. Reading it as
    pixels is off by 1/SensorPixelSizeMm (~365x) -- assert the magnitude so
    a future 'fix' to the obvious reading fails loudly."""
    cam = read_pinhole_camera(
        os.path.join(_snapshot(_PINHOLE_SNAPSHOTS[0]), "Calibration.xml"), "1")
    assert cam.f_px == pytest.approx(75.95150936765468 / 0.0027400000000000002, rel=1e-12)
    assert cam.f_px > 20000


@_has_real
def test_camera_centres_are_physically_separated():
    """A sanity check with real content: the two camera centres must sit on
    opposite sides of the field with a metre-scale baseline, not on top of
    each other (which a rotation-convention error produces)."""
    cx = os.path.join(_snapshot(_PINHOLE_SNAPSHOTS[0]), "Calibration.xml")
    c0 = read_pinhole_camera(cx, "1").centre
    c1 = read_pinhole_camera(cx, "2").centre
    assert c0[0] < 0 < c1[0]
    assert np.linalg.norm(c1 - c0) > 1000.0


# --------------------------------------------------- per-pixel geometry --

@_has_real
def test_view_angles_match_the_stored_extrinsic_rotation():
    """The derived mean viewing angle must agree with the calibration's own
    RotationAngles/Ry, which the derivation never reads."""
    cx = os.path.join(_snapshot(_PINHOLE_SNAPSHOTS[0]), "Calibration.xml")
    c0, c1 = read_pinhole_camera(cx, "1"), read_pinhole_camera(cx, "2")
    ny, nx = c0.corrected_wh[1], c0.corrected_wh[0]
    x, y = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    a0, _ = c0.view_angles(x, y)
    a1, _ = c1.view_angles(x, y)
    assert a0.mean() == pytest.approx(np.degrees(-0.7882908866115399), abs=2.5)
    assert a1.mean() == pytest.approx(np.degrees(0.78923816442476491), abs=2.5)
    # DaVis's own calibration UI reports "Min/Max angle 1-2: 89.53 deg".
    assert (a1.mean() - a0.mean()) == pytest.approx(89.53, abs=2.0)


@_has_real
def test_view_angles_vary_across_the_field_of_view():
    """The point of deriving angles per pixel: they are NOT constant. If this
    spread ever collapses, a scalar would have been adequate and this whole
    mechanism is unnecessary -- so assert the premise directly."""
    cx = os.path.join(_snapshot(_PINHOLE_SNAPSHOTS[0]), "Calibration.xml")
    cam = read_pinhole_camera(cx, "1")
    ny, nx = cam.corrected_wh[1], cam.corrected_wh[0]
    x, y = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    alpha, beta = cam.view_angles(x, y)
    assert np.ptp(alpha) > 5.0
    assert np.ptp(beta) > 5.0


def test_view_angles_convention_matches_reconstruct_stereo():
    """view_angles must return the angle reconstruct_stereo actually solves
    with (dx = dX - dZ*tan(alpha)). Built from an explicit camera centre, so
    the expected value comes from geometry rather than from the code."""
    # Camera on the -Z side of the sheet -- a real rig configuration, and the
    # one that used to come back 180-deg wrapped.
    cam = PinholeCameraMapping(
        f_px=10000.0, cx=0.0, cy=0.0, k1=0.0, k2=0.0, p1=0.0, p2=0.0,
        R=np.eye(3), T=[0.0, 0.0, 1000.0],
        scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0, z_mm=0.0)
    # centre = -R.T @ T = (0, 0, -1000): directly behind the plane.
    alpha, beta = cam.view_angles(np.array([0.0]), np.array([0.0]))
    assert alpha == pytest.approx(0.0, abs=1e-9)
    assert beta == pytest.approx(0.0, abs=1e-9)
    # A point offset in +X sees the camera at a real angle: the centre is at
    # x=0, z=-1000, the point at x=+100, z=0 -> tan = (0-100)/(-1000-0).
    alpha, _ = cam.view_angles(np.array([100.0]), np.array([0.0]))
    assert np.tan(np.radians(alpha[0])) == pytest.approx(0.1, rel=1e-9)


def test_euler_zyx_composes_in_the_documented_order():
    rx, ry, rz = 0.1, -0.2, 0.3
    R = euler_zyx(rx, ry, rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    expect = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
              @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
              @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
    assert np.allclose(R, expect)


# ------------------------------------------- integration with the schema --

@_has_real
def test_read_stereo_calibration_returns_pinhole_settings_and_no_scalar_angles():
    settings = read_stereo_calibration_from_set(os.path.join(_REAL_PROJECT, "dt_opt.set"))
    assert isinstance(settings.cam0_pinhole, PinholeMappingSettings)
    assert isinstance(settings.cam1_pinhole, PinholeMappingSettings)
    # The polynomial fields stay at their defaults for a pinhole project.
    assert settings.cam0_mapping_plane2 is None
    # All four angles stay None: they are derived per pixel downstream, and a
    # value here would silently override that with a worse approximation.
    assert (settings.alpha1_deg, settings.alpha2_deg,
            settings.beta1_deg, settings.beta2_deg) == (None, None, None, None)


@_has_real
def test_pinhole_settings_round_trip_through_build_camera_mapping():
    settings = read_stereo_calibration_from_set(os.path.join(_REAL_PROJECT, "dt_opt.set"))
    cam = build_camera_mapping(settings.cam0_mapping, None, settings.sheet_z_mm,
                               settings.cam0_pinhole, settings.world_scale_px_per_mm)
    # dt_opt was recorded at 18:08, after the CURRENT calibration was written
    # at 17:51 by a self-calibration run -- so that, not the newest History
    # entry (16:10), is the one in effect. See _select_calibration_snapshot.
    direct = read_pinhole_camera(
        os.path.join(_REAL_PROJECT, "Properties", "Calibration", "Calibration.xml"), "1")
    x, y = np.meshgrid(np.linspace(0, 5000, 5), np.linspace(0, 3000, 5))
    assert np.allclose(cam.world_to_raw(x, y), direct.world_to_raw(x, y))


@_has_real
def test_correction_field_polynomial_snapshot_is_refused_not_silently_used():
    """A polynomial snapshot with a self-calibration correction field decodes
    its base layer without error but is 16-32 px wrong. Refusing is correct;
    silently returning it is the failure mode this guards."""
    assert os.path.isdir(os.path.join(_snapshot(_CORRECTION_FIELD_SNAPSHOT), "Correction field"))
    with pytest.raises(NotImplementedError, match="[Cc]orrection"):
        read_stereo_calibration_from_set(
            os.path.join(_REAL_PROJECT, "Initial_Test", "Self2_04.set"))


def test_project_round_trips_pinhole_settings_through_json(tmp_path):
    """A pinhole project must survive save_project/load_project.

    Two real bugs this covers, both invisible until a project is actually
    saved: PinholeMappingSettings must come back as a dataclass rather than
    the bare dict json gives us (config.io.from_dict rebuilds only the
    polynomial keys unless told otherwise), and every field must be a plain
    float -- the extrinsics come off a numpy array, and json has no encoder
    for np.float64, so saving raised."""
    from piv_suite.config import io as config_io
    from piv_suite.config.schema import ProjectConfig

    cfg = ProjectConfig()
    cfg.project.mode = "stereo"
    cfg.stereo.cam0_pinhole = PinholeMappingSettings(
        f_px=27719.5, cx=3066.4, cy=1221.1, k1=0.173, k2=-10.64,
        p1=6.2e-4, p2=2.4e-3, rx=3.125, ry=-0.788, rz=0.0147,
        tx=np.float64(-60.49), ty=np.float64(12.35), tz=np.float64(1693.11),
        scale_x=0.0609, scale_y=-0.0609, offset_x=-162.42, offset_y=86.40,
        name="cam0", raw_width=4096, raw_height=3008, fit_rms=0.4259)
    cfg.stereo.world_shape = (3027, 5628)

    path = str(tmp_path / "p.pivproj")
    config_io.save_project(path, cfg)
    back = config_io.load_project(path)

    assert isinstance(back.stereo.cam0_pinhole, PinholeMappingSettings)
    assert back.stereo.cam0_pinhole.f_px == pytest.approx(27719.5)
    assert back.stereo.cam0_pinhole.tx == pytest.approx(-60.49)
    assert back.stereo.cam1_pinhole is None
    assert back.stereo.world_shape == (3027, 5628)


def test_loading_a_project_saved_before_pinhole_existed_still_works(tmp_path):
    """An older .pivproj has no cam*_pinhole keys at all -- it must load and
    keep working on the polynomial path."""
    import json

    from piv_suite.config import io as config_io

    legacy = {
        "project": {"mode": "stereo"},
        "stereo": {"cam0_mapping": {"x0": 1.0, "x_span": 2.0},
                   "world_shape": [100, 200], "beta1_deg": 0.0},
    }
    path = str(tmp_path / "legacy.pivproj")
    with open(path, "w") as fh:
        json.dump(legacy, fh)
    back = config_io.load_project(path)
    assert back.stereo.cam0_pinhole is None
    assert back.stereo.cam0_mapping.x0 == pytest.approx(1.0)
    assert back.stereo.world_shape == (100, 200)


def test_stereo_view_angles_passes_overrides_through_unchanged():
    """stereo_view_angles applies no sign convention of its own: both models
    already answer in the canvas frame, so an override is used as typed. An
    earlier version negated beta here, which was right for the pinhole model
    and silently wrong for the polynomial one."""
    cam = PinholeCameraMapping(
        f_px=10000.0, cx=0.0, cy=0.0, k1=0.0, k2=0.0, p1=0.0, p2=0.0,
        R=np.eye(3), T=[0.0, 0.0, 1000.0],
        scale_x=1.0, scale_y=-1.0, offset_x=0.0, offset_y=0.0, z_mm=0.0)
    x = np.zeros((2, 2))
    a1, a2, b1, b2 = stereo_view_angles(cam, cam, x, x, (11.0, -12.0, 13.0, -14.0))
    assert np.degrees(a1) == pytest.approx(11.0)
    assert np.degrees(a2) == pytest.approx(-12.0)
    assert np.degrees(b1) == pytest.approx(13.0)
    assert np.degrees(b2) == pytest.approx(-14.0)


@_has_real
@pytest.mark.parametrize("camera", ("1", "2"))
def test_view_angles_predict_the_real_apparent_shift_of_an_out_of_plane_particle(camera):
    """THE contract, tested directly rather than by convention.

    reconstruct_stereo solves dx = dX - dZ*tan(alpha) and dy = dY -
    dZ*tan(beta). So for a particle that moves ONLY in Z, the apparent
    in-plane shift of its image, expressed on the dewarp canvas in mm, must
    equal -dZ*tan(angle) -- for BOTH components, with the signs that come out.

    Nothing here assumes a frame: the apparent shift is measured by actually
    projecting the displaced particle and finding which canvas point lands on
    the same sensor pixel. A sign error anywhere in view_angles fails this.
    """
    from scipy.optimize import fsolve

    cam = read_pinhole_camera(
        os.path.join(_snapshot(_PINHOLE_SNAPSHOTS[0]), "Calibration.xml"), camera)
    dz = 0.75  # mm out of the light sheet
    for xp, yp in ((1500.0, 900.0), (4000.0, 2200.0), (2800.0, 1500.0)):
        X, Y = cam.canvas_to_world_mm(xp, yp)
        target = cam.project(np.array([X, Y, dz]))

        def residual(c):
            return cam.project(np.array([*cam.canvas_to_world_mm(c[0], c[1]), 0.0])) - target

        xp2, yp2 = fsolve(residual, [xp, yp], full_output=False)
        # Apparent shift on the canvas, in mm, using each axis's own scale.
        dx_mm = (xp2 - xp) * abs(cam.scale_x)
        dy_mm = (yp2 - yp) * abs(cam.scale_y)
        alpha, beta = cam.view_angles(np.array([xp]), np.array([yp]))
        assert dx_mm == pytest.approx(-dz * np.tan(np.radians(alpha[0])), abs=2e-3)
        assert dy_mm == pytest.approx(-dz * np.tan(np.radians(beta[0])), abs=2e-3)


@_has_real
def test_polynomial_and_pinhole_models_agree_on_the_viewing_angles():
    """THE cross-model invariant, and the one that would have caught the
    double-negation bug: two entirely independent decodes of the same physical
    rig must return the same angles, in the same frame.

    The two calibration snapshots are 13 days apart and the cameras were
    physically moved between them, so this is a convention/frame check to a
    couple of degrees, not a numerical-equality one. A SIGN error shows up
    enormously against that tolerance; a genuine rig change does not."""
    from piv_suite.io.davis_set import _exact_camera_mapping_from_calibration_xml

    poly_xml = os.path.join(_snapshot("Calibration_260309_161814"), "Calibration.xml")
    pin_xml = os.path.join(_snapshot("Calibration_260323_161023"), "Calibration.xml")

    for ident in ("1", "2"):
        planes, ppm, wh = _exact_camera_mapping_from_calibration_xml(poly_xml, ident)
        poly_cam = build_camera_mapping(planes[0], planes[1], sheet_z_mm=-0.5, px_per_mm=ppm)
        pin_cam = read_pinhole_camera(pin_xml, ident)

        # Sample well inside both canvases, in each model's own pixel units.
        for cam, (w, h) in ((poly_cam, wh), (pin_cam, pin_cam.corrected_wh)):
            cam._sample = np.meshgrid(np.linspace(0.2 * w, 0.8 * w, 5),
                                      np.linspace(0.2 * h, 0.8 * h, 5))
        a_poly, b_poly = poly_cam.view_angles(*poly_cam._sample)
        a_pin, b_pin = pin_cam.view_angles(*pin_cam._sample)

        assert a_poly.mean() == pytest.approx(a_pin.mean(), abs=3.0)
        assert b_poly.mean() == pytest.approx(b_pin.mean(), abs=3.0)
        # Beta's mean is near zero, so a mean-only check would pass even with
        # the sign inverted. Compare the top-to-bottom TREND, which does not.
        assert np.sign(b_poly[-1].mean() - b_poly[0].mean()) == \
            np.sign(b_pin[-1].mean() - b_pin[0].mean())


@_has_real
def test_stereo_view_angles_on_a_real_project_are_arrays_not_scalars():
    settings = read_stereo_calibration_from_set(os.path.join(_REAL_PROJECT, "dt_opt.set"))
    cam0 = build_camera_mapping(settings.cam0_mapping, None, settings.sheet_z_mm,
                                settings.cam0_pinhole, settings.world_scale_px_per_mm)
    cam1 = build_camera_mapping(settings.cam1_mapping, None, settings.sheet_z_mm,
                                settings.cam1_pinhole, settings.world_scale_px_per_mm)
    x, y = np.meshgrid(np.linspace(0, 5000, 8), np.linspace(0, 3000, 6))
    a1, a2, b1, b2 = stereo_view_angles(cam0, cam1, x, y)
    for a in (a1, a2, b1, b2):
        assert a.shape == x.shape
    assert np.ptp(a1) > np.radians(5.0)


@_has_real
def test_calibration_panel_round_trips_the_pinhole_calibration(qtbot):
    """The GUI panel must carry cam*_pinhole through set_settings ->
    get_settings.

    Every processing entry point builds its config from this panel
    (run_panel, preview_panel), so dropping the pinhole settings here made the
    whole PinholeOpenCV path unreachable from the GUI -- while the status bar
    reported a successful extraction. The failure then surfaced much later,
    inside view_angles, as "supply the angles manually": the exact opposite of
    what the user had just been told."""
    from piv_suite_gui.widgets.calibration_panel import CalibrationPanel

    extracted = read_stereo_calibration_from_set(os.path.join(_REAL_PROJECT, "dt_opt.set"))
    panel = CalibrationPanel()
    qtbot.addWidget(panel)
    panel.set_settings(extracted)
    got = panel.get_settings()

    assert got.cam0_pinhole is not None and got.cam1_pinhole is not None
    assert got.cam0_pinhole.f_px == pytest.approx(extracted.cam0_pinhole.f_px)
    assert got.cam1_pinhole.tz == pytest.approx(extracted.cam1_pinhole.tz)
    assert got.world_shape == extracted.world_shape
    # And the panel must be able to build real cameras from what it hands back.
    cam0, cam1 = build_stereo_cameras(got)
    x, y = np.meshgrid(np.linspace(0, 5000, 4), np.linspace(0, 3000, 4))
    a1, a2, b1, b2 = stereo_view_angles(cam0, cam1, x, y,
                                        (got.alpha1_deg, got.alpha2_deg,
                                         got.beta1_deg, got.beta2_deg))
    assert np.ptp(np.degrees(a1)) > 3.0


def test_calibration_panel_reports_which_model_it_holds(qtbot):
    from piv_suite_gui.widgets.calibration_panel import CalibrationPanel
    from piv_suite.config.schema import StereoSettings

    panel = CalibrationPanel()
    qtbot.addWidget(panel)
    panel.set_settings(StereoSettings())
    assert "polynomial" in panel.model_label.text()
    panel.set_settings(StereoSettings(cam0_pinhole=PinholeMappingSettings(f_px=1.0)))
    assert "Pinhole" in panel.model_label.text()
