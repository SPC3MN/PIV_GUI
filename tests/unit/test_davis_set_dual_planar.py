"""Tests for davis_set.py's DaVis "SideBySide2D" dual-camera planar
calibration extraction, and pipeline.combine_dual_planar_pair's stitching
math.

Unlike the stereo case (read_stereo_calibration_from_set), there's no
fitting here: RegionWithinCorrectedImage/OriginalImageSize/LinearScaleX/
LinearScaleY are used exactly as DaVis wrote them in Calibration.xml, so
the synthetic tests below just build a Calibration.xml with known values
and check they come back unchanged. The real-data-gated tests lock in the
exact values confirmed by direct inspection of
D:\\Truck_PIV_Round4\\Loaded_CFD_Truck\\Properties\\Calibration\\
Calibration.xml (see SESSION_HANDOFF.md-adjacent design notes) -- if these
ever fail, either the fixture project changed or extraction broke.
"""

import os

import numpy as np
import pytest

from piv_suite.config.schema import DualPlanarSettings
from piv_suite.io.davis_set import (
    _read_corrected_image_size, _read_dual_planar_camera, _read_field_of_view,
    _read_linear_scale, detect_dual_planar_from_set, get_dual_planar_from_set,
    iter_dual_planar_from_set, read_dual_planar_calibration_from_set,
)
from piv_suite.processing.pipeline import combine_dual_planar_pair

REAL_PROJECT_ROOT = r"D:\Truck_PIV_Round4\Loaded_CFD_Truck"
REAL_SET = r"D:\Truck_PIV_Round4\Loaded_CFD_Truck\X_150_mm_Y_0_mm.set"
REAL_SET_2 = r"D:\Truck_PIV_Round4\Loaded_CFD_Truck\X_50_mm_Y_0_mm.set"


# ---- synthetic Calibration.xml fixture (mirrors the real SideBySide2D layout) ----

_SIDE_BY_SIDE_XML = """<?xml version="1.0"?>
<Calibration Version="2" CalibrationIdentifier="test">
    <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
        <CoordinateSystem FieldOfView="SideBySide2D">
            <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1">
                <CorrectedImageSize Width="8009" Height="3046"/>
                <RegionWithinCorrectedImage x="3865" y="4" Width="4144" Height="3041"/>
                <OriginalImageSize Width="4096" Height="3008"/>
                <LinearScaleX FactorMmPerPixel="0.0392775752732" OffsetMm="-167.839561915" Unit="mm" Description=""/>
                <LinearScaleY FactorMmPerPixel="-0.0392775752732" OffsetMm="92.83808624811" Unit="mm" Description=""/>
            </CoordinateMapper>
            <CoordinateMapper CameraIdentifier="2" Type="Polynomial3rdOrder" GroupId="1">
                <CorrectedImageSize Width="8009" Height="3046"/>
                <RegionWithinCorrectedImage x="0" y="0" Width="4111" Height="3018"/>
                <OriginalImageSize Width="4096" Height="3008"/>
                <LinearScaleX FactorMmPerPixel="0.0392775752732" OffsetMm="-167.839561915" Unit="mm" Description=""/>
                <LinearScaleY FactorMmPerPixel="-0.0392775752732" OffsetMm="92.83808624811" Unit="mm" Description=""/>
            </CoordinateMapper>
        </CoordinateSystem>
    </CoordinateSystemsForEachView>
</Calibration>
"""

_PLAIN_PLANAR_XML = """<?xml version="1.0"?>
<Calibration Version="2" CalibrationIdentifier="test">
    <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
        <CoordinateSystem FieldOfView="Planar">
            <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1">
                <CorrectedImageSize Width="4096" Height="3008"/>
            </CoordinateMapper>
        </CoordinateSystem>
    </CoordinateSystemsForEachView>
</Calibration>
"""


def _write_project(tmp_path, xml_text):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    (root / "Properties" / "Calibration" / "Calibration.xml").write_text(xml_text)
    recording = root / "Recording.set"
    recording.write_text('#GROUP Sets\nSetTime = "2026-06-15T11:39:43-04:00";\n')
    return root, recording


# ---- pure XML parsing helpers ----

def test_read_field_of_view_side_by_side(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_SIDE_BY_SIDE_XML)
    assert _read_field_of_view(str(xml_path)) == "SideBySide2D"


def test_read_field_of_view_plain_planar(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_PLAIN_PLANAR_XML)
    assert _read_field_of_view(str(xml_path)) == "Planar"


def test_read_dual_planar_camera_extracts_region_and_raw_size(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_SIDE_BY_SIDE_XML)
    cam0 = _read_dual_planar_camera(str(xml_path), "1")
    assert (cam0.region_x, cam0.region_y) == (3865.0, 4.0)
    assert (cam0.region_width, cam0.region_height) == (4144.0, 3041.0)
    assert (cam0.raw_width, cam0.raw_height) == (4096, 3008)

    cam1 = _read_dual_planar_camera(str(xml_path), "2")
    assert (cam1.region_x, cam1.region_y) == (0.0, 0.0)
    assert (cam1.region_width, cam1.region_height) == (4111.0, 3018.0)


def test_read_dual_planar_camera_missing_camera_raises(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_SIDE_BY_SIDE_XML)
    with pytest.raises(ValueError, match="no CoordinateMapper"):
        _read_dual_planar_camera(str(xml_path), "3")


def test_read_corrected_image_size(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_SIDE_BY_SIDE_XML)
    assert _read_corrected_image_size(str(xml_path), "1") == (8009, 3046)


def test_read_linear_scale_x_and_y(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_SIDE_BY_SIDE_XML)
    sx, ox = _read_linear_scale(str(xml_path), "1", "X")
    sy, oy = _read_linear_scale(str(xml_path), "1", "Y")
    assert sx == pytest.approx(0.0392775752732)
    assert ox == pytest.approx(-167.839561915)
    # Y's factor is the SAME magnitude, NEGATED -- combine_dual_planar_pair
    # relies on this sign, don't "fix" it to positive.
    assert sy == pytest.approx(-sx)
    assert oy == pytest.approx(92.83808624811)


# ---- detect / read, end to end on a synthetic project ----

def test_detect_dual_planar_true_for_side_by_side_project(tmp_path):
    _root, recording = _write_project(tmp_path, _SIDE_BY_SIDE_XML)
    assert detect_dual_planar_from_set(str(recording)) is True


def test_detect_dual_planar_false_for_plain_planar_project(tmp_path):
    _root, recording = _write_project(tmp_path, _PLAIN_PLANAR_XML)
    assert detect_dual_planar_from_set(str(recording)) is False


def test_detect_dual_planar_false_when_no_calibration_at_all(tmp_path):
    recording = tmp_path / "loose_recording.set"
    recording.write_text("")
    assert detect_dual_planar_from_set(str(recording)) is False


def test_read_dual_planar_calibration_from_synthetic_project(tmp_path):
    _root, recording = _write_project(tmp_path, _SIDE_BY_SIDE_XML)
    dp = read_dual_planar_calibration_from_set(str(recording))

    assert dp.enabled is True
    assert (dp.canvas_width, dp.canvas_height) == (8009, 3046)
    assert dp.cam0.region_x == 3865.0
    assert dp.cam1.region_x == 0.0
    assert dp.scale_x_mm_per_px == pytest.approx(0.0392775752732)
    assert dp.scale_y_mm_per_px == pytest.approx(-0.0392775752732)
    # overlap: cam0 starts where cam1's region ends, minus the overlap width
    overlap_lo = dp.cam0.region_x
    overlap_hi = dp.cam1.region_x + dp.cam1.region_width
    assert overlap_hi - overlap_lo == pytest.approx(246.0)


def test_read_dual_planar_calibration_rejects_non_side_by_side(tmp_path):
    _root, recording = _write_project(tmp_path, _PLAIN_PLANAR_XML)
    with pytest.raises(ValueError, match="SideBySide2D"):
        read_dual_planar_calibration_from_set(str(recording))


def test_read_dual_planar_calibration_missing_project_raises(tmp_path):
    recording = tmp_path / "no_calibration_anywhere.set"
    recording.write_text("")
    with pytest.raises(FileNotFoundError):
        read_dual_planar_calibration_from_set(str(recording))


# ---- pipeline.combine_dual_planar_pair: synthetic placement/overlap math ----

def _uniform_camera(u_value, v_value, nx, ny, valid=True):
    """A camera field with a uniform velocity and a regular raw-pixel grid
    -- x/y in the row-down, engine.coords convention combine_dual_planar_
    pair expects."""
    xs = np.linspace(0, 100, nx)
    ys = np.linspace(0, 80, ny)
    x, y = np.meshgrid(xs, ys)
    u = np.full_like(x, u_value, dtype=float)
    v = np.full_like(x, v_value, dtype=float)
    valid_mask = np.full(x.shape, valid, dtype=bool)
    return u, v, x, y, valid_mask


def test_combine_dual_planar_pair_places_cameras_and_scales_velocity():
    # Two cameras, no overlap (cam1 to the right of cam0), identity raw-
    # to-canvas scale (region size == raw size) and a simple mm scale, so
    # the math is easy to check by hand: 1 raw px == 1 canvas px == 2 mm.
    dp = DualPlanarSettings(
        enabled=True,
        canvas_width=300, canvas_height=100,
        scale_x_mm_per_px=2.0, scale_x_offset_mm=0.0,
        scale_y_mm_per_px=-2.0, scale_y_offset_mm=0.0,
    )
    dp.cam0.region_x, dp.cam0.region_y = 100.0, 0.0
    dp.cam0.region_width, dp.cam0.region_height = 100.0, 80.0
    dp.cam0.raw_width, dp.cam0.raw_height = 100, 80
    dp.cam1.region_x, dp.cam1.region_y = 0.0, 0.0
    dp.cam1.region_width, dp.cam1.region_height = 100.0, 80.0
    dp.cam1.raw_width, dp.cam1.raw_height = 100, 80

    cam0 = _uniform_camera(u_value=1.0, v_value=0.0, nx=11, ny=9)   # raw x in [0,100]
    cam1 = _uniform_camera(u_value=-1.0, v_value=0.0, nx=11, ny=9)  # raw x in [0,100]

    X, Y, U, V, valid = combine_dual_planar_pair(cam0, cam1, dp, frame_dt_s=1.0)

    assert valid.any()
    # cam1 occupies canvas x in [0,100] -> mm [0,200]; cam0 occupies canvas
    # x in [100,200] -> mm [200,400]. No overlap, so each side should show
    # only that camera's own (scaled) velocity: u_mm/frame = u_raw_px *
    # scale_x(1.0 local) * 2.0mm/px, then /dt(1.0)/1000 -> m/s.
    expected_u0 = 1.0 * 1.0 * 2.0 / 1000.0
    expected_u1 = -1.0 * 1.0 * 2.0 / 1000.0
    cam0_side = valid & (X > 210)  # comfortably inside cam0's placed region, away from any edge NaNs
    cam1_side = valid & (X < 190)
    assert cam0_side.any() and cam1_side.any()
    assert np.allclose(U[cam0_side], expected_u0, atol=1e-9)
    assert np.allclose(U[cam1_side], expected_u1, atol=1e-9)


def test_combine_dual_planar_pair_averages_overlap():
    # cam1 covers raw x in [0,100] -> canvas/mm [0,100]; cam0 covers raw x
    # in [0,100] placed at region_x=80 -> canvas/mm [80,180]. Overlap is
    # mm [80,100] -- with cam0's U=+1 everywhere and cam1's U=+3
    # everywhere (in the same units), the overlap should average to +2,
    # not pick one camera arbitrarily.
    dp = DualPlanarSettings(
        enabled=True, canvas_width=200, canvas_height=100,
        scale_x_mm_per_px=1.0, scale_x_offset_mm=0.0,
        scale_y_mm_per_px=-1.0, scale_y_offset_mm=0.0,
    )
    dp.cam0.region_x, dp.cam0.region_y = 80.0, 0.0
    dp.cam0.region_width, dp.cam0.region_height = 100.0, 80.0
    dp.cam0.raw_width, dp.cam0.raw_height = 100, 80
    dp.cam1.region_x, dp.cam1.region_y = 0.0, 0.0
    dp.cam1.region_width, dp.cam1.region_height = 100.0, 80.0
    dp.cam1.raw_width, dp.cam1.raw_height = 100, 80

    cam0 = _uniform_camera(u_value=1.0, v_value=0.0, nx=21, ny=9)
    cam1 = _uniform_camera(u_value=3.0, v_value=0.0, nx=21, ny=9)

    X, Y, U, V, valid = combine_dual_planar_pair(cam0, cam1, dp, frame_dt_s=None)
    # frame_dt_s=None -> stays in mm/frame (see combine_dual_planar_pair's
    # docstring: position conversion is mandatory, only the /dt step is skipped)
    overlap = valid & (X > 85) & (X < 95)  # well inside [80,100] overlap, away from edges
    assert overlap.any()
    assert np.allclose(U[overlap], 2.0, atol=1e-6)  # mean of 1.0 and 3.0 mm/frame


# ---- real-data-gated: the actual project this feature was built against ----

@pytest.mark.skipif(not os.path.exists(REAL_SET), reason="real dual-camera planar project not available on this machine")
def test_detect_dual_planar_true_for_real_truck_set():
    assert detect_dual_planar_from_set(REAL_SET) is True


@pytest.mark.skipif(not os.path.exists(REAL_SET), reason="real dual-camera planar project not available on this machine")
def test_read_dual_planar_calibration_from_real_truck_set():
    dp = read_dual_planar_calibration_from_set(REAL_SET)
    assert dp.enabled is True
    assert (dp.canvas_width, dp.canvas_height) == (8009, 3046)
    assert dp.cam0.region_x == pytest.approx(3865.0)
    assert dp.cam0.region_y == pytest.approx(4.0)
    assert (dp.cam0.region_width, dp.cam0.region_height) == pytest.approx((4144.0, 3041.0))
    assert (dp.cam0.raw_width, dp.cam0.raw_height) == (4096, 3008)
    assert dp.cam1.region_x == pytest.approx(0.0)
    assert (dp.cam1.region_width, dp.cam1.region_height) == pytest.approx((4111.0, 3018.0))
    assert dp.scale_x_mm_per_px == pytest.approx(0.03927757527320482)
    assert dp.scale_y_mm_per_px == pytest.approx(-0.03927757527320482)
    # the confirmed real overlap: cam1 spans x in [0,4111], cam0 starts at
    # x=3865 -> a 246px-wide overlap strip, read directly off DaVis's own
    # RegionWithinCorrectedImage, not re-derived.
    overlap_px = (dp.cam1.region_x + dp.cam1.region_width) - dp.cam0.region_x
    assert overlap_px == pytest.approx(246.0)


@pytest.mark.skipif(not os.path.exists(REAL_SET_2), reason="real dual-camera planar project not available on this machine")
def test_read_dual_planar_calibration_shared_across_recordings():
    # Different recordings in the same DaVis project share one
    # Properties/Calibration/ -- confirmed by direct inspection of
    # D:\Truck_PIV_Round4\Loaded_CFD_Truck (multiple .set recordings, one
    # Properties folder).
    dp1 = read_dual_planar_calibration_from_set(REAL_SET)
    dp2 = read_dual_planar_calibration_from_set(REAL_SET_2)
    assert dp1.cam0.region_x == dp2.cam0.region_x
    assert dp1.scale_x_mm_per_px == dp2.scale_x_mm_per_px
    assert dp1.canvas_width == dp2.canvas_width


@pytest.mark.skipif(not os.path.exists(REAL_SET), reason="real dual-camera planar project not available on this machine")
def test_iter_dual_planar_from_set_yields_matching_raw_frame_sizes():
    # Confirmed via lvpyio inspection: this project's own StreamSet.xml
    # pins frames 0-1 to "Camera1" and 2-3 to "Camera2" (camera_major),
    # each a (3008, 4096) raw frame matching Calibration.xml's own
    # OriginalImageSize for both cameras.
    pair_id, fa0, fb0, fa1, fb1 = get_dual_planar_from_set(REAL_SET, 0)
    assert pair_id == "0000"
    assert fa0.shape == fb0.shape == (3008, 4096)
    assert fa1.shape == fb1.shape == (3008, 4096)

    it = iter_dual_planar_from_set(REAL_SET)
    first = next(it)
    assert first[0] == "0000"
    it.close()
