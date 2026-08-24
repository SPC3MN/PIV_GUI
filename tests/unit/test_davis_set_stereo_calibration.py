"""Tests for davis_set.py's stereo dewarp calibration auto-extraction.

The core design choice this file locks in: rather than parsing DaVis's own
Calibration.xml CoefficientsA/CoefficientsB (confirmed, empirically, not to
match CameraMapping's convention under any forward/backward/factor-of-2
variant tried), extraction fits CameraMapping's OWN, UNCHANGED polynomial
directly from MarkPositionTable.xml's real raw-pixel<->world-mm ground
truth. The synthetic tests below construct marks the same way -- by
generating raw positions FROM known CameraMapping parameters -- so the fit
can be checked against exact, known-correct coefficients without needing
real DaVis data.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import COEF_KEYS, CameraMapping
from piv_suite.io.davis_set import (
    _derive_world_shape, _find_calibration_project_root, _fit_camera_mapping_planes,
    _read_pixel_per_mm, _read_set_time, _select_calibration_snapshot,
    read_stereo_calibration_from_set,
)

REAL_PROJECT_ROOT = r"J:\Final_Stereo"
REAL_SWIRL_SET = (
    r"J:\Final_Stereo\Swirl\On Time=0.7_Burst On Time=0.0_Burst Off Time=0.0.set"
)


# ---- project-root discovery / snapshot selection (pure, tmp_path-based) ----

def test_find_calibration_project_root_walks_up_from_recording(tmp_path):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    (root / "Properties" / "Calibration" / "Calibration.xml").write_text("<Calibration/>")
    recording = root / "Swirl" / "On Time=0.7.set"
    recording.parent.mkdir(parents=True)
    recording.write_text("")

    found = _find_calibration_project_root(str(recording))
    assert found == str(root)


def test_find_calibration_project_root_returns_none_beyond_max_levels(tmp_path):
    recording = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "recording.set"
    recording.parent.mkdir(parents=True)
    recording.write_text("")
    assert _find_calibration_project_root(str(recording), max_levels=2) is None


def test_read_set_time_parses_real_format(tmp_path):
    set_path = tmp_path / "recording.set"
    set_path.write_text(
        '// set file <...> created by DaVis\n'
        '#GROUP Sets\nSetType = 4352;\n'
        'SetTime = "2026-07-21T22:09:52-04:00";\nSetInc = 1;\n'
    )
    dt = _read_set_time(str(set_path))
    assert dt == datetime.fromisoformat("2026-07-21T22:09:52-04:00")


def test_read_set_time_missing_field_returns_none(tmp_path):
    set_path = tmp_path / "recording.set"
    set_path.write_text("#GROUP Sets\nSetType = 4352;\n")
    assert _read_set_time(str(set_path)) is None


def _write_history_snapshot(history_dir, name):
    snap_dir = history_dir / name
    snap_dir.mkdir(parents=True)
    (snap_dir / "Calibration.xml").write_text("<Calibration/>")
    return snap_dir


def test_select_calibration_snapshot_picks_nearest_preceding_history_entry(tmp_path):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    history = root / "Properties" / "Calibration History"
    _write_history_snapshot(history, "Calibration_260713_181401")
    expected = _write_history_snapshot(history, "Calibration_260715_171237")
    _write_history_snapshot(history, "Calibration_260722_165055")  # after the recording

    recording_dt = datetime(2026, 7, 21, 22, 9, 52)
    snap_dir, label = _select_calibration_snapshot(str(root), recording_dt)
    assert snap_dir == str(expected)
    assert label == "Calibration_260715_171237"


def test_select_calibration_snapshot_falls_back_to_current_when_none_precede(tmp_path):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    history = root / "Properties" / "Calibration History"
    _write_history_snapshot(history, "Calibration_260722_165055")

    recording_dt = datetime(2026, 7, 1, 0, 0, 0)  # before every history entry
    snap_dir, label = _select_calibration_snapshot(str(root), recording_dt)
    assert snap_dir == str(root / "Properties" / "Calibration")
    assert "current" in label


def test_select_calibration_snapshot_falls_back_to_current_when_recording_dt_unknown(tmp_path):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    snap_dir, label = _select_calibration_snapshot(str(root), None)
    assert snap_dir == str(root / "Properties" / "Calibration")
    assert "current" in label


# ---- PixelPerMmFactor, regardless of calibration Type ----

_POLY_CAL_XML = """<?xml version="1.0"?>
<Calibration Version="2">
 <CoordinateSystemsForEachView>
  <CoordinateSystem FieldOfView="SideBySideStereoVolume">
   <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1">
    <PolynomialParameters>
     <CommonParameters><PixelPerMmFactor Value="17.920975188143096" /></CommonParameters>
    </PolynomialParameters>
   </CoordinateMapper>
  </CoordinateSystem>
 </CoordinateSystemsForEachView>
</Calibration>
"""

_PINHOLE_CAL_XML = """<?xml version="1.0"?>
<Calibration Version="2">
 <CoordinateSystemsForEachView>
  <CoordinateSystem FieldOfView="SideBySideStereoVolume">
   <CoordinateMapper CameraIdentifier="1" Type="PinholeOpenCV" GroupId="1">
    <PinholeParameters Bundled="false">
     <CommonParameters><PixelPerMmFactor Value="17.920975188143096" /></CommonParameters>
    </PinholeParameters>
   </CoordinateMapper>
  </CoordinateSystem>
 </CoordinateSystemsForEachView>
</Calibration>
"""


@pytest.mark.parametrize("xml_text", [_POLY_CAL_XML, _PINHOLE_CAL_XML])
def test_read_pixel_per_mm_finds_value_regardless_of_calibration_type(tmp_path, xml_text):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(xml_text)
    assert _read_pixel_per_mm(str(xml_path), "1") == pytest.approx(17.920975188143096)


def test_read_pixel_per_mm_missing_camera_raises(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_POLY_CAL_XML)
    with pytest.raises(ValueError):
        _read_pixel_per_mm(str(xml_path), "2")


# ---- mark-fitting: synthetic ground truth (portable, no real data needed) ----

def _write_mark_table(path, marks_by_plane):
    """marks_by_plane: {z_index_str: (z_mm, [(x_raw,y_raw,x_world_mm,y_world_mm), ...])}."""
    root = ET.Element("MarkTable")
    cam = ET.SubElement(root, "Camera", CameraNumber="1")
    view = ET.SubElement(cam, "View")
    for z_index, (z_mm, rows) in marks_by_plane.items():
        for x_raw, y_raw, x_w, y_w in rows:
            mark = ET.SubElement(view, "Mark")
            ET.SubElement(mark, "Index", z=z_index, y="0", x="0")
            ET.SubElement(mark, "RawPos", x=repr(x_raw), y=repr(y_raw))
            ET.SubElement(mark, "WorldPos", z=repr(z_mm), x=repr(x_w), y=repr(y_w))
    ET.ElementTree(root).write(path)


def _synthetic_plane_marks(x0, x_span, y0, y_span, dx_coefs, dy_coefs, px_per_mm, n=8):
    """Generate marks consistent with CameraMapping's OWN formula (raw =
    world - poly(s_world,t_world)) for a grid of world-pixel positions,
    then convert world back to mm (what MarkPositionTable.xml stores)."""
    cm = CameraMapping(x0, x_span, y0, y_span, dx_coefs, dy_coefs)
    # span the FULL nominal extent -- _fit_camera_mapping_planes derives
    # x0/x_span/y0/y_span from the marks' own actual min/max, so the
    # fitted span only matches x_span/y_span exactly if the marks do too.
    xw = np.linspace(x0 - x_span / 2, x0 + x_span / 2, n)
    yw = np.linspace(y0 - y_span / 2, y0 + y_span / 2, n)
    xg, yg = np.meshgrid(xw, yw)
    xr, yr = cm.world_to_raw(xg.ravel(), yg.ravel())
    xw_mm, yw_mm = xg.ravel() / px_per_mm, yg.ravel() / px_per_mm
    return list(zip(xr.tolist(), yr.tolist(), xw_mm.tolist(), yw_mm.tolist()))


def test_fit_camera_mapping_planes_recovers_synthetic_ground_truth(tmp_path):
    px_per_mm = 18.0
    x0, x_span, y0, y_span = 100.0, 200.0, 50.0, 150.0
    dx_coefs = {k: v for k, v in zip(COEF_KEYS, [5.0, 2.0, -0.5, 0.1, 0.3, -0.2, 0.05, 0.4, -0.1, 0.2])}
    dy_coefs = {k: v for k, v in zip(COEF_KEYS, [3.0, -1.0, 0.4, -0.05, 1.5, 0.1, -0.02, -0.3, 0.05, -0.15])}
    rows = _synthetic_plane_marks(x0, x_span, y0, y_span, dx_coefs, dy_coefs, px_per_mm)
    mark_path = tmp_path / "MarkPositionTable.xml"
    _write_mark_table(str(mark_path), {"0": (1.0, rows)})

    planes = _fit_camera_mapping_planes(str(mark_path), px_per_mm)
    assert len(planes) == 1
    fitted = planes[0]
    assert fitted.z_mm == pytest.approx(1.0)
    assert fitted.x0 == pytest.approx(x0, abs=1e-6)
    assert fitted.x_span == pytest.approx(x_span, abs=1e-6)
    assert fitted.y0 == pytest.approx(y0, abs=1e-6)
    assert fitted.y_span == pytest.approx(y_span, abs=1e-6)
    for k in COEF_KEYS:
        assert fitted.dx_coefs[k] == pytest.approx(dx_coefs[k], abs=1e-4)
        assert fitted.dy_coefs[k] == pytest.approx(dy_coefs[k], abs=1e-4)


def test_fit_camera_mapping_planes_two_planes_detected(tmp_path):
    px_per_mm = 18.0
    zero = {k: 0.0 for k in COEF_KEYS}
    rows0 = _synthetic_plane_marks(100.0, 200.0, 50.0, 150.0, zero, zero, px_per_mm)
    rows1 = _synthetic_plane_marks(90.0, 190.0, 45.0, 140.0, zero, zero, px_per_mm)
    mark_path = tmp_path / "MarkPositionTable.xml"
    _write_mark_table(str(mark_path), {"0": (1.0, rows0), "1": (-2.0, rows1)})

    planes = _fit_camera_mapping_planes(str(mark_path), px_per_mm)
    assert len(planes) == 2
    assert planes[0].z_mm == pytest.approx(1.0)
    assert planes[1].z_mm == pytest.approx(-2.0)


def test_fit_camera_mapping_planes_skips_plane_with_too_few_marks(tmp_path):
    px_per_mm = 18.0
    zero = {k: 0.0 for k in COEF_KEYS}
    rows_ok = _synthetic_plane_marks(100.0, 200.0, 50.0, 150.0, zero, zero, px_per_mm, n=8)
    rows_sparse = rows_ok[:5]  # fewer than the 20-mark minimum
    mark_path = tmp_path / "MarkPositionTable.xml"
    _write_mark_table(str(mark_path), {"0": (1.0, rows_ok), "1": (-2.0, rows_sparse)})

    planes = _fit_camera_mapping_planes(str(mark_path), px_per_mm)
    assert len(planes) == 1
    assert planes[0].z_mm == pytest.approx(1.0)


def test_fit_camera_mapping_planes_no_usable_plane_raises(tmp_path):
    mark_path = tmp_path / "MarkPositionTable.xml"
    _write_mark_table(str(mark_path), {})
    with pytest.raises(ValueError):
        _fit_camera_mapping_planes(str(mark_path), 18.0)


def test_derive_world_shape_uses_largest_extent():
    from piv_suite.config.schema import CameraMappingSettings

    def plane(x_span, y_span):
        return CameraMappingSettings(x_span=x_span, y_span=y_span)

    shape = _derive_world_shape([plane(100.0, 50.0), plane(80.0, 60.0)], [plane(90.0, 40.0)])
    assert shape == (60, 100)


# ---- real-data-gated: the actual project this feature was built against ----

@pytest.mark.skipif(not os.path.exists(REAL_SWIRL_SET), reason="real stereo project not available on this machine")
def test_read_stereo_calibration_from_real_swirl_set():
    # The flagship real recording -- its applicable calibration snapshot
    # (Calibration_260715_171237, confirmed by direct inspection) is a
    # PinholeOpenCV-type calibration, but extraction succeeds anyway since
    # it fits from MarkPositionTable.xml ground truth, not DaVis's own
    # Type-specific coefficient representation.
    result = read_stereo_calibration_from_set(REAL_SWIRL_SET)

    assert result.world_scale_px_per_mm == pytest.approx(17.920975188143096)
    assert "Calibration_260715_171237" in result.cam0_mapping.name
    assert result.cam0_mapping.z_mm == pytest.approx(1.0, abs=1e-6)
    assert result.cam0_mapping_plane2 is not None
    assert result.cam0_mapping_plane2.z_mm == pytest.approx(-2.0, abs=1e-6)
    assert result.cam1_mapping.z_mm == pytest.approx(1.0, abs=1e-6)
    assert result.cam1_mapping_plane2 is not None
    assert result.world_shape[0] > 0 and result.world_shape[1] > 0
    assert result.sheet_z_mm is None


@pytest.mark.skipif(not os.path.exists(REAL_SWIRL_SET), reason="real stereo project not available on this machine")
def test_real_stereo_calibration_dewarp_round_trips_and_interpolates():
    from piv_suite.calibration.camera_mapping import build_camera_mapping

    result = read_stereo_calibration_from_set(REAL_SWIRL_SET)

    # boundary sheet_z_mm must reduce exactly to that plane's own mapping
    cam0_at_plane1 = build_camera_mapping(result.cam0_mapping, result.cam0_mapping_plane2, sheet_z_mm=1.0)
    assert cam0_at_plane1.x0 == result.cam0_mapping.x0
    assert cam0_at_plane1.dx_coefs["s"] == result.cam0_mapping.dx_coefs["s"]

    # interpolated dewarp of a real-shaped raw frame produces a sane,
    # non-degenerate (partially nonzero, not all-zero/all-NaN) result
    cam0_mid = build_camera_mapping(result.cam0_mapping, result.cam0_mapping_plane2, sheet_z_mm=-0.5)
    raw = np.random.rand(3008, 4096).astype(np.float32)
    dewarped = cam0_mid.dewarp_image(raw, result.world_shape, order=1)
    assert dewarped.shape == result.world_shape
    assert not np.isnan(dewarped).any()
    assert 0.05 < np.mean(dewarped > 0) < 0.95


@pytest.mark.skipif(not os.path.exists(REAL_PROJECT_ROOT), reason="real stereo project not available on this machine")
def test_find_calibration_project_root_on_real_swirl_recording():
    found = _find_calibration_project_root(REAL_SWIRL_SET)
    assert found == REAL_PROJECT_ROOT
