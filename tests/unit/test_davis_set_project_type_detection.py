"""Tests for davis_set.py's detect_project_type_from_set -- the tri-state
(planar/stereo/dual_planar) acquisition-geometry auto-detector that lets
main_window._on_input_path_changed auto-select the correct Mode radio the
moment a .set is selected, instead of requiring the user to correctly
guess Planar vs Stereo BEFORE selecting a project for calibration
auto-extraction to look in the right place (see SESSION_HANDOFF.md, and
test_gui_smoke.py's real-GUI-driven auto-detect tests for the GUI-level
wiring on top of this pure function).

Synthetic fixtures mirror test_davis_set_dual_planar.py's own
_write_project pattern -- detect_project_type_from_set shares
_find_calibration_project_root/_select_calibration_snapshot/
_read_field_of_view with detect_dual_planar_from_set, so the same minimal
Calibration.xml shape is enough to exercise it."""

import os

import pytest

from piv_suite.io.davis_set import detect_project_type_from_set

REAL_DUAL_PLANAR_SET = r"D:\Truck_PIV_Round4\Loaded_CFD_Truck\X_150_mm_Y_0_mm.set"
REAL_STEREO_SET = (
    r"J:\Final_Stereo\Swirl\On Time=0.7_Burst On Time=0.0_Burst Off Time=0.0.set"
)

_SIDE_BY_SIDE_2D_XML = """<?xml version="1.0"?>
<Calibration Version="2" CalibrationIdentifier="test">
    <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
        <CoordinateSystem FieldOfView="SideBySide2D">
            <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1"/>
            <CoordinateMapper CameraIdentifier="2" Type="Polynomial3rdOrder" GroupId="1"/>
        </CoordinateSystem>
    </CoordinateSystemsForEachView>
</Calibration>
"""

_SIDE_BY_SIDE_STEREO_XML = """<?xml version="1.0"?>
<Calibration Version="2" CalibrationIdentifier="test">
    <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
        <CoordinateSystem FieldOfView="SideBySideStereoVolume">
            <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1"/>
            <CoordinateMapper CameraIdentifier="2" Type="Polynomial3rdOrder" GroupId="1"/>
        </CoordinateSystem>
    </CoordinateSystemsForEachView>
</Calibration>
"""

_PLAIN_PLANAR_XML = """<?xml version="1.0"?>
<Calibration Version="2" CalibrationIdentifier="test">
    <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
        <CoordinateSystem FieldOfView="Planar">
            <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder" GroupId="1"/>
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


def test_detects_dual_planar_for_side_by_side_2d(tmp_path):
    _root, recording = _write_project(tmp_path, _SIDE_BY_SIDE_2D_XML)
    assert detect_project_type_from_set(str(recording)) == "dual_planar"


def test_detects_stereo_for_side_by_side_stereo_volume(tmp_path):
    _root, recording = _write_project(tmp_path, _SIDE_BY_SIDE_STEREO_XML)
    assert detect_project_type_from_set(str(recording)) == "stereo"


def test_falls_back_to_planar_for_plain_planar_field_of_view(tmp_path):
    _root, recording = _write_project(tmp_path, _PLAIN_PLANAR_XML)
    assert detect_project_type_from_set(str(recording)) == "planar"


def test_falls_back_to_planar_when_no_calibration_at_all(tmp_path):
    recording = tmp_path / "loose_recording.set"
    recording.write_text("")
    assert detect_project_type_from_set(str(recording)) == "planar"


def test_falls_back_to_planar_for_unrecognized_field_of_view(tmp_path):
    xml = _PLAIN_PLANAR_XML.replace('FieldOfView="Planar"', 'FieldOfView="SomethingElseEntirely"')
    _root, recording = _write_project(tmp_path, xml)
    assert detect_project_type_from_set(str(recording)) == "planar"


def test_never_raises_on_unparseable_calibration_xml(tmp_path):
    root = tmp_path / "Project"
    (root / "Properties" / "Calibration").mkdir(parents=True)
    (root / "Properties" / "Calibration" / "Calibration.xml").write_text("not valid xml <<<")
    recording = root / "Recording.set"
    recording.write_text("")
    assert detect_project_type_from_set(str(recording)) == "planar"


# ---- real-data-gated (see SESSION_HANDOFF.md -- read-only reference data,
# never modified by these tests) ----

@pytest.mark.skipif(not os.path.exists(REAL_DUAL_PLANAR_SET),
                     reason="real dual-camera planar project not available on this machine")
def test_detects_dual_planar_for_real_truck_set():
    assert detect_project_type_from_set(REAL_DUAL_PLANAR_SET) == "dual_planar"


@pytest.mark.skipif(not os.path.exists(REAL_STEREO_SET),
                     reason="real stereo project not available on this machine")
def test_detects_stereo_for_real_swirl_set():
    assert detect_project_type_from_set(REAL_STEREO_SET) == "stereo"
