"""Tests for davis_set.py's stereo dewarp calibration auto-extraction.

SECOND ATTEMPT'S OUTCOME (this file's current design): the FIRST parse
attempt at DaVis's own Calibration.xml CoefficientsA/CoefficientsB
(confirmed, empirically, not to match CameraMapping's convention under
any forward/backward/factor-of-2 variant tried against an ARBITRARY
'world pixel = world_mm * px_per_mm' grid -- a flexible affine-transform
search DID get an excellent-looking fit, but failed a hold-out test
against the SAME camera's own second Z-plane, proving it was
overfitting) was superseded by fitting CameraMapping's OWN polynomial
directly from MarkPositionTable.xml's real raw-pixel<->world-mm ground
truth instead (_fit_camera_mapping_planes, still here, still tested
below -- kept as a standalone utility, no longer wired into
read_stereo_calibration_from_set's automatic path, see next paragraph).

A THIRD attempt (the one that actually cracked exact extraction) found
that CoefficientsA/CoefficientsB DO match CameraMapping's formula
EXACTLY, ZERO transformation needed -- the earlier attempt's grid was
simply the wrong one. DaVis's Origin/NormalisationFactor define
x0/x_span/y0/y_span in DaVis's OWN 'corrected image' pixel space (offset
from the naive mm*px_per_mm grid by a per-snapshot constant baked into
Scales/LinearScaleX/Y's OffsetMm), not the app's simple grid. Verified
against real MarkPositionTable.xml ground truth (feeding real marks'
WorldPos mm through the REAL LinearScaleX/Y inverse, and a real
CameraMapping.world_to_raw call with x0=s_o/x_span=nx/y0=t_o/y_span=ny/
dx_coefs=CoefficientsA/dy_coefs=CoefficientsB straight off Calibration.xml,
no fitting at all): 0.29-0.69px RMS across 2 real Polynomial3rdOrder
snapshots (both Z-planes each) -- see
_exact_camera_mapping_from_calibration_xml's own docstring for the full
derivation, including why a third real snapshot showed a bigger residual
explained by an unrelated data-consistency quirk, not a decode failure.
DaVis's OTHER internal calibration Type, 'PinholeOpenCV' (a standard
OpenCV pinhole+distortion model), was ALSO attempted and got close
(y-pixel already sub-pixel with a Euler-angle rotation) but not exact
(x-pixel off by a few px in a way that isn't a simple affine correction)
-- it has no exact path and read_stereo_calibration_from_set raises for
it rather than silently falling back to the (real, but non-exact)
mark-fit utility, per this project's explicit "exact values or manual
entry, never present a fit as the automatic answer" requirement.

A separate real-data finding, caught during manual GUI testing while the
mark-fit approach was still the automatic path (kept here since
_fit_camera_mapping_planes itself is still tested, and the underlying
real-data characteristic is still true): DaVis writes BYTE-IDENTICAL
MarkPositionTable.xml files into both camera1/ and camera2/ folders for
every calibration snapshot in the real project this was built against --
confirmed with a plain diff, and even the file's own internal
<Camera CameraNumber="1"> tag stays "1" inside camera2's copy. This is
now IRRELEVANT to the automatic extraction path (exact decode reads only
Calibration.xml, never MarkPositionTable.xml) but is exactly why every
real recording in that project -- which all resolve to a duplicate-marked
PinholeOpenCV snapshot, never a Polynomial3rdOrder one -- still ends up
needing manual entry: PinholeOpenCV has no exact decode, and the
mark-fit fallback that WOULD otherwise apply is unusable on this
project's real data regardless.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import COEF_KEYS, CameraMapping
from piv_suite.config.schema import CameraMappingSettings
from piv_suite.io.davis_set import (
    _derive_world_shape, _estimate_stereo_angles, _exact_camera_mapping_from_calibration_xml,
    _files_are_identical, _find_calibration_project_root, _fit_camera_mapping_planes,
    _POLY3_COEF_SUFFIXES, _read_pixel_per_mm, _read_set_time, _select_calibration_snapshot,
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


# ---- duplicate-marks detection ----

def test_files_are_identical_true_for_same_content(tmp_path):
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text("same content")
    b.write_text("same content")
    assert _files_are_identical(str(a), str(b))


def test_files_are_identical_false_for_different_content(tmp_path):
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text("camera one's marks")
    b.write_text("camera two's marks")
    assert not _files_are_identical(str(a), str(b))


def test_files_are_identical_false_when_one_missing(tmp_path):
    a, b = tmp_path / "a.xml", tmp_path / "missing.xml"
    a.write_text("content")
    assert not _files_are_identical(str(a), str(b))


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


# ---- exact decode: _exact_camera_mapping_from_calibration_xml (synthetic) ----

def _write_polynomial3rd_calibration_xml(path, planes_by_camera, ppm=18.0, corrected_wh=(2000.0, 1500.0),
                                          field_of_view="SideBySideStereoVolume", original_wh=(4096.0, 3008.0)):
    """planes_by_camera: {camera_id: [(z_position, s_o, t_o, nx, ny, a_coefs, b_coefs), ...]}
    a_coefs/b_coefs: dicts keyed like CameraMapping.COEF_KEYS. Builds a
    minimal but real-shaped Polynomial3rdOrder Calibration.xml -- the same
    element names/nesting confirmed against real DaVis projects (see
    _exact_camera_mapping_from_calibration_xml's docstring).

    original_wh: this camera's OriginalImageSize (Width, Height) --
    included by default (matching every real Calibration.xml, which
    always has it -- see J:\\Final_Stereo's own, Width=4096 Height=3008)
    so existing tests exercise the SAME real-shaped XML this decode
    actually sees; pass None to omit it, for a test that specifically
    checks the "not present" fallback."""
    mappers = []
    for cam_id, planes in planes_by_camera.items():
        plane_xml = []
        for z_pos, s_o, t_o, nx, ny, a, b in planes:
            a_attrs = " ".join(f'a_{suf}="{a[key]}"' for key, suf in zip(COEF_KEYS, _POLY3_COEF_SUFFIXES))
            b_attrs = " ".join(f'b_{suf}="{b[key]}"' for key, suf in zip(COEF_KEYS, _POLY3_COEF_SUFFIXES))
            plane_xml.append(f"""
     <PolynomialMapping>
      <ZPosition Value="{z_pos}" />
      <Origin s_o="{s_o}" t_o="{t_o}" />
      <NormalisationFactor nx="{nx}" ny="{ny}" />
      <Polynomial3rdOrder>
       <CoefficientsA {a_attrs} />
       <CoefficientsB {b_attrs} />
      </Polynomial3rdOrder>
     </PolynomialMapping>""")
        original_xml = (f'<OriginalImageSize Width="{original_wh[0]}" Height="{original_wh[1]}" />'
                         if original_wh is not None else "")
        mappers.append(f"""
   <CoordinateMapper CameraIdentifier="{cam_id}" Type="Polynomial3rdOrder" GroupId="1">
    <PolynomialParameters>
     <CommonParameters>
      <PixelPerMmFactor Value="{ppm}" />
      <CorrectedImageSize Width="{corrected_wh[0]}" Height="{corrected_wh[1]}" />
      {original_xml}
     </CommonParameters>{"".join(plane_xml)}
    </PolynomialParameters>
   </CoordinateMapper>""")
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0"?>
<Calibration Version="2">
 <CoordinateSystemsForEachView>
  <CoordinateSystem FieldOfView="{field_of_view}">{"".join(mappers)}
  </CoordinateSystem>
 </CoordinateSystemsForEachView>
</Calibration>
""")


def test_exact_camera_mapping_passes_coefficients_through_unchanged(tmp_path):
    """The core claim this decode rests on: CoefficientsA/CoefficientsB
    and Origin/NormalisationFactor map DIRECTLY onto
    CameraMapping's dx_coefs/dy_coefs and x0/x_span/y0/y_span -- no
    fitting, no transformation, exact pass-through (validated separately
    against real ground truth by
    test_exact_camera_mapping_matches_real_marks below)."""
    a = {k: float(i) for i, k in enumerate(COEF_KEYS)}
    b = {k: float(i) * 10 for i, k in enumerate(COEF_KEYS)}
    ppm = 17.920975188143096
    xml_path = tmp_path / "Calibration.xml"
    _write_polynomial3rd_calibration_xml(
        str(xml_path), {"1": [(ppm, 2807.0, 1387.0, 4096.0, 3008.0, a, b)]}, ppm=ppm)

    result = _exact_camera_mapping_from_calibration_xml(str(xml_path), "1")
    assert result is not None
    planes, decoded_ppm, corrected_wh = result
    assert len(planes) == 1
    plane = planes[0]
    assert decoded_ppm == pytest.approx(ppm)
    assert corrected_wh == (2000.0, 1500.0)
    assert plane.x0 == pytest.approx(2807.0)
    assert plane.x_span == pytest.approx(4096.0)
    assert plane.y0 == pytest.approx(1387.0)
    assert plane.y_span == pytest.approx(3008.0)
    assert plane.dx_coefs == pytest.approx(a)
    assert plane.dy_coefs == pytest.approx(b)
    assert plane.z_mm == pytest.approx(1.0)  # ZPosition == ppm -> z_mm == 1.0 exactly
    assert plane.raw_width == 4096
    assert plane.raw_height == 3008


def test_exact_camera_mapping_missing_original_image_size_defaults_to_zero(tmp_path):
    """OriginalImageSize is read on a best-effort basis (see
    CameraMappingSettings.raw_width/raw_height's own comment) -- its
    absence must NOT fail the exact decode (dewarp itself never needed
    it), just leave raw_width/raw_height at their "unknown, no masking
    possible" default of 0."""
    zero = {k: 0.0 for k in COEF_KEYS}
    xml_path = tmp_path / "Calibration.xml"
    _write_polynomial3rd_calibration_xml(
        str(xml_path), {"1": [(18.0, 0.0, 0.0, 100.0, 100.0, zero, zero)]}, original_wh=None)

    planes, _, _ = _exact_camera_mapping_from_calibration_xml(str(xml_path), "1")
    assert planes[0].raw_width == 0
    assert planes[0].raw_height == 0


def test_exact_camera_mapping_two_planes_z_mm_recovered(tmp_path):
    zero = {k: 0.0 for k in COEF_KEYS}
    ppm = 18.0
    xml_path = tmp_path / "Calibration.xml"
    _write_polynomial3rd_calibration_xml(str(xml_path), {"1": [
        (ppm * 1.0, 2807.0, 1387.0, 4096.0, 3008.0, zero, zero),
        (ppm * -2.0, 2807.0, 1387.0, 4120.0, 3025.0, zero, zero),
    ]}, ppm=ppm)

    planes, _, _ = _exact_camera_mapping_from_calibration_xml(str(xml_path), "1")
    assert len(planes) == 2
    assert planes[0].z_mm == pytest.approx(1.0)
    assert planes[1].z_mm == pytest.approx(-2.0)


def test_exact_camera_mapping_returns_none_for_pinhole_opencv_type(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    xml_path.write_text(_PINHOLE_CAL_XML)
    assert _exact_camera_mapping_from_calibration_xml(str(xml_path), "1") is None


def test_exact_camera_mapping_returns_none_when_no_planes(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    _write_polynomial3rd_calibration_xml(str(xml_path), {"1": []})
    assert _exact_camera_mapping_from_calibration_xml(str(xml_path), "1") is None


def test_exact_camera_mapping_returns_none_for_missing_camera(tmp_path):
    xml_path = tmp_path / "Calibration.xml"
    zero = {k: 0.0 for k in COEF_KEYS}
    _write_polynomial3rd_calibration_xml(str(xml_path), {"1": [(18.0, 0.0, 0.0, 100.0, 100.0, zero, zero)]})
    assert _exact_camera_mapping_from_calibration_xml(str(xml_path), "2") is None


# ---- _estimate_stereo_angles: synthetic, exact-angle recovery ----

def _planes_with_known_angle(angle_deg, dz_mm=1.0, px_per_mm=1.0):
    """Two CameraMappingSettings ("planes" of ONE camera) whose world_to_
    raw x-shift between them, at ANY sampled (X,Y) point (constant dx_coefs
    only -- no s/t dependence, so the estimate is exact, not merely
    averaged-to-approximately-correct), corresponds EXACTLY to angle_deg
    under _estimate_stereo_angles' own sign convention (dx = dX -
    dZ*tan(alpha) -- see that function's docstring): d_raw_mm = -tan
    (angle_deg)*dz_mm, achieved here via each plane's constant ("1") dx_
    coef term alone, everything else zero."""
    # world_to_raw_x = xp - dx_coefs["1"] (constant-only poly), so
    # rx1-rx2 = -dx_coefs["1"] for plane2's coefs=0 -- the coefficient
    # itself must be the NEGATION of the target (rx1-rx2)/px_per_mm value
    # (matches _estimate_stereo_angles' own -atan2(...) sign convention;
    # confirmed against the real Calibration.xml-driven end-to-end test
    # below, which does NOT pre-negate its own coefficient and recovers
    # its target angle correctly).
    zero = {k: 0.0 for k in COEF_KEYS}
    d_raw_mm = np.tan(np.radians(angle_deg)) * dz_mm
    c1 = {**zero, "1": d_raw_mm}  # plane1's world_to_raw_x = xp - d_raw_mm
    c2 = dict(zero)               # plane2's world_to_raw_x = xp
    plane1 = CameraMappingSettings(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                                    dx_coefs=c1, dy_coefs=zero, z_mm=dz_mm)
    plane2 = CameraMappingSettings(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                                    dx_coefs=c2, dy_coefs=zero, z_mm=0.0)
    return [plane1, plane2], px_per_mm


def test_estimate_stereo_angles_recovers_known_alpha_exactly():
    planes, px_per_mm = _planes_with_known_angle(30.0)
    alpha1, alpha2, beta1, beta2 = _estimate_stereo_angles(
        planes, planes, px_per_mm, world_shape=(100, 100), n_samples=5)
    assert alpha1 == pytest.approx(30.0, abs=1e-9)
    assert alpha2 == pytest.approx(30.0, abs=1e-9)
    assert beta1 == pytest.approx(0.0, abs=1e-9)
    assert beta2 == pytest.approx(0.0, abs=1e-9)


def test_estimate_stereo_angles_recovers_known_negative_alpha_exactly():
    planes, px_per_mm = _planes_with_known_angle(-35.78)
    alpha1, _alpha2, _beta1, _beta2 = _estimate_stereo_angles(
        planes, planes, px_per_mm, world_shape=(100, 100), n_samples=5)
    assert alpha1 == pytest.approx(-35.78, abs=1e-9)


def test_estimate_stereo_angles_returns_zeros_for_single_plane():
    zero = {k: 0.0 for k in COEF_KEYS}
    one_plane = [CameraMappingSettings(x0=0.0, x_span=100.0, y0=0.0, y_span=100.0,
                                        dx_coefs=zero, dy_coefs=zero, z_mm=1.0)]
    result = _estimate_stereo_angles(one_plane, one_plane, 1.0, world_shape=(100, 100))
    assert result == (0.0, 0.0, 0.0, 0.0)


def test_read_stereo_calibration_from_set_populates_angles_when_two_planes(tmp_path):
    """End-to-end: read_stereo_calibration_from_set's own returned
    StereoSettings carries non-default angles when the calibration has
    two Z-planes with a real angular difference between them -- not just
    _estimate_stereo_angles in isolation."""
    zero = {k: 0.0 for k in COEF_KEYS}
    ppm = 18.0
    # _estimate_stereo_angles divides the raw dx_coefs["1"] (RAW PIXEL
    # units, matching what a real a_o coefficient is in Calibration.xml)
    # by px_per_mm before comparing to dz (mm) -- so the coefficient
    # itself must be pre-scaled by ppm to land on a known target angle,
    # not just -tan(angle)*dz directly (that would be correct only if
    # px_per_mm==1, which the real pipeline never has).
    a_o = np.tan(np.radians(20.0)) * ppm  # plane1 z=1, plane2 z=0, dz=1
    a1 = {**zero, "1": a_o}
    root = tmp_path / "Project"
    cal_dir = root / "Properties" / "Calibration"
    cal_dir.mkdir(parents=True)
    _write_polynomial3rd_calibration_xml(str(cal_dir / "Calibration.xml"), {
        "1": [(ppm * 1.0, 0.0, 0.0, 100.0, 100.0, a1, zero), (ppm * 0.0, 0.0, 0.0, 100.0, 100.0, zero, zero)],
        "2": [(ppm * 1.0, 0.0, 0.0, 100.0, 100.0, a1, zero), (ppm * 0.0, 0.0, 0.0, 100.0, 100.0, zero, zero)],
    }, ppm=ppm)
    recording = root / "Recording" / "recording.set"
    recording.parent.mkdir(parents=True)
    recording.write_text("#GROUP Sets\n")

    result = read_stereo_calibration_from_set(str(recording))
    assert result.alpha1_deg == pytest.approx(20.0, abs=1e-6)
    assert result.alpha2_deg == pytest.approx(20.0, abs=1e-6)
    assert result.beta1_deg == pytest.approx(0.0, abs=1e-9)


# ---- full pipeline, synthetic exact-decode happy path (Calibration.xml
# alone is enough -- no camera1/camera2 MarkPositionTable.xml needed at
# all, unlike the old marks-fit path this replaced as the automatic
# route) ----

def _build_synthetic_exact_project(tmp_path, cam0_coefs, cam1_coefs, recording_time=None):
    """A minimal DaVis project with ONLY Properties/Calibration/
    Calibration.xml (Polynomial3rdOrder, both cameras) -- deliberately NO
    camera1/camera2 MarkPositionTable.xml folders, to prove the exact
    decode path never touches them."""
    ppm = 18.0
    root = tmp_path / "Project"
    cal_dir = root / "Properties" / "Calibration"
    cal_dir.mkdir(parents=True)
    a0, b0 = cam0_coefs
    a1, b1 = cam1_coefs
    _write_polynomial3rd_calibration_xml(str(cal_dir / "Calibration.xml"), {
        "1": [(ppm, 2807.0, 1387.0, 4096.0, 3008.0, a0, b0)],
        "2": [(ppm, 2807.0, 1387.0, 4096.0, 3008.0, a1, b1)],
    }, ppm=ppm)

    recording = root / "Swirl" / "recording.set"
    recording.parent.mkdir(parents=True)
    text = "#GROUP Sets\n"
    if recording_time:
        text += f'SetTime = "{recording_time}";\n'
    recording.write_text(text)
    return str(recording)


def test_read_stereo_calibration_from_set_synthetic_exact_decode_happy_path(tmp_path):
    zero = {k: 0.0 for k in COEF_KEYS}
    cam0_coefs = ({k: 1.0 for k in COEF_KEYS}, zero)
    cam1_coefs = ({k: 2.0 for k in COEF_KEYS}, zero)  # genuinely different from cam0
    recording = _build_synthetic_exact_project(tmp_path, cam0_coefs, cam1_coefs)

    result = read_stereo_calibration_from_set(recording)

    assert result.world_scale_px_per_mm == pytest.approx(18.0)
    assert result.cam0_mapping.dx_coefs == pytest.approx(cam0_coefs[0])
    assert result.cam1_mapping.dx_coefs == pytest.approx(cam1_coefs[0])
    assert result.cam0_mapping.dx_coefs != result.cam1_mapping.dx_coefs
    assert result.cam0_mapping_plane2 is None  # only one plane in this fixture
    assert result.world_shape == (1500, 2000)  # CorrectedImageSize (height, width)
    # OriginalImageSize (4096x3008, this file's default -- see
    # _write_polynomial3rd_calibration_xml) round-trips onto both
    # cameras' CameraMappingSettings, for CameraMapping.raw_domain_valid.
    assert result.cam0_mapping.raw_width == 4096
    assert result.cam0_mapping.raw_height == 3008
    assert result.cam1_mapping.raw_width == 4096
    assert result.cam1_mapping.raw_height == 3008


def test_read_stereo_calibration_from_set_rejects_side_by_side_2d(tmp_path):
    """A DaVis 'SideBySide2D' project (two coplanar cameras stitched into
    one wider field -- config.schema.DualPlanarSettings, NOT an angled
    stereo pair) uses the identical CoordinateMapper/CoefficientsA/B XML
    shape as real stereo -- confirmed by comparing a real SideBySide2D
    Calibration.xml against a real stereo one. Caught via real GUI
    testing: selecting a real SideBySide2D project's .set while
    Mode=Stereo silently extracted a 'Stereo calibration ... exact'
    status with no error, before this FieldOfView check was added. Must
    raise instead of geometrically-wrong-but-numerically-successful
    triangulation calibration."""
    zero = {k: 0.0 for k in COEF_KEYS}
    root = tmp_path / "Project"
    cal_dir = root / "Properties" / "Calibration"
    cal_dir.mkdir(parents=True)
    _write_polynomial3rd_calibration_xml(str(cal_dir / "Calibration.xml"), {
        "1": [(18.0, 2807.0, 1387.0, 4096.0, 3008.0, zero, zero)],
        "2": [(18.0, 2807.0, 1387.0, 4096.0, 3008.0, zero, zero)],
    }, field_of_view="SideBySide2D")
    recording = root / "Recording" / "recording.set"
    recording.parent.mkdir(parents=True)
    recording.write_text("#GROUP Sets\n")

    with pytest.raises(ValueError, match="SideBySide2D"):
        read_stereo_calibration_from_set(str(recording))


@pytest.mark.skipif(not os.path.exists(r"D:\Truck_PIV_Round4\Loaded_CFD_Truck"),
                     reason="real dual-camera planar project not available on this machine")
def test_read_stereo_calibration_from_real_side_by_side_2d_project_rejects():
    """The real project this repo's dual-camera planar feature was built
    against (D:\\Truck_PIV_Round4\\Loaded_CFD_Truck) must be rejected by
    the STEREO extractor, not silently accepted as if it were a valid
    stereo pair -- see test_read_stereo_calibration_from_set_rejects_side_by_side_2d
    for the synthetic version of this same regression."""
    with pytest.raises(ValueError, match="SideBySide2D"):
        read_stereo_calibration_from_set(
            r"D:\Truck_PIV_Round4\Loaded_CFD_Truck\X_150_mm_Y_0_mm.set")


def test_read_stereo_calibration_from_set_raises_when_type_not_polynomial3rd(tmp_path):
    """PinholeOpenCV (or anything else) has no exact decode -- must raise
    (never silently fall back to the non-exact marks fit) so the GUI's
    existing never-crash wiring surfaces this as 'enter manually'.
    _PINHOLE_CAL_XML only defines camera '1' -- camera '2' being entirely
    absent is itself a valid (if different) reason exact decode can't
    proceed, exercised the same way as a genuine Type mismatch would be."""
    root = tmp_path / "Project"
    cal_dir = root / "Properties" / "Calibration"
    cal_dir.mkdir(parents=True)
    (cal_dir / "Calibration.xml").write_text(_PINHOLE_CAL_XML)
    recording = root / "Swirl" / "recording.set"
    recording.parent.mkdir(parents=True)
    recording.write_text("#GROUP Sets\n")

    with pytest.raises(NotImplementedError, match="Polynomial3rdOrder"):
        read_stereo_calibration_from_set(str(recording))


# ---- real-data-gated: the actual project this feature was built against ----

@pytest.mark.skipif(not os.path.exists(REAL_SWIRL_SET), reason="real stereo project not available on this machine")
def test_read_stereo_calibration_from_real_swirl_set_raises_pinhole_not_exact():
    # The flagship real recording's applicable calibration snapshot
    # (Calibration_260715_171237, confirmed by direct inspection) is a
    # 'PinholeOpenCV' calibration Type -- confirmed, every real recording
    # in this project resolves to a PinholeOpenCV-typed snapshot, never a
    # Polynomial3rdOrder one, so exact extraction never applies here (it
    # would if a Polynomial3rdOrder snapshot were ever the applicable
    # one -- see test_exact_camera_mapping_matches_real_marks below,
    # which validates the decode itself against two real
    # Polynomial3rdOrder snapshots in this same project's History).
    with pytest.raises(NotImplementedError, match="Polynomial3rdOrder"):
        read_stereo_calibration_from_set(REAL_SWIRL_SET)


@pytest.mark.skipif(not os.path.exists(REAL_PROJECT_ROOT), reason="real stereo project not available on this machine")
@pytest.mark.parametrize("calibration_xml,marks_path", [
    (r"J:\Final_Stereo\Properties\Calibration\Calibration.xml",
     r"J:\Final_Stereo\Properties\Calibration\camera1\MarkPositionTable.xml"),
    (r"J:\Final_Stereo\Properties\Calibration History\Calibration_260713_181401\Calibration.xml",
     r"J:\Final_Stereo\Properties\Calibration History\Calibration_260713_181401\camera1\MarkPositionTable.xml"),
])
def test_exact_camera_mapping_matches_real_marks(calibration_xml, marks_path):
    """THE key real-ground-truth validation for the exact decode: camera1's
    own MarkPositionTable.xml (confirmed genuine, unlike camera2's
    byte-duplicated copy in this project -- see
    test_files_are_identical_matches_real_duplicated_mark_tables) gives
    real RawPos<->WorldPos correspondences that camera1's own
    Calibration.xml CoefficientsA/CoefficientsB, decoded with ZERO
    fitting, must reproduce. Verified: 0.29-0.69px RMS per plane across
    both of this project's non-corrupted Polynomial3rdOrder snapshots
    (the current calibration and Calibration_260713_181401 -- a third,
    Calibration_260713_191850, has an unrelated data-consistency issue
    between ITS marks and ITS own LinearScaleX/Y, documented in
    _exact_camera_mapping_from_calibration_xml's docstring, so isn't used
    as a regression gate here to avoid a flaky/misleading threshold)."""
    planes, _, _ = _exact_camera_mapping_from_calibration_xml(calibration_xml, "1")
    root = ET.parse(marks_path).getroot()
    view = root.find(".//View")
    by_z = {}
    for mark in view.findall("Mark"):
        z_index = mark.find("Index").attrib["z"]
        raw = mark.find("RawPos").attrib
        world = mark.find("WorldPos").attrib
        by_z.setdefault(z_index, []).append(
            (float(raw["x"]), float(raw["y"]), float(world["x"]), float(world["y"])))

    # DaVis's own LinearScaleX/Y (real-mm -> DaVis's 'corrected image'
    # pixel space that Origin/NormalisationFactor -- and so this decode --
    # are defined in) -- read straight from the same CoordinateMapper.
    cal_root = ET.parse(calibration_xml).getroot()
    scales = cal_root.find(".//CoordinateMapper[@CameraIdentifier='1']//Scales")
    fx, ox = (float(scales.find("LinearScaleX").attrib[k]) for k in ("FactorMmPerPixel", "OffsetMm"))
    fy, oy = (float(scales.find("LinearScaleY").attrib[k]) for k in ("FactorMmPerPixel", "OffsetMm"))

    assert len(planes) >= 1
    for i, plane in enumerate(planes):
        rows = by_z.get(str(i))
        if not rows:
            continue
        xr = np.array([r[0] for r in rows]); yr = np.array([r[1] for r in rows])
        xw_mm = np.array([r[2] for r in rows]); yw_mm = np.array([r[3] for r in rows])
        cmap = CameraMapping(plane.x0, plane.x_span, plane.y0, plane.y_span, plane.dx_coefs, plane.dy_coefs)
        cx = (xw_mm - ox) / fx
        cy = (yw_mm - oy) / fy
        px, py = cmap.world_to_raw(cx, cy)
        rms_x = float(np.sqrt(np.mean((px - xr) ** 2)))
        rms_y = float(np.sqrt(np.mean((py - yr) ** 2)))
        assert rms_x < 1.0, f"plane {i}: rms_x={rms_x:.4f}px"
        assert rms_y < 1.0, f"plane {i}: rms_y={rms_y:.4f}px"


@pytest.mark.skipif(not os.path.exists(REAL_SWIRL_SET), reason="real stereo project not available on this machine")
def test_files_are_identical_matches_real_duplicated_mark_tables():
    snapshot = r"J:\Final_Stereo\Properties\Calibration History\Calibration_260715_171237"
    assert _files_are_identical(
        snapshot + r"\camera1\MarkPositionTable.xml", snapshot + r"\camera2\MarkPositionTable.xml")


@pytest.mark.skipif(not os.path.exists(REAL_PROJECT_ROOT), reason="real stereo project not available on this machine")
def test_find_calibration_project_root_on_real_swirl_recording():
    found = _find_calibration_project_root(REAL_SWIRL_SET)
    assert found == REAL_PROJECT_ROOT
