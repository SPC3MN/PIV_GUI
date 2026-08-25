"""DaVis `.set` project ingestion (planar and stereo), plus the
single-set-vs-folder-of-sets resolution shared by both.

Migrated from piv_common.resolve_set_paths/set_label/iter_pairs_from_set
and stereo_common.iter_stereo_from_set (byte-identical across all four
source repos). This already implements ".set DaVis project file" ingestion
via lvpyio -- both a single `.set` path and a folder containing multiple
`.set` entries (batch mode) were supported from the start.
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from piv_suite.calibration.camera_mapping import COEF_KEYS
from piv_suite.config.schema import (
    CalibrationSettings, CameraMappingSettings, DualPlanarCameraSettings,
    DualPlanarSettings, StereoSettings,
)

from .buffers import frames_from_buffer, frames_from_stereo_buffer


def resolve_set_paths(input_path):
    """Decide whether input_path is ONE DaVis set to process directly, or a
    folder holding MULTIPLE sets to batch through one after another.

    - a path ending in '.set' is always treated as a single set.
    - otherwise, if it's a directory containing nested '*.set' entries,
      those are treated as the sets to batch over (folder-of-sets mode).
    - otherwise input_path itself is treated as the (single) set -- e.g. a
      raw DaVis project folder not named with a '.set' suffix.

    Returns (set_paths, is_batch)."""
    if input_path.lower().endswith(".set"):
        return [input_path], False
    if os.path.isdir(input_path):
        nested = sorted(glob.glob(os.path.join(input_path, "*.set")))
        if nested:
            return nested, True
    return [input_path], False


def set_label(set_path):
    """Short name for a set path, used for per-set output subfolders and
    logging -- strips a trailing '.set' if present."""
    base = os.path.basename(os.path.normpath(set_path))
    if base.lower().endswith(".set"):
        base = base[: -len(".set")]
    return base


def _open_dataset(set_path, multiset_index, quiet=False):
    """Shared multi-set-vs-single-set resolution for all four
    iter/list/get functions below. Returns (dataset, owns_dataset) --
    owns_dataset tells the caller whether it's responsible for closing
    it (a multi-set's sub-dataset is a view into the parent `sets`
    object, which itself doesn't need (and can't sensibly) be closed
    per sub-dataset)."""
    import lvpyio as lv
    if lv.is_multiset(set_path):
        if not quiet:
            print(f"[info] '{set_path}' is a multi-set (e.g. multiple cameras) "
                  f"-- using sub-set index {multiset_index}")
        sets = lv.read_set(set_path)
        return sets[multiset_index], False
    return lv.read_set(set_path), True


def iter_pairs_from_set(set_path, multiset_index=0):
    """Yield (pair_id, frame_a, frame_b) from a DaVis image set (planar)."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index)
    try:
        n = len(dataset)
        for i in range(n):
            pair_id = f"{i:04d}"
            buf = dataset[i]
            frame_a, frame_b = frames_from_buffer(buf)
            yield pair_id, frame_a, frame_b
    finally:
        if owns_dataset:
            dataset.close()


def iter_stereo_from_set(set_path, multiset_index=0, stereo_frame_order="camera_major"):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a DaVis stereo image set."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index)
    try:
        n = len(dataset)
        for i in range(n):
            pair_id = f"{i:04d}"
            buf = dataset[i]
            fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, stereo_frame_order)
            yield pair_id, fa0, fb0, fa1, fb1
    finally:
        if owns_dataset:
            dataset.close()


def iter_dual_planar_from_set(set_path, multiset_index=0):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a DaVis "SideBySide2D"
    dual-camera planar .set -- i.e. two COPLANAR cameras each imaging a
    different (overlapping) region of one larger flat plane, stitched
    into a wider planar field (see config.schema.DualPlanarSettings),
    NOT stereo (no triangulation).

    This is a thin wrapper around iter_stereo_from_set, not a
    reimplementation: confirmed directly against a real SideBySide2D
    project (D:\\Truck_PIV_Round4\\Loaded_CFD_Truck) that lvpyio reports
    `is_multiset() == False` for it (unlike the stereo case) and each
    Buffer already holds the same 4-frame layout frames_from_stereo_buffer
    expects (2 cameras x 2 exposures) -- verified via that project's own
    StreamSet.xml, whose FrameReader ContentPurpose entries pin frames
    0-1 to FilePrefix="Camera1" and frames 2-3 to FilePrefix="Camera2",
    i.e. exactly stereo_frame_order="camera_major"'s
    [cam0_A, cam0_B, cam1_A, cam1_B] layout -- not a free per-project
    choice the way stereo's own (ambiguous, user-facing) frame_order
    setting is, so it's hard-coded here rather than exposed as a GUI
    option."""
    return iter_stereo_from_set(set_path, multiset_index, stereo_frame_order="camera_major")


def get_dual_planar_from_set(set_path, index, multiset_index=0):
    """Load a single (pair_id, fa0, fb0, fa1, fb1) at `index` directly --
    see iter_dual_planar_from_set's docstring for why this is a thin
    camera_major-pinned wrapper around get_stereo_from_set rather than a
    separate implementation."""
    return get_stereo_from_set(set_path, index, multiset_index, stereo_frame_order="camera_major")


def list_pair_ids_from_set(set_path, multiset_index=0):
    """Cheap: just the pair ids (e.g. for a GUI pair-picker), without
    loading any frame image data."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index, quiet=True)
    try:
        return [f"{i:04d}" for i in range(len(dataset))]
    finally:
        if owns_dataset:
            dataset.close()


def get_pair_from_set(set_path, index, multiset_index=0):
    """Load a single (pair_id, frame_a, frame_b) at `index` directly,
    without iterating/loading every pair before it."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index, quiet=True)
    try:
        buf = dataset[index]
        frame_a, frame_b = frames_from_buffer(buf)
        return f"{index:04d}", frame_a, frame_b
    finally:
        if owns_dataset:
            dataset.close()


def get_stereo_from_set(set_path, index, multiset_index=0, stereo_frame_order="camera_major"):
    """Load a single (pair_id, fa0, fb0, fa1, fb1) at `index` directly,
    without iterating/loading every pair before it."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index, quiet=True)
    try:
        buf = dataset[index]
        fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, stereo_frame_order)
        return f"{index:04d}", fa0, fb0, fa1, fb1
    finally:
        if owns_dataset:
            dataset.close()


def read_calibration_from_set(set_path, multiset_index=0):
    """Best-effort planar calibration extraction (pixel pitch + frame Δt)
    from a DaVis `.set` project, for auto-filling the GUI's Calibration
    fields instead of requiring manual entry.

    Each field is independently best-effort: a field that couldn't be
    determined comes back None (never raised) -- matching
    CalibrationSettings' own "None = keep px/frame units" convention, so
    the GUI can leave that one field unfilled. Only a genuinely unusable
    set_path (doesn't exist / isn't a DaVis set at all) raises, propagated
    from lvpyio -- same contract as every other function in this module.

    pixel_pitch_mm comes from frame.scales.x.slope (already mm-per-raw-
    pixel; y is intentionally never used -- its slope is sign-flipped by a
    DaVis convention unrelated to pixel size). Read directly off
    dataset[0].frames[0], NOT via frames_from_buffer(), because a
    calibration-only read must also work on a single-frame buffer (e.g. a
    DaVis *processing job* .set, which frames_from_buffer would reject).

    frame_dt_s is NOT available through lvpyio's buffer/scales API -- it
    lives in a sibling Settings_Acquisition_Timing_{GUID}.xml file (GUID
    varies per recording, hence a glob) next to the set's own data
    folder. See _frame_dt_from_timing_xml's docstring for the exact
    on-disk layout and a critical gotcha in that XML's structure."""
    dataset, owns_dataset = _open_dataset(set_path, multiset_index, quiet=True)
    try:
        pixel_pitch_mm = _read_pixel_pitch_mm(dataset)
        frame_dt_s = _read_frame_dt_s(set_path, dataset, is_multiset_sub=not owns_dataset)
    finally:
        if owns_dataset:
            dataset.close()
    return CalibrationSettings(pixel_pitch_mm=pixel_pitch_mm, frame_dt_s=frame_dt_s)


def _read_pixel_pitch_mm(dataset):
    try:
        return float(dataset[0].frames[0].scales.x.slope)
    except Exception:
        # best-effort: an empty set, an unusual buffer shape, or any
        # lvpyio-internal surprise all just mean "couldn't determine this
        # field" here -- not a reason to fail the whole extraction.
        return None


def _read_frame_dt_s(set_path, dataset, is_multiset_sub):
    xml_path = _find_timing_xml(set_path, dataset, is_multiset_sub)
    return None if xml_path is None else _frame_dt_from_timing_xml(xml_path)


def _find_timing_xml(set_path, dataset, is_multiset_sub):
    """Locate the sibling Settings_Acquisition_Timing_{GUID}.xml for
    set_path. Verified against a real DaVis dataset on disk:

    - a single (non-multiset) .set's data lives in a same-named sibling
      folder: dirname(set_path)/<set_label(set_path)>/  -- and that
      folder's own dataset.title equals set_label(set_path).
    - a MULTISET .set's sub-entry data is nested one level deeper:
      dirname(set_path)/<set_label(set_path)>/<dataset.title>/  -- the
      top-level .set's own label names an intermediate folder, and each
      sub-entry's OWN title (not the top-level label) names its data
      folder within that. Using set_label(set_path) alone for the
      multiset case finds the wrong (parent) directory.

    dataset.title is used in both branches (available on whatever
    _open_dataset() returned) rather than re-deriving the folder name
    from set_path by string manipulation, since it's the one thing
    lvpyio itself agrees on. Falls back to dirname(set_path) itself as a
    last resort, in case some other set layout doesn't match either
    verified case.

    Returns the first Settings_Acquisition_Timing_*.xml match found (a
    given set should only produce one), or None."""
    top_dir = os.path.dirname(set_path)
    title = getattr(dataset, "title", None)
    candidates = []
    if title:
        if is_multiset_sub:
            candidates.append(os.path.join(top_dir, set_label(set_path), title))
        else:
            candidates.append(os.path.join(top_dir, title))
    candidates.append(top_dir)
    for d in candidates:
        matches = sorted(glob.glob(os.path.join(d, "Settings_Acquisition_Timing_*.xml")))
        if matches:
            return matches[0]
    return None


def _frame_dt_from_timing_xml(xml_path):
    """Parse a Settings_Acquisition_Timing_*.xml's real frame Δt.

    THE FILE HAS MULTIPLE <timespan> ELEMENTS -- confirmed against a real
    file: two DeviceOffsetList entries (both value="0", irrelevant here)
    plus the one that actually matters, nested under
    aligner[@class='Acq::DoubleFrameAligner']/dt/timespan. A naive "first
    <timespan> anywhere" query silently returns 0 instead of the real
    value (700000000 on the reference dataset) -- since frame_dt_s is
    later used as a DIVISOR (processing.postprocess.apply_calibration),
    that would corrupt every velocity in the field rather than raise, so
    the specific XPath below is required, not a shortcut.

    value is in picoseconds; returns seconds, or None if the expected
    element isn't present or the file can't be parsed."""
    try:
        root = ET.parse(xml_path).getroot()
        el = root.find(".//aligner[@class='Acq::DoubleFrameAligner']/dt/timespan")
        if el is None:
            return None
        return int(el.attrib["value"]) / 1e12
    except (ET.ParseError, KeyError, ValueError, OSError):
        return None


def _find_calibration_project_root(set_path, max_levels=6):
    """Walk up from a recording's .set path looking for the DaVis project
    root -- identified by a 'Properties/Calibration/Calibration.xml'
    sibling. The number of levels between a recording and its project
    root varies (single-set vs multiset layout), hence the bounded walk
    rather than a fixed number of parents. Returns None if not found."""
    d = os.path.dirname(os.path.abspath(set_path))
    for _ in range(max_levels + 1):
        if os.path.isfile(os.path.join(d, "Properties", "Calibration", "Calibration.xml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


_SET_TIME_RE = re.compile(r'SetTime\s*=\s*"([^"]+)"')


def _read_set_time(set_path):
    """A DaVis .set file is plain text (a '#GROUP Sets' block), not XML --
    confirmed against real files. Returns the recording's own SetTime as
    a datetime, or None if unreadable/unparseable."""
    try:
        with open(set_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    m = _SET_TIME_RE.search(text)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1))
    except ValueError:
        return None


_HISTORY_RE = re.compile(r"Calibration_(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def _select_calibration_snapshot(project_root, recording_dt):
    """Pick whichever calibration snapshot was actually in effect for a
    recording taken at recording_dt: the 'Properties/Calibration History/
    Calibration_YYMMDD_HHMMSS/' entry with the latest timestamp that's
    still <= the recording's own, falling back to 'Properties/
    Calibration/' (current) only if none precede it or the recording's
    own timestamp is unknown. A project's *current* calibration can
    postdate an older recording -- confirmed on a real project, where
    blindly using "current" would silently pick up a later recalibration
    rather than the one actually valid at acquisition time.

    Returns (snapshot_dir, label) -- snapshot_dir contains Calibration.xml
    plus camera1/, camera2/ mark-data subfolders, either way."""
    current_dir = os.path.join(project_root, "Properties", "Calibration")
    history_dir = os.path.join(project_root, "Properties", "Calibration History")
    candidates = []
    if os.path.isdir(history_dir):
        for name in os.listdir(history_dir):
            m = _HISTORY_RE.fullmatch(name)
            snap_dir = os.path.join(history_dir, name)
            if m and os.path.isfile(os.path.join(snap_dir, "Calibration.xml")):
                yy, mm, dd, hh, mi, ss = (int(g) for g in m.groups())
                candidates.append((datetime(2000 + yy, mm, dd, hh, mi, ss), snap_dir, name))

    if recording_dt is None:
        return current_dir, "current (recording timestamp unknown)"
    recording_naive = recording_dt.replace(tzinfo=None)
    preceding = [c for c in candidates if c[0] <= recording_naive]
    if not preceding:
        return current_dir, "current (no History snapshot precedes the recording)"
    preceding.sort(key=lambda c: c[0])
    _, snap_dir, name = preceding[-1]
    return snap_dir, name


def _read_pixel_per_mm(calibration_xml_path, camera_identifier):
    """PixelPerMmFactor sits at CoordinateMapper/{Polynomial,Pinhole}
    Parameters/CommonParameters/PixelPerMmFactor -- the parent element
    name depends on DaVis's own calibration Type, but CommonParameters
    is at the same relative depth either way, so '//CommonParameters'
    finds it regardless of type without needing to branch on Type."""
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(f".//CoordinateMapper[@CameraIdentifier='{camera_identifier}']")
    if cm is None:
        raise ValueError(f"'{calibration_xml_path}' has no CoordinateMapper for camera {camera_identifier}")
    el = cm.find(".//CommonParameters/PixelPerMmFactor")
    if el is None:
        raise ValueError(f"'{calibration_xml_path}' camera {camera_identifier} has no PixelPerMmFactor")
    return float(el.attrib["Value"])


def _fit_camera_mapping_planes(mark_table_path, px_per_mm, name_prefix=""):
    """Fit CameraMapping-compatible coefficients (CameraMapping itself is
    UNCHANGED -- same x0/x_span/y0/y_span/dx_coefs/dy_coefs, same formula)
    directly from real calibration-target marks: MarkPositionTable.xml
    pairs each mark's detected RawPos (raw sensor pixel) with its known
    WorldPos (real mm, on the physical calibration plate).

    This deliberately does NOT parse Calibration.xml's own CoefficientsA/
    CoefficientsB -- confirmed empirically (fitting a 3rd-order polynomial
    directly from real marks and comparing coefficients) that DaVis's
    stored coefficients don't match CameraMapping's convention under any
    forward/backward/factor-of-2 variant tried, i.e. they're some other,
    undocumented internal parameterization. Fitting directly against the
    marks sidesteps needing to decode that: verified against real data to
    sub-pixel accuracy (0.3-0.6px RMS, ~16-35 micron) on both a
    Polynomial3rdOrder and a real PinholeOpenCV calibration snapshot --
    this works regardless of which internal model DaVis itself used,
    since MarkPositionTable.xml's raw<->world ground truth is the same
    universal format either way.

    world coordinates are converted mm -> an arbitrary 'world pixel grid'
    via px_per_mm (this app's own free choice of dewarp-target scale,
    same as world_scale_px_per_mm elsewhere) before fitting, since
    CameraMapping's xp/yp are pixel-grid coordinates, not mm.

    Returns a list of 1 or 2 CameraMappingSettings (one per calibrated
    Z-plane found in the mark table, each with z_mm set) -- most DaVis
    stereo calibrations have 2 (a 'dual-plane' fit)."""
    root = ET.parse(mark_table_path).getroot()
    view = root.find(".//View")
    if view is None:
        raise ValueError(f"'{mark_table_path}' has no <View> with mark data")

    by_z_index = {}
    for mark in view.findall("Mark"):
        z_index = mark.find("Index").attrib["z"]
        raw = mark.find("RawPos").attrib
        world = mark.find("WorldPos").attrib
        by_z_index.setdefault(z_index, []).append(
            (float(raw["x"]), float(raw["y"]), float(world["x"]), float(world["y"]), float(world["z"])))

    planes = []
    for z_index in sorted(by_z_index):
        rows = by_z_index[z_index]
        if len(rows) < 20:
            continue  # too few marks for a well-conditioned 10-term fit
        xr = np.array([r[0] for r in rows])
        yr = np.array([r[1] for r in rows])
        xw_mm = np.array([r[2] for r in rows])
        yw_mm = np.array([r[3] for r in rows])
        z_mm = sum(r[4] for r in rows) / len(rows)  # ~constant within one plane

        xw_px, yw_px = xw_mm * px_per_mm, yw_mm * px_per_mm
        x0 = (xw_px.min() + xw_px.max()) / 2
        x_span = xw_px.max() - xw_px.min()
        y0 = (yw_px.min() + yw_px.max()) / 2
        y_span = yw_px.max() - yw_px.min()
        s = 2 * (xw_px - x0) / x_span
        t = 2 * (yw_px - y0) / y_span
        terms = np.column_stack([np.ones_like(s), s, s**2, s**3, t, t**2, t**3, s * t, s**2 * t, s * t**2])
        sol_x, *_ = np.linalg.lstsq(terms, xw_px - xr, rcond=None)
        sol_y, *_ = np.linalg.lstsq(terms, yw_px - yr, rcond=None)
        dx_coefs = dict(zip(COEF_KEYS, (float(v) for v in sol_x)))
        dy_coefs = dict(zip(COEF_KEYS, (float(v) for v in sol_y)))
        planes.append(CameraMappingSettings(
            x0=float(x0), x_span=float(x_span), y0=float(y0), y_span=float(y_span),
            dx_coefs=dx_coefs, dy_coefs=dy_coefs,
            name=f"{name_prefix} z={z_mm:.2f}mm", z_mm=float(z_mm)))

    if not planes:
        raise ValueError(f"'{mark_table_path}' has no Z-plane with enough marks (>=20) to fit")
    return planes


def _files_are_identical(path_a, path_b):
    """Byte-for-byte comparison, used to catch DaVis writing the SAME
    MarkPositionTable.xml into both camera1/ and camera2/ folders --
    confirmed to happen on real project data (every calibration snapshot
    in one real project had this), which would otherwise silently make
    read_stereo_calibration_from_set fit IDENTICAL coefficients for both
    cameras. Cheap and exact -- no need to parse first."""
    try:
        with open(path_a, "rb") as f:
            a = f.read()
        with open(path_b, "rb") as f:
            b = f.read()
    except OSError:
        return False
    return a == b


def _derive_world_shape(cam0_planes, cam1_planes):
    """world_shape must be ONE fixed grid shared by every dewarp call
    regardless of which (possibly interpolated) Z is in use -- size it to
    the largest extent seen across every plane/camera so nothing clips."""
    all_planes = cam0_planes + cam1_planes
    width = max(p.x_span for p in all_planes)
    height = max(p.y_span for p in all_planes)
    return (int(np.ceil(height)), int(np.ceil(width)))


def read_stereo_calibration_from_set(set_path, multiset_index=0):
    """Extract stereo/dewarp calibration for both cameras directly from a
    real DaVis stereo .set project's own calibration-target mark data
    (see _fit_camera_mapping_planes), instead of hand-transcribing it
    from DaVis's on-screen calibration report.

    Locates the project's calibration by walking up from set_path to find
    'Properties/Calibration/' (see _find_calibration_project_root), picks
    whichever calibration snapshot was actually in effect for this
    recording's own timestamp (see _select_calibration_snapshot -- a
    project's calibration can be recalibrated after older recordings were
    taken), then fits each camera's mapping straight from that snapshot's
    real MarkPositionTable.xml ground truth. Works regardless of whether
    DaVis's own internal calibration model was 'Polynomial3rdOrder' or
    'PinholeOpenCV' -- both were verified to fit the mark ground truth
    equally well, since we're not depending on DaVis's own coefficient
    representation at all.

    Raises (never silently wrong) if the project structure, the selected
    snapshot, or its mark data genuinely isn't there -- there's nothing
    sensible to fall back to for a stereo dewarp mapping specifically
    (unlike the planar read_calibration_from_set, where a missing field
    can just stay None)."""
    project_root = _find_calibration_project_root(set_path)
    if project_root is None:
        raise FileNotFoundError(
            f"Couldn't find 'Properties/Calibration/Calibration.xml' walking up from "
            f"'{set_path}' -- this doesn't look like a DaVis project with stereo "
            f"calibration. Enter calibration manually on the Calibration panel."
        )
    recording_dt = _read_set_time(set_path)
    snapshot_dir, snapshot_label = _select_calibration_snapshot(project_root, recording_dt)
    calibration_xml = os.path.join(snapshot_dir, "Calibration.xml")
    if not os.path.isfile(calibration_xml):
        raise FileNotFoundError(
            f"Selected calibration snapshot '{snapshot_label}' ({snapshot_dir}) has no "
            f"Calibration.xml.")

    px_per_mm_0 = _read_pixel_per_mm(calibration_xml, "1")
    px_per_mm_1 = _read_pixel_per_mm(calibration_xml, "2")
    if abs(px_per_mm_0 - px_per_mm_1) > 1e-6 * max(px_per_mm_0, px_per_mm_1):
        print(f"[warn] davis_set: cam0/cam1 PixelPerMmFactor disagree "
              f"({px_per_mm_0} vs {px_per_mm_1}) -- using cam0's")

    cam0_marks = os.path.join(snapshot_dir, "camera1", "MarkPositionTable.xml")
    cam1_marks = os.path.join(snapshot_dir, "camera2", "MarkPositionTable.xml")
    if not os.path.isfile(cam0_marks) or not os.path.isfile(cam1_marks):
        raise FileNotFoundError(
            f"Calibration snapshot '{snapshot_label}' ({snapshot_dir}) has no "
            f"camera1/camera2 MarkPositionTable.xml -- can't fit a dewarp mapping "
            f"from it. Enter calibration manually on the Calibration panel.")
    if _files_are_identical(cam0_marks, cam1_marks):
        raise ValueError(
            f"Calibration snapshot '{snapshot_label}' ({snapshot_dir}) has "
            f"byte-identical camera1/camera2 MarkPositionTable.xml files -- "
            f"confirmed on real DaVis project data that this happens (DaVis "
            f"writing the same mark-detection data into both camera folders "
            f"rather than each camera's own). Fitting from it would silently "
            f"produce IDENTICAL dewarp coefficients for both cameras instead "
            f"of each camera's own real distortion, which is worse than no "
            f"calibration at all. Enter calibration manually on the "
            f"Calibration panel.")

    cam0_planes = _fit_camera_mapping_planes(cam0_marks, px_per_mm_0, name_prefix=f"cam0 (DaVis {snapshot_label})")
    cam1_planes = _fit_camera_mapping_planes(cam1_marks, px_per_mm_1, name_prefix=f"cam1 (DaVis {snapshot_label})")
    world_shape = _derive_world_shape(cam0_planes, cam1_planes)

    return StereoSettings(
        cam0_mapping=cam0_planes[0],
        cam0_mapping_plane2=cam0_planes[1] if len(cam0_planes) > 1 else None,
        cam1_mapping=cam1_planes[0],
        cam1_mapping_plane2=cam1_planes[1] if len(cam1_planes) > 1 else None,
        world_shape=world_shape,
        world_scale_px_per_mm=px_per_mm_0,
        sheet_z_mm=None,
    )


def _read_field_of_view(calibration_xml_path):
    """Calibration.xml's own tag for which acquisition geometry a
    calibration snapshot describes -- "SideBySide2D" for the dual-camera
    planar case this module's DualPlanar* functions target, something
    else (typically absent/different for a plain single-camera planar
    project) otherwise. Returns None if the file has no CoordinateSystem
    element at all."""
    root = ET.parse(calibration_xml_path).getroot()
    cs = root.find(".//CoordinateSystem")
    return cs.attrib.get("FieldOfView") if cs is not None else None


def detect_dual_planar_from_set(set_path, multiset_index=0):
    """Best-effort: does this DaVis .set project's calibration say
    FieldOfView="SideBySide2D" (two coplanar cameras stitched into one
    wider field, config.schema.DualPlanarSettings) rather than a plain
    single-camera planar project? Returns False (never raises) if
    there's no calibration to check, or the check itself fails for any
    reason -- this is used purely to auto-select the GUI's "Dual camera"
    checkbox, matching read_calibration_from_set's "best-effort, no
    field is ever required" contract, not a required calibration gate
    the way read_stereo_calibration_from_set's raise-on-failure is (a
    stereo dewarp mapping is useless if wrong; a wrong initial guess
    about SideBySide2D here just means the user unchecks/checks the box
    themselves)."""
    project_root = _find_calibration_project_root(set_path)
    if project_root is None:
        return False
    recording_dt = _read_set_time(set_path)
    snapshot_dir, _ = _select_calibration_snapshot(project_root, recording_dt)
    calibration_xml = os.path.join(snapshot_dir, "Calibration.xml")
    if not os.path.isfile(calibration_xml):
        return False
    try:
        return _read_field_of_view(calibration_xml) == "SideBySide2D"
    except ET.ParseError:
        return False


def _read_dual_planar_camera(calibration_xml_path, camera_identifier):
    """One camera's RegionWithinCorrectedImage (its raw frame's placement
    within the shared canvas) + OriginalImageSize (its own raw sensor
    size) -- see config.schema.DualPlanarCameraSettings' docstring for
    what these mean and why region_width/height differs slightly from
    raw_width/height (the "corrected" canvas footprint isn't identical
    to the raw sensor size -- lens distortion this feature's flat-scale
    approach approximates rather than removes)."""
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(f".//CoordinateMapper[@CameraIdentifier='{camera_identifier}']")
    if cm is None:
        raise ValueError(f"'{calibration_xml_path}' has no CoordinateMapper for camera {camera_identifier}")
    region = cm.find(".//RegionWithinCorrectedImage")
    original = cm.find(".//OriginalImageSize")
    if region is None or original is None:
        raise ValueError(
            f"'{calibration_xml_path}' camera {camera_identifier} is missing "
            f"RegionWithinCorrectedImage/OriginalImageSize -- not a SideBySide2D "
            f"calibration snapshot?")
    return DualPlanarCameraSettings(
        region_x=float(region.attrib["x"]), region_y=float(region.attrib["y"]),
        region_width=float(region.attrib["Width"]), region_height=float(region.attrib["Height"]),
        raw_width=int(original.attrib["Width"]), raw_height=int(original.attrib["Height"]),
    )


def _read_corrected_image_size(calibration_xml_path, camera_identifier):
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(f".//CoordinateMapper[@CameraIdentifier='{camera_identifier}']")
    el = cm.find(".//CorrectedImageSize") if cm is not None else None
    if el is None:
        raise ValueError(f"'{calibration_xml_path}' camera {camera_identifier} has no CorrectedImageSize")
    return int(el.attrib["Width"]), int(el.attrib["Height"])


def _read_linear_scale(calibration_xml_path, camera_identifier, axis):
    """LinearScaleX/LinearScaleY's FactorMmPerPixel (signed) + OffsetMm,
    converting a shared-canvas pixel coordinate straight to real-world
    mm. axis is 'X' or 'Y'."""
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(f".//CoordinateMapper[@CameraIdentifier='{camera_identifier}']")
    el = cm.find(f".//LinearScale{axis}") if cm is not None else None
    if el is None:
        raise ValueError(f"'{calibration_xml_path}' camera {camera_identifier} has no LinearScale{axis}")
    return float(el.attrib["FactorMmPerPixel"]), float(el.attrib["OffsetMm"])


def read_dual_planar_calibration_from_set(set_path, multiset_index=0):
    """Extract DaVis "SideBySide2D" dual-camera planar calibration (each
    camera's placement within a shared combined canvas, plus that
    canvas's real-world mm scale) directly from a DaVis .set project,
    instead of hand-transcribing it -- the planar counterpart to
    read_stereo_calibration_from_set, sharing its project-root/snapshot-
    selection logic (_find_calibration_project_root,
    _select_calibration_snapshot) since both read the same
    Properties/Calibration/Calibration.xml layout, just different
    CoordinateSystem geometries.

    Unlike read_stereo_calibration_from_set, there's no fitting involved
    here at all -- RegionWithinCorrectedImage/OriginalImageSize/
    LinearScaleX/LinearScaleY are used exactly as DaVis wrote them (see
    config.schema.DualPlanarSettings/DualPlanarCameraSettings). Raises
    (never silently wrong) if the project structure or the selected
    snapshot isn't there, or if its CoordinateSystem isn't
    FieldOfView="SideBySide2D" -- there's nothing sensible to fall back
    to for cam0/cam1 placement specifically, matching
    read_stereo_calibration_from_set's raise-on-failure contract rather
    than read_calibration_from_set's best-effort one."""
    project_root = _find_calibration_project_root(set_path)
    if project_root is None:
        raise FileNotFoundError(
            f"Couldn't find 'Properties/Calibration/Calibration.xml' walking up from "
            f"'{set_path}' -- this doesn't look like a DaVis project with dual-camera "
            f"planar calibration.")
    recording_dt = _read_set_time(set_path)
    snapshot_dir, snapshot_label = _select_calibration_snapshot(project_root, recording_dt)
    calibration_xml = os.path.join(snapshot_dir, "Calibration.xml")
    if not os.path.isfile(calibration_xml):
        raise FileNotFoundError(
            f"Selected calibration snapshot '{snapshot_label}' ({snapshot_dir}) has no "
            f"Calibration.xml.")

    field_of_view = _read_field_of_view(calibration_xml)
    if field_of_view != "SideBySide2D":
        raise ValueError(
            f"'{calibration_xml}' CoordinateSystem FieldOfView is {field_of_view!r}, "
            f"not 'SideBySide2D' -- this doesn't look like a dual-camera planar "
            f"project (use detect_dual_planar_from_set to check before calling this).")

    cam0 = _read_dual_planar_camera(calibration_xml, "1")
    cam1 = _read_dual_planar_camera(calibration_xml, "2")
    canvas_w0, canvas_h0 = _read_corrected_image_size(calibration_xml, "1")
    canvas_w1, canvas_h1 = _read_corrected_image_size(calibration_xml, "2")
    if (canvas_w0, canvas_h0) != (canvas_w1, canvas_h1):
        print(f"[warn] davis_set: cam0/cam1 CorrectedImageSize disagree "
              f"({canvas_w0}x{canvas_h0} vs {canvas_w1}x{canvas_h1}) -- using cam0's")

    scale_x_mm_per_px, scale_x_offset_mm = _read_linear_scale(calibration_xml, "1", "X")
    scale_y_mm_per_px, scale_y_offset_mm = _read_linear_scale(calibration_xml, "1", "Y")
    scale_x1, offset_x1 = _read_linear_scale(calibration_xml, "2", "X")
    scale_y1, offset_y1 = _read_linear_scale(calibration_xml, "2", "Y")
    if (abs(scale_x1 - scale_x_mm_per_px) > 1e-9 * abs(scale_x_mm_per_px)
            or abs(offset_x1 - scale_x_offset_mm) > 1e-6 * max(1.0, abs(scale_x_offset_mm))):
        print(f"[warn] davis_set: cam0/cam1 LinearScaleX disagree -- using cam0's")
    if (abs(scale_y1 - scale_y_mm_per_px) > 1e-9 * abs(scale_y_mm_per_px)
            or abs(offset_y1 - scale_y_offset_mm) > 1e-6 * max(1.0, abs(scale_y_offset_mm))):
        print(f"[warn] davis_set: cam0/cam1 LinearScaleY disagree -- using cam0's")

    return DualPlanarSettings(
        enabled=True, cam0=cam0, cam1=cam1,
        canvas_width=canvas_w0, canvas_height=canvas_h0,
        scale_x_mm_per_px=scale_x_mm_per_px, scale_x_offset_mm=scale_x_offset_mm,
        scale_y_mm_per_px=scale_y_mm_per_px, scale_y_offset_mm=scale_y_offset_mm,
    )
