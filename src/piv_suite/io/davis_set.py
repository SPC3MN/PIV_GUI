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

from piv_suite.calibration.camera_mapping import COEF_KEYS, CameraMapping
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


# DaVis's Polynomial3rdOrder CoefficientsA/CoefficientsB attribute suffixes,
# in the SAME order as CameraMapping.COEF_KEYS -- the two naming schemes
# refer to the identical 10 terms (1, s, s2, s3, t, t2, t3, st, s2t, s*t2),
# DaVis just spells the last one 'a_st2' (s*t^2) where CameraMapping spells
# it 't2s' (t^2*s) -- same term, cosmetic difference only. See
# _exact_camera_mapping_from_calibration_xml's docstring for how this
# mapping was determined to be exact (not assumed).
_POLY3_COEF_SUFFIXES = ("o", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "st2")


def _exact_camera_mapping_from_calibration_xml(calibration_xml_path, camera_identifier, name_prefix=""):
    """Decode a camera's dewarp mapping EXACTLY from DaVis's own stored
    Calibration.xml -- no fitting, no marks needed at all -- for the
    'Polynomial3rdOrder' calibration Type. Returns None if this camera's
    CoordinateMapper isn't that Type (e.g. 'PinholeOpenCV', a genuinely
    different, standard-pinhole-camera parameterization that was tried and
    NOT successfully decoded to sub-pixel precision -- see
    read_stereo_calibration_from_set's docstring) or has no usable
    <PolynomialMapping> plane data.

    THE DECODE (verified against real ground truth, not assumed): DaVis
    stores, per calibrated Z-plane, an <Origin s_o t_o>, a
    <NormalisationFactor nx ny>, and <CoefficientsA>/<CoefficientsB> --
    which turn out to be EXACTLY CameraMapping's own x0/y0, x_span/y_span,
    and dx_coefs/dy_coefs, with ZERO transformation needed, PROVIDED the
    (xp, yp) fed into CameraMapping.world_to_raw are DaVis's own 'corrected
    image' pixel coordinates (the same space CorrectedImageSize describes),
    not some other world-pixel convention. Confirmed by taking real marks'
    WorldPos (mm) from MarkPositionTable.xml, converting to corrected-image
    pixels via the INVERSE of that same CoordinateMapper's own
    LinearScaleX/Y (Cx = (world_mm_x - OffsetMm) / FactorMmPerPixel, same
    for y), constructing a real CameraMapping(x0=s_o, x_span=nx, y0=t_o,
    y_span=ny, dx_coefs=CoefficientsA, dy_coefs=CoefficientsB) UNCHANGED,
    calling its real .world_to_raw(Cx, Cy), and comparing the result to
    that same mark's real RawPos: 0.29-0.69px RMS across 2 real
    Polynomial3rdOrder calibration snapshots (both Z-planes each,
    J:\\Final_Stereo's current calibration and its
    Calibration_260713_181401 History snapshot) -- matching DaVis's own
    reported <FitError RMS> for those snapshots almost exactly, i.e. our
    residual IS DaVis's own fit noise, not decode error. A third
    Polynomial3rdOrder snapshot (Calibration_260713_191850) showed a
    LINEAR (affine) discrepancy on top of this instead (~5-7px before
    accounting for it, ~0.3-0.6px after subtracting a fitted affine
    offset) -- diagnosed as that specific snapshot's own
    MarkPositionTable.xml not being paired with the SAME LinearScaleX/Y
    used to derive Cx/Cy for it (a data-consistency quirk in that one
    History folder, the same general category of real DaVis inconsistency
    already documented for the byte-duplicated camera1/camera2 mark files
    elsewhere in this module) -- NOT a failure of the decode formula
    itself, since removing a simple affine trend recovers the same
    sub-pixel residual seen on the other two clean snapshots.

    This was NOT tried by the previous session (see SESSION_HANDOFF.md /
    the module's git history) -- that attempt tested CoefficientsA/B as a
    DELTA correction against an arbitrary 'world pixel = world_mm *
    px_per_mm' grid with no offset (CameraMapping's existing convention,
    the one _fit_camera_mapping_planes's marks-based fit ALSO uses) and
    concluded it didn't match under any variant tried. It doesn't, under
    THAT world-pixel convention -- DaVis's Origin/NormalisationFactor are
    defined in ITS OWN 'corrected image' pixel space (offset from the
    simple mm*px_per_mm grid by a per-snapshot constant baked into
    LinearScaleX/Y's OffsetMm), not the app's. Once fed genuinely
    corrected-image-space (xp, yp) -- not w a substitute grid -- the exact
    same CameraMapping formula the previous session already ruled out
    (correctly, for a DIFFERENT grid) turns out to be exactly right.

    world_shape for CameraMapping.dewarp_image doesn't need to match any
    particular mm origin (stereo triangulation in
    processing.pipeline.combine_stereo_pair only ever uses DISPLACEMENTS
    between two dewarped frames of the SAME pair, warped through the SAME
    fixed grid -- any constant offset in the grid's absolute origin
    cancels out of a displacement and never appears in the output U/V/W).
    So CorrectedImageSize (DaVis's own choice of how large a canvas this
    camera's mapping is valid over -- returned here alongside the planes)
    is used as-is for world_shape, not re-derived from x_span/y_span the
    way the marks-fit path's _derive_world_shape does.

    Returns (planes, pixel_per_mm, corrected_image_wh) where planes is a
    list of 1 or 2 CameraMappingSettings (one per <PolynomialMapping>,
    z_mm recovered as ZPosition/PixelPerMmFactor -- DaVis stores ZPosition
    pre-scaled by PixelPerMmFactor, confirmed exactly: 17.920975188143096
    /17.920975188143096 == 1.0 for a real z=1mm plane, -35.841950.../
    17.920975188143096 == -2.0 for a real z=-2mm plane; each plane also
    carries this camera's raw_width/raw_height, read from this same
    CommonParameters block's OriginalImageSize -- see CameraMappingSettings'
    own comment), or None if this camera isn't Polynomial3rdOrder or has
    no usable plane."""
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(f".//CoordinateMapper[@CameraIdentifier='{camera_identifier}']")
    if cm is None or cm.attrib.get("Type") != "Polynomial3rdOrder":
        return None
    common = cm.find(".//CommonParameters")
    ppm_el = common.find("PixelPerMmFactor") if common is not None else None
    corrected_el = common.find("CorrectedImageSize") if common is not None else None
    if ppm_el is None or corrected_el is None:
        return None
    ppm = float(ppm_el.attrib["Value"])
    corrected_wh = (float(corrected_el.attrib["Width"]), float(corrected_el.attrib["Height"]))

    # OriginalImageSize is this camera's real raw sensor size -- sibling of
    # PixelPerMmFactor/CorrectedImageSize in the same CommonParameters
    # block (confirmed on real data: J:\Final_Stereo\Properties\
    # Calibration\Calibration.xml, Width=4096 Height=3008), the same
    # element _read_dual_planar_camera already reads for the SideBySide2D
    # path. A camera-level constant, not per-plane -- read once here and
    # stamped onto every CameraMappingSettings this function returns for
    # this camera (see CameraMappingSettings.raw_width/raw_height's own
    # comment for what it's used for: calibration.camera_mapping.
    # CameraMapping.raw_domain_valid). Missing (0, 0) rather than raising
    # if this element isn't there for some reason -- matches this field's
    # "0 = unknown, no masking possible" contract instead of making an
    # otherwise-valid exact-decode calibration fail over a field that's
    # only an OPTIONAL refinement (FOV masking), not required for dewarp
    # itself.
    original_el = common.find("OriginalImageSize")
    raw_wh = ((int(float(original_el.attrib["Width"])), int(float(original_el.attrib["Height"])))
              if original_el is not None else (0, 0))

    planes = []
    for pm in cm.findall(".//PolynomialMapping"):
        z_el, origin_el, norm_el, poly3_el = (
            pm.find("ZPosition"), pm.find("Origin"), pm.find("NormalisationFactor"), pm.find("Polynomial3rdOrder"))
        if None in (z_el, origin_el, norm_el, poly3_el):
            continue
        a_el, b_el = poly3_el.find("CoefficientsA"), poly3_el.find("CoefficientsB")
        if a_el is None or b_el is None:
            continue
        dx_coefs = {key: float(a_el.attrib[f"a_{suffix}"]) for key, suffix in zip(COEF_KEYS, _POLY3_COEF_SUFFIXES)}
        dy_coefs = {key: float(b_el.attrib[f"b_{suffix}"]) for key, suffix in zip(COEF_KEYS, _POLY3_COEF_SUFFIXES)}
        z_mm = float(z_el.attrib["Value"]) / ppm
        planes.append(CameraMappingSettings(
            x0=float(origin_el.attrib["s_o"]), x_span=float(norm_el.attrib["nx"]),
            y0=float(origin_el.attrib["t_o"]), y_span=float(norm_el.attrib["ny"]),
            dx_coefs=dx_coefs, dy_coefs=dy_coefs,
            name=f"{name_prefix} z={z_mm:.2f}mm (DaVis Polynomial3rdOrder, exact)", z_mm=z_mm,
            raw_width=raw_wh[0], raw_height=raw_wh[1]))

    if not planes:
        return None
    return planes, ppm, corrected_wh


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
    """Extract stereo/dewarp calibration for both cameras EXACTLY from a
    real DaVis stereo .set project's own Calibration.xml -- no fitting, no
    MarkPositionTable.xml needed at all -- when the applicable calibration
    snapshot is DaVis's 'Polynomial3rdOrder' Type (see
    _exact_camera_mapping_from_calibration_xml's docstring for the full
    decode derivation and its real-ground-truth validation: 0.29-0.69px
    RMS across 2 clean real snapshots, both Z-planes each).

    A SECOND DaVis internal calibration Type exists, 'PinholeOpenCV' --
    the standard OpenCV pinhole camera model (FocalLengthPixel,
    PrincipalPoint, Radial/TangentialDistortion, TranslationMm,
    RotationAngles). This WAS attempted (a Rodrigues-rotation projection
    with Brown-Conrady distortion, plus Euler-angle rotation variants,
    focal-length unit conversions, sign/axis permutations -- see this
    module's git history / SESSION_HANDOFF.md for the exact sweep) and
    got CLOSE but not exact: with a Z-Y-X Euler rotation order, the
    y-pixel comes out already sub-pixel (0.7px RMS, matching
    <FitError RMS>) but x is off by several px in a way that isn't a pure
    additive constant (varies ~2.5px across the calibration volume even
    after removing the best-fit affine trend) -- i.e. genuinely not
    decoded, not merely a units/sign slip like the corrected-image-space
    discovery that cracked Polynomial3rdOrder. PinholeOpenCV therefore
    has NO exact path here and raises (see below) rather than silently
    reusing the mark-fitting approach as if it were the automatic answer.

    Locates the project's calibration by walking up from set_path to find
    'Properties/Calibration/' (see _find_calibration_project_root), picks
    whichever calibration snapshot was actually in effect for this
    recording's own timestamp (see _select_calibration_snapshot -- a
    project's calibration can be recalibrated after older recordings were
    taken), then decodes each camera's mapping straight from that
    snapshot's Calibration.xml.

    Deliberately does NOT fall back to _fit_camera_mapping_planes's
    least-squares mark fit when exact decode isn't available (wrong
    calibration Type, or the plane/coefficient data isn't there) --
    that fit is real, tested, and still in this module, but presenting it
    as this function's AUTOMATIC answer would mean silently handing back
    an approximate result where the user asked for exact values or
    nothing. Raises instead (never silently wrong, and never silently
    approximate) so the GUI's existing never-crash status-bar wiring
    falls through to manual calibration entry -- the same behavior
    duplicate-marks-refusal already relied on before this function
    stopped needing marks at all."""
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

    # A DaVis "SideBySide2D" calibration (two COPLANAR cameras stitched into
    # one wider field -- config.schema.DualPlanarSettings, detected by
    # detect_dual_planar_from_set) uses the EXACT SAME CoordinateMapper/
    # CoefficientsA/CoefficientsB XML shape as a real angled stereo pair
    # ("SideBySideStereoVolume" on a real project) -- confirmed by directly
    # comparing a real SideBySide2D Calibration.xml (D:\Truck_PIV_Round4)
    # against a real stereo one (J:\Final_Stereo): both cameras' polynomial
    # blocks parse identically either way. Without this check,
    # _exact_camera_mapping_from_calibration_xml would happily decode a
    # SideBySide2D project's per-camera mapping and this function would
    # silently hand it back AS IF it were a valid stereo triangulation pair
    # -- geometrically wrong (coplanar cameras have no baseline angle to
    # triangulate a W component from) even though every individual number
    # decoded correctly. Caught via real GUI testing: selecting a real
    # SideBySide2D project's .set while Mode=Stereo extracted a "Stereo
    # calibration ... exact" status message with no error at all.
    field_of_view = _read_field_of_view(calibration_xml)
    if field_of_view == "SideBySide2D":
        raise ValueError(
            f"Calibration snapshot '{snapshot_label}' ({snapshot_dir}) is a DaVis "
            f"'SideBySide2D' calibration -- two coplanar cameras stitched into one "
            f"wider field, not an angled stereo pair. Use Planar mode with 'Dual "
            f"camera (SideBySide)' checked instead of Stereo mode for this project.")

    pinhole = _pinhole_stereo_settings_from_calibration_xml(
        calibration_xml, snapshot_dir, snapshot_label)
    if pinhole is not None:
        return pinhole

    cam0_exact = _exact_camera_mapping_from_calibration_xml(
        calibration_xml, "1", name_prefix=f"cam0 (DaVis {snapshot_label})")
    cam1_exact = _exact_camera_mapping_from_calibration_xml(
        calibration_xml, "2", name_prefix=f"cam1 (DaVis {snapshot_label})")
    if cam0_exact is None or cam1_exact is None:
        raise NotImplementedError(
            f"Calibration snapshot '{snapshot_label}' ({snapshot_dir}) is neither a "
            f"'Polynomial3rdOrder' nor a 'PinholeOpenCV' DaVis calibration for both "
            f"cameras (or is missing plane data) -- those are the two models this app "
            f"decodes exactly. A least-squares fit from calibration marks exists in "
            f"this module (_fit_camera_mapping_planes) but isn't exact, so it isn't "
            f"used here automatically. Enter calibration manually on the Calibration "
            f"panel.")

    # A self-calibration CORRECTION FIELD makes the stored polynomial only a
    # BASE LAYER: the true mapping is polynomial + correction, and decoding
    # the base alone is silently wrong (measured 16-32 px on this project's
    # own snapshots) with nothing in the file marking it. Refuse rather than
    # hand back a plausible-looking mapping that is tens of pixels out --
    # every downstream vector would be wrong with no signal anywhere.
    if os.path.isdir(os.path.join(snapshot_dir, "Correction field")):
        raise NotImplementedError(
            f"Calibration snapshot '{snapshot_label}' ({snapshot_dir}) is a "
            f"'Polynomial3rdOrder' calibration with a self-calibration CORRECTION "
            f"FIELD ('Correction field\\' is present). The stored polynomial is only "
            f"the base layer -- DaVis applies the correction field on top of it, and "
            f"this app does not read that yet, so decoding the polynomial alone would "
            f"be silently wrong (measured 16-32 px on real snapshots). Use a "
            f"calibration snapshot without a correction field, or enter calibration "
            f"manually on the Calibration panel.")

    cam0_planes, px_per_mm_0, corrected_wh_0 = cam0_exact
    cam1_planes, px_per_mm_1, corrected_wh_1 = cam1_exact
    if abs(px_per_mm_0 - px_per_mm_1) > 1e-6 * max(px_per_mm_0, px_per_mm_1):
        print(f"[warn] davis_set: cam0/cam1 PixelPerMmFactor disagree "
              f"({px_per_mm_0} vs {px_per_mm_1}) -- using cam0's")

    # CorrectedImageSize is DaVis's own choice of shared canvas size for
    # BOTH cameras (confirmed identical between camera1/camera2's
    # CoordinateMapper on real data) -- take the larger of the two (should
    # match exactly) rather than assuming which one to trust.
    world_shape = (int(np.ceil(max(corrected_wh_0[1], corrected_wh_1[1]))),
                   int(np.ceil(max(corrected_wh_0[0], corrected_wh_1[0]))))

    # All four triangulation angles are left None: they are no longer a
    # scalar at all. calibration.camera_mapping.stereo_view_angles derives
    # them PER PIXEL from this same calibration at each processing entry
    # point (CameraMapping.view_angles recovers the viewing ray from the two
    # calibrated Z-planes). Setting any of them here would silently override
    # that derivation with a worse global approximation -- see
    # config.schema.StereoSettings.alpha1_deg's comment for the measured
    # cost of a global angle, and for why the previous auto-derive
    # (_estimate_stereo_angles, deleted) was ~10 deg wrong.
    return StereoSettings(
        cam0_mapping=cam0_planes[0],
        cam0_mapping_plane2=cam0_planes[1] if len(cam0_planes) > 1 else None,
        cam1_mapping=cam1_planes[0],
        cam1_mapping_plane2=cam1_planes[1] if len(cam1_planes) > 1 else None,
        world_shape=world_shape,
        world_scale_px_per_mm=px_per_mm_0,
        alpha1_deg=None, alpha2_deg=None, beta1_deg=None, beta2_deg=None,
        sheet_z_mm=None,
    )


def _pinhole_stereo_settings_from_calibration_xml(calibration_xml, snapshot_dir, snapshot_label):
    """StereoSettings for a DaVis 'PinholeOpenCV' snapshot, or None if this
    snapshot isn't that Type for both cameras.

    Every decoded convention (and its validation against real
    MarkPositionTable.xml ground truth) is documented in
    calibration.pinhole's module docstring. Summary of why this is exact
    where a previous attempt concluded it wasn't: FocalLengthPixel is
    stored in MILLIMETRES, PrincipalPoint is the projection AND distortion
    centre, OriginPixelPosition is a canvas quantity that plays no part in
    the projection, the rotation is Euler zyx, and the distortion is plain
    OpenCV Brown-Conrady on normalized coordinates. Used verbatim, those
    reproduce DaVis's own declared <FitError RMS> to a relative difference
    of ~1e-14 on this project's snapshots.

    Unlike the polynomial model there is no per-Z-plane coefficient set: a
    pinhole camera is valid at any Z by construction, so sheet_z_mm is a
    plain evaluation parameter rather than an interpolation weight, and no
    second plane is needed."""
    from ..calibration.pinhole import read_pinhole_camera
    from ..config.schema import PinholeMappingSettings

    cams = []
    for ident, label in (("1", "cam0"), ("2", "cam1")):
        cam = read_pinhole_camera(calibration_xml, ident,
                                  name=f"{label} (DaVis {snapshot_label}, PinholeOpenCV, exact)")
        if cam is None:
            return None
        cams.append(cam)
    cam0, cam1 = cams

    _warn_if_marks_disagree(snapshot_dir, snapshot_label, cams)

    def to_settings(cam, rx_ry_rz):
        rx, ry, rz = rx_ry_rz
        return PinholeMappingSettings(
            f_px=cam.f_px, cx=cam.cx, cy=cam.cy,
            k1=cam.k1, k2=cam.k2, p1=cam.p1, p2=cam.p2,
            rx=rx, ry=ry, rz=rz,
            tx=cam.T[0], ty=cam.T[1], tz=cam.T[2],
            scale_x=cam.scale_x, scale_y=cam.scale_y,
            offset_x=cam.offset_x, offset_y=cam.offset_y,
            name=cam.name, raw_width=cam.raw_width, raw_height=cam.raw_height,
            fit_rms=cam.fit_rms)

    root = ET.parse(calibration_xml).getroot()
    eulers = []
    for ident in ("1", "2"):
        ro = root.find(f".//CoordinateMapper[@CameraIdentifier='{ident}']"
                       f"//ExternalCameraParameters/RotationAngles").attrib
        eulers.append((float(ro["Rx"]), float(ro["Ry"]), float(ro["Rz"])))

    # Both cameras share DaVis's own corrected-canvas size and scale
    # (confirmed identical between camera1/camera2 on real data), same as
    # the polynomial path -- take the larger of the two rather than
    # assuming which to trust.
    world_shape = (int(np.ceil(max(cam0.corrected_wh[1], cam1.corrected_wh[1]))),
                   int(np.ceil(max(cam0.corrected_wh[0], cam1.corrected_wh[0]))))
    return StereoSettings(
        cam0_pinhole=to_settings(cam0, eulers[0]),
        cam1_pinhole=to_settings(cam1, eulers[1]),
        world_shape=world_shape,
        world_scale_px_per_mm=1.0 / abs(cam0.scale_x),
        alpha1_deg=None, alpha2_deg=None, beta1_deg=None, beta2_deg=None,
        sheet_z_mm=None,
    )


def _warn_if_marks_disagree(snapshot_dir, snapshot_label, cams):
    """Check each decoded camera against its own snapshot's
    MarkPositionTable.xml and warn if the reprojection RMS doesn't match
    DaVis's declared <FitError RMS>.

    This is a real staleness detector, not a formality. When a coordinate
    system is re-datumed after calibrating, DaVis writes new extrinsics and
    a new canvas but BYTE-COPIES the old mark table, so the marks are then
    expressed in the previous world frame -- reprojecting them gives a
    large RMS (21.5 px measured) even though the calibration itself is
    perfectly self-consistent and correct to use. So this warns rather than
    raising: the projection never needs the marks (extrinsics and the
    canvas offsets always share one frame), they are only the independent
    check.

    NOTE the file layout, which is easy to get wrong: DaVis writes the SAME
    MarkPositionTable.xml into camera1\\ and camera2\\, and each copy holds
    BOTH cameras' blocks. Selecting by folder rather than by CameraNumber
    silently pairs one camera's marks with the other's parameters -- that
    mistake is what previously made this model look undecodable."""
    from ..calibration.pinhole import read_marks, weighted_reprojection_rms
    for i, cam in enumerate(cams):
        number = i + 1
        path = os.path.join(snapshot_dir, f"camera{number}", "MarkPositionTable.xml")
        if not os.path.isfile(path) or cam.fit_rms is None:
            continue
        try:
            marks = read_marks(path, number)
            if marks is None or not len(marks[0]):
                continue
            rms = weighted_reprojection_rms(cam, marks)
        except Exception as exc:                       # noqa: BLE001 - advisory only
            print(f"[warn] davis_set: couldn't cross-check camera{number}'s marks "
                  f"for '{snapshot_label}': {exc}")
            continue
        if rms > max(1.5 * cam.fit_rms, cam.fit_rms + 0.25):
            print(f"[warn] davis_set: camera{number}'s MarkPositionTable.xml reprojects "
                  f"at {rms:.3f} px but '{snapshot_label}' declares FitError RMS "
                  f"{cam.fit_rms:.3f} px. That normally means the mark table is STALE "
                  f"(the coordinate system was re-datumed after calibrating and DaVis "
                  f"copied the old table forward), which does NOT affect the "
                  f"calibration itself -- the extrinsics and canvas offsets share one "
                  f"frame and are used directly. Proceeding.")


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


def detect_project_type_from_set(set_path, multiset_index=0):
    """Best-effort: which acquisition geometry does this DaVis .set
    project's OWN calibration describe -- "planar" (plain single-camera,
    the existing default), "stereo" (two angled cameras triangulated into
    3-component U/V/W, CoordinateSystem/@FieldOfView ==
    "SideBySideStereoVolume" on a real project), or "dual_planar" (two
    COPLANAR cameras stitched into one wider planar field, FieldOfView ==
    "SideBySide2D", config.schema.DualPlanarSettings -- same signal
    detect_dual_planar_from_set already checks)?

    Used purely to steer the GUI's Mode radio (and, via the separate,
    already-existing detect_dual_planar_from_set call, the "Dual camera"
    checkbox) the moment a .set is selected -- instead of requiring the
    user to correctly guess Planar vs Stereo mode BEFORE selecting a
    project, which is what made calibration auto-extraction only ever
    look in the right place by accident. Matches detect_dual_planar_
    from_set's own contract exactly: never raises, always returns a
    definite answer, falling back to "planar" (the existing, unchanged
    default) for anything it can't positively identify -- no calibration
    found, a single-camera project, or a FieldOfView value that's neither
    of the two recognized ones -- since a wrong initial guess here just
    means the user picks the right radio themselves, same as every other
    auto-extracted/auto-guessed field in this app.

    Deliberately does NOT try to distinguish "genuinely no calibration"
    from "unrecognized FieldOfView" -- both collapse to "planar" here,
    since a real angled-stereo/dual-planar project always writes one of
    the two recognized values (confirmed against real projects of both
    kinds, see read_stereo_calibration_from_set's docstring) and nothing
    else meaningfully implies either one."""
    project_root = _find_calibration_project_root(set_path)
    if project_root is None:
        return "planar"
    recording_dt = _read_set_time(set_path)
    snapshot_dir, _ = _select_calibration_snapshot(project_root, recording_dt)
    calibration_xml = os.path.join(snapshot_dir, "Calibration.xml")
    if not os.path.isfile(calibration_xml):
        return "planar"
    try:
        field_of_view = _read_field_of_view(calibration_xml)
    except ET.ParseError:
        return "planar"
    if field_of_view == "SideBySide2D":
        return "dual_planar"
    if field_of_view == "SideBySideStereoVolume":
        return "stereo"
    return "planar"


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
