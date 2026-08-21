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
import sys
import xml.etree.ElementTree as ET

from piv_suite.config.schema import CalibrationSettings

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


def read_stereo_calibration_from_set(set_path, multiset_index=0):
    """Extract stereo/dewarp calibration (per-camera CameraMappingSettings
    polynomial coefficients, world grid, viewing angles) directly from a
    DaVis stereo .set project, instead of hand-transcribing it from
    DaVis's calibration report.

    Not implemented -- unlike read_calibration_from_set (whose file
    layout was reverse-engineered against a real local dataset), no real
    stereo/dewarp DaVis dataset has been available to determine the
    actual calibration file format. This function exists purely so GUI
    plumbing (e.g. a disabled "Load from .set..." button on the
    Calibration panel) can be wired to a defined interface now, matching
    calibration.report_parser.parse_davis_calibration_report's stub
    pattern, without needing UI rework once a real dataset surfaces."""
    raise NotImplementedError(
        "Automated stereo calibration extraction from a DaVis .set isn't "
        "implemented yet -- no real stereo/dewarp dataset has been "
        "available to reverse-engineer the file format. Enter calibration "
        "coefficients manually on the Calibration panel (or via 'Load "
        "from DaVis report...' once that parser is implemented)."
    )
