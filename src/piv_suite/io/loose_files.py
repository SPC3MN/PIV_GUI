"""Loose-folder ingestion (planar and stereo): a plain folder of image
files, not a DaVis `.set` project.

Migrated from piv_common.iter_pairs_from_loose_files and
stereo_common.iter_stereo_from_loose_files. The "combined double/quad-frame
buffer per pair" branch is inherently a LaVision .im7 concept and stays
lvpyio-only; the "frame A/B are separate files matched by suffix" branch is
generalized here to use io.readers.read_frame, so plain labeled TIFF/PNG
image pairs work too, not just .im7 (see io/readers.py).
"""

import os
import sys
import glob

from .buffers import frames_from_buffer, frames_from_stereo_buffer
from .readers import read_frame


def iter_pairs_from_loose_files(input_path, loose_glob="*.im7",
                                 suffix_a="_a.im7", suffix_b="_b.im7"):
    """Yield (pair_id, frame_a, frame_b) from a plain folder of image
    files -- either double-frame .im7 buffers (one file per pair), or
    frame A/frame B as separate files (any supported format) matched by
    suffix."""
    paths = sorted(glob.glob(os.path.join(input_path, loose_glob)))
    if not paths:
        sys.exit(f"No files matching '{loose_glob}' in '{input_path}'")

    is_im7 = os.path.splitext(paths[0])[1].lower() == ".im7"
    double_frame = False
    if is_im7:
        import lvpyio as lv
        double_frame = len(lv.read_buffer(paths[0]).frames) >= 2

    if double_frame:
        import lvpyio as lv
        for path in paths:
            pair_id = os.path.splitext(os.path.basename(path))[0]
            buf = lv.read_buffer(path)
            frame_a, frame_b = frames_from_buffer(buf)
            yield pair_id, frame_a, frame_b
    else:
        # frame A / frame B are separate single-frame files, matched by
        # suffix -- compared case-INsensitively, since Windows/macOS
        # filesystems are case-insensitive and users naturally expect
        # "_a.tif" and "_a.TIF" to be treated the same
        files_a = sorted(p for p in paths if p.lower().endswith(suffix_a.lower()))
        if not files_a:
            sys.exit(
                f"Files are single-frame but none end in '{suffix_a}' "
                "-- set suffix_a/suffix_b to match your naming"
            )
        for path_a in files_a:
            path_b = path_a[: -len(suffix_a)] + suffix_b
            if not os.path.exists(path_b):
                print(f"[warn] no match for {os.path.basename(path_a)} "
                      f"(expected {os.path.basename(path_b)}) -- skipping")
                continue
            pair_id = os.path.basename(path_a)[: -len(suffix_a)]
            frame_a = read_frame(path_a)
            frame_b = read_frame(path_b)
            yield pair_id, frame_a, frame_b


def iter_stereo_from_loose_files(input_path, loose_glob="*.im7",
                                  suffix_cam0="_cam1.im7", suffix_cam1="_cam2.im7",
                                  stereo_frame_order="camera_major"):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a plain folder of .im7
    files -- either one combined 4-frame file per stereo pair, or each
    camera's double-frame pair as a separate file (auto-detected)."""
    import lvpyio as lv
    paths = sorted(glob.glob(os.path.join(input_path, loose_glob)))
    if not paths:
        sys.exit(f"No files matching '{loose_glob}' in '{input_path}'")

    first_buf = lv.read_buffer(paths[0])
    combined = len(first_buf.frames) >= 4

    if combined:
        for path in paths:
            pair_id = os.path.splitext(os.path.basename(path))[0]
            buf = lv.read_buffer(path)
            fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, stereo_frame_order)
            yield pair_id, fa0, fb0, fa1, fb1
    else:
        # each camera's double-frame pair is a SEPARATE file, matched by
        # suffix -- case-insensitive, see the comment in
        # iter_pairs_from_loose_files above
        files0 = sorted(p for p in paths if p.lower().endswith(suffix_cam0.lower()))
        if not files0:
            sys.exit(
                f"Files aren't combined 4-frame stereo buffers but none end "
                f"in '{suffix_cam0}' -- set suffix_cam0/suffix_cam1 to "
                "match your naming"
            )
        for path0 in files0:
            path1 = path0[: -len(suffix_cam0)] + suffix_cam1
            if not os.path.exists(path1):
                print(f"[warn] no match for {os.path.basename(path0)} "
                      f"(expected {os.path.basename(path1)}) -- skipping")
                continue
            pair_id = os.path.basename(path0)[: -len(suffix_cam0)]
            fa0, fb0 = frames_from_buffer(lv.read_buffer(path0))
            fa1, fb1 = frames_from_buffer(lv.read_buffer(path1))
            yield pair_id, fa0, fb0, fa1, fb1


def _resolve_planar_pairs(input_path, loose_glob, suffix_a, suffix_b):
    """Shared file-matching for the planar list/get helpers below --
    filesystem/glob work only, no image data read. Returns a list of
    (pair_id, path_a, path_b) for double_frame=False, or a list of
    (pair_id, path) single-file entries for double_frame=True, alongside
    the double_frame flag."""
    paths = sorted(glob.glob(os.path.join(input_path, loose_glob)))
    if not paths:
        return [], False

    is_im7 = os.path.splitext(paths[0])[1].lower() == ".im7"
    double_frame = False
    if is_im7:
        import lvpyio as lv
        double_frame = len(lv.read_buffer(paths[0]).frames) >= 2

    if double_frame:
        entries = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    else:
        files_a = sorted(p for p in paths if p.lower().endswith(suffix_a.lower()))
        entries = []
        for path_a in files_a:
            path_b = path_a[: -len(suffix_a)] + suffix_b
            if not os.path.exists(path_b):
                continue
            entries.append((os.path.basename(path_a)[: -len(suffix_a)], path_a, path_b))
    return entries, double_frame


def list_pair_ids_from_loose_files(input_path, loose_glob="*.im7", suffix_a="_a.im7", suffix_b="_b.im7"):
    """Cheap: just the pair ids (e.g. for a GUI pair-picker), without
    reading any image data."""
    entries, _ = _resolve_planar_pairs(input_path, loose_glob, suffix_a, suffix_b)
    return [e[0] for e in entries]


def get_pair_from_loose_files(input_path, index, loose_glob="*.im7", suffix_a="_a.im7", suffix_b="_b.im7"):
    """Load a single (pair_id, frame_a, frame_b) at list position
    `index` (as returned by list_pair_ids_from_loose_files) directly."""
    entries, double_frame = _resolve_planar_pairs(input_path, loose_glob, suffix_a, suffix_b)
    if double_frame:
        pair_id, path = entries[index]
        import lvpyio as lv
        frame_a, frame_b = frames_from_buffer(lv.read_buffer(path))
    else:
        pair_id, path_a, path_b = entries[index]
        frame_a = read_frame(path_a)
        frame_b = read_frame(path_b)
    return pair_id, frame_a, frame_b


def _resolve_stereo_pairs(input_path, loose_glob, suffix_cam0, suffix_cam1):
    """Stereo counterpart to _resolve_planar_pairs."""
    paths = sorted(glob.glob(os.path.join(input_path, loose_glob)))
    if not paths:
        return [], False

    import lvpyio as lv
    combined = len(lv.read_buffer(paths[0]).frames) >= 4

    if combined:
        entries = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    else:
        files0 = sorted(p for p in paths if p.lower().endswith(suffix_cam0.lower()))
        entries = []
        for path0 in files0:
            path1 = path0[: -len(suffix_cam0)] + suffix_cam1
            if not os.path.exists(path1):
                continue
            entries.append((os.path.basename(path0)[: -len(suffix_cam0)], path0, path1))
    return entries, combined


def list_pair_ids_stereo_from_loose_files(input_path, loose_glob="*.im7",
                                           suffix_cam0="_cam1.im7", suffix_cam1="_cam2.im7"):
    """Cheap: just the pair ids for the stereo loose-file layout."""
    entries, _ = _resolve_stereo_pairs(input_path, loose_glob, suffix_cam0, suffix_cam1)
    return [e[0] for e in entries]


def get_stereo_from_loose_files(input_path, index, loose_glob="*.im7",
                                 suffix_cam0="_cam1.im7", suffix_cam1="_cam2.im7",
                                 stereo_frame_order="camera_major"):
    """Load a single (pair_id, fa0, fb0, fa1, fb1) at list position
    `index` (as returned by list_pair_ids_stereo_from_loose_files)
    directly."""
    entries, combined = _resolve_stereo_pairs(input_path, loose_glob, suffix_cam0, suffix_cam1)
    import lvpyio as lv
    if combined:
        pair_id, path = entries[index]
        fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(lv.read_buffer(path), stereo_frame_order)
    else:
        pair_id, path0, path1 = entries[index]
        fa0, fb0 = frames_from_buffer(lv.read_buffer(path0))
        fa1, fb1 = frames_from_buffer(lv.read_buffer(path1))
    return pair_id, fa0, fb0, fa1, fb1
