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


def iter_pairs_from_set(set_path, multiset_index=0):
    """Yield (pair_id, frame_a, frame_b) from a DaVis image set (planar)."""
    import lvpyio as lv
    if lv.is_multiset(set_path):
        print(f"[info] '{set_path}' is a multi-set (e.g. multiple cameras) "
              f"-- using sub-set index {multiset_index}")
        sets = lv.read_set(set_path)
        dataset = sets[multiset_index]
        owns_dataset = False
    else:
        dataset = lv.read_set(set_path)
        owns_dataset = True

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
    import lvpyio as lv
    if lv.is_multiset(set_path):
        print(f"[info] '{set_path}' is a multi-set -- using sub-set "
              f"index {multiset_index}")
        sets = lv.read_set(set_path)
        dataset = sets[multiset_index]
        owns_dataset = False
    else:
        dataset = lv.read_set(set_path)
        owns_dataset = True

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
