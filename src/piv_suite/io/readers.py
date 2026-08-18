"""Pluggable single-frame image reader, keyed by file extension.

The original four repos only ever read LaVision .im7 files via lvpyio.
This registry generalizes loose_files.py's "frame A / frame B are separate
files" branch to accept plain labeled image pairs (TIFF, PNG, ...) too,
not just .im7 -- satisfying the "labeled image pairs" input requirement
without touching the .set/lvpyio path, which stays im7-only (DaVis project
sets are inherently a LaVision format).
"""

import os

IM7_EXTENSIONS = frozenset({".im7"})
GENERIC_IMAGE_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"})


def _read_im7_frame(path):
    import lvpyio as lv
    return lv.read_buffer(path).frames[0].images[0]


def _read_generic_frame(path):
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path)
    else:
        import imageio.v3 as iio
        arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # collapse an RGB(A) image to grayscale intensity -- PIV correlates
        # on intensity, not color
        arr = arr[..., :3].mean(axis=-1)
    return arr


def read_frame(path):
    """Read a single frame (2-D intensity array) from `path`, dispatching
    on file extension. Raises ValueError for unrecognized extensions."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IM7_EXTENSIONS:
        return _read_im7_frame(path)
    if ext in GENERIC_IMAGE_EXTENSIONS:
        return _read_generic_frame(path)
    raise ValueError(
        f"no reader registered for extension {ext!r} (path: {path}) -- "
        f"supported: {sorted(IM7_EXTENSIONS | GENERIC_IMAGE_EXTENSIONS)}"
    )


def is_supported(path):
    return os.path.splitext(path)[1].lower() in (IM7_EXTENSIONS | GENERIC_IMAGE_EXTENSIONS)
