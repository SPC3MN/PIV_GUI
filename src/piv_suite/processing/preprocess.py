"""Image preprocessing: applied to raw camera frames BEFORE any
correlation (and, for stereo, before dewarping -- each camera's own raw
pixel grid, not the calibrated/mapped one).

`min_max_filter` is a LaVision-style sliding min/max intensity filter --
removes local background level and normalizes local contrast. See its
own docstring for the exact 5-step formula.
"""

import numpy as np


def min_max_filter(image, length):
    """LaVision-style min/max intensity filter -- removes local
    background intensity and normalizes local contrast via a sliding
    min/max window of side `length` (L, in pixels) and a 10x-larger
    window (10*L):

      1. MinL      = sliding minimum over an LxL window
      2. tmp       = image - MinL                    (background removed)
      3. MaxL      = sliding maximum over an LxL window, of tmp
      4. Max10L    = sliding maximum over a (10*L)x(10*L) window, of tmp
      5. filtered  = tmp * Max10L / MaxL

    Vectorized via scipy.ndimage.minimum_filter/maximum_filter -- not a
    per-pixel Python loop.

    Where MaxL is 0 (tmp is 0 or negative everywhere in that window --
    e.g. a fully uniform-background patch), the ratio in step 5 is
    undefined; those pixels are returned as 0.0 (matching tmp itself
    being 0 there) rather than propagating inf/NaN.
    """
    image = np.asarray(image, dtype=np.float64)
    length = int(length)
    if length < 1:
        raise ValueError(f"min_max_filter: length must be >= 1, got {length}")
    length_10 = max(1, int(round(length * 10)))

    from scipy.ndimage import minimum_filter, maximum_filter

    min_l = minimum_filter(image, size=length, mode="nearest")
    tmp = image - min_l
    max_l = maximum_filter(tmp, size=length, mode="nearest")
    max_10l = maximum_filter(tmp, size=length_10, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = np.where(max_l > 0, tmp * max_10l / max_l, 0.0)
    return filtered


def apply_preprocess_pair(frame_a, frame_b, settings):
    """Apply min_max_filter to both frames of a pair if
    settings.min_max_filter_enabled, else return them unchanged. The
    single choke point every call site (GUI batch/preview, CLI) should
    use, so preprocessing timing (e.g. before dewarp for stereo) only
    needs to be gotten right once per call site."""
    if not getattr(settings, "min_max_filter_enabled", False):
        return frame_a, frame_b
    length = settings.min_max_filter_length
    return min_max_filter(frame_a, length), min_max_filter(frame_b, length)
