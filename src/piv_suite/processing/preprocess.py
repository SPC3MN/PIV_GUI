"""Image preprocessing: applied to raw camera frames BEFORE any
correlation (and, for stereo, before dewarping -- each camera's own raw
pixel grid, not the calibrated/mapped one).

`min_max_filter` is a LaVision-style sliding min/max intensity filter --
removes local background level and normalizes local contrast. See its
own docstring for the exact formula, and for the real-data measurements
that replaced an earlier, unclipped version of it.
"""

import numpy as np

# Fraction of a typical PARTICLE's local dynamic range below which
# min_max_filter refuses to normalize -- the noise floor whose absence made
# the previous formulation amplify background 24x harder than signal (see
# that function's docstring).
#
# 0.10 measured, not assumed. Swept against DaVis's own vectors on a real
# planar recording (PIV_Samples/Planar, L=4, everything else fixed):
#
#     clip   density   corr(U)  corr(V)  mean|diff|
#     0.02    99.24%    0.984    0.979    13.68 mm/s
#     0.05    99.31%    0.988    0.985    11.76
#     0.10    99.36%    0.992    0.990     9.53
#     0.25    99.35%    0.990    0.987    10.72
#     0.50    99.34%    0.985    0.981    12.67
#
# A real interior optimum with a shallow basin either side: too low and the
# floor stops holding back noise in particle-free patches, too high and it
# starts suppressing genuine mid-contrast particles. Deliberately a constant
# rather than a setting -- the curve is flat enough within a factor of ~2.5
# that there is nothing here for a user to usefully tune, and a knob would
# mostly offer new ways to re-break it.
MIN_MAX_CLIP_FRACTION = 0.10


def min_max_filter(image, length, clip_fraction=None):
    """LaVision-style min/max intensity filter -- removes local background
    intensity and normalizes local particle contrast, via a sliding
    min/max window of side `length` (L, in pixels):

      1. MinL     = sliding minimum over an LxL window
      2. MaxL     = sliding maximum over an LxL window
      3. RangeL   = MaxL - MinL                  (local dynamic range)
      4. floor    = clip_fraction * p99.5(RangeL)          (noise floor)
      5. filtered = (image - MinL) / max(RangeL, floor)

    Vectorized via scipy.ndimage.minimum_filter/maximum_filter -- not a
    per-pixel Python loop.

    STEP 4 IS THE WHOLE POINT, and its absence was a real bug. This filter
    previously ended `filtered = (image - MinL) * Max10L / MaxL`, using the
    maximum over a 10x-larger window as the numerator instead of a clipped
    constant. Max10L >= MaxL by construction, so that gain is >= 1
    everywhere and unbounded wherever MaxL is small -- i.e. exactly in the
    particle-free regions where the only thing to amplify is sensor noise.
    Measured on a real DaVis recording (4096x3008, L=4): median gain 15.2x,
    68% of pixels amplified more than 10x, up to 563x, and -- decisively --
    a mean gain of 24.4x in the darkest half of the frame against 2.28x in
    the brightest 1%. It amplified background an order of magnitude harder
    than signal, cutting particle-to-background contrast (p99.9/median)
    from the raw frame's 8.0 to... 20.7, where simply subtracting MinL and
    stopping gives 40.1.

    The cost was not subtle. Against DaVis's own vectors on the same
    frames, with everything else held fixed:

        planar   density  corr(U)  corr(V)  mean|diff|
        old       94.3%    0.783    0.736    33.7 mm/s
        off       98.2%    0.962    0.951    15.4 mm/s
        this      99.4%    0.992    0.990     9.5 mm/s
        (DaVis    98.1%)

    i.e. the filter DaVis itself reports having applied to these images
    (JobHistory.xml: useMinMaxFilter=true, minMaxFilterLength=4) was, as
    implemented here, worse than not preprocessing at all -- and is now
    the single largest contributor to agreement with DaVis.

    The floor is a fraction of p99.5(RangeL) rather than of the global
    intensity range: p99.5 of the local dynamic range is what a real
    PARTICLE's contrast looks like in this frame, so the threshold means
    "don't normalize a patch whose contrast is under `clip_fraction` of a
    particle's" -- which is the physically meaningful statement, and is
    robust to a few saturated pixels in a way a plain max is not.

    clip_fraction=None uses MIN_MAX_CLIP_FRACTION (see its comment).
    """
    image = np.asarray(image, dtype=np.float64)
    length = int(length)
    if length < 1:
        raise ValueError(f"min_max_filter: length must be >= 1, got {length}")
    if clip_fraction is None:
        clip_fraction = MIN_MAX_CLIP_FRACTION

    from scipy.ndimage import minimum_filter, maximum_filter

    min_l = minimum_filter(image, size=length, mode="nearest")
    max_l = maximum_filter(image, size=length, mode="nearest")
    range_l = max_l - min_l

    # A frame with no contrast anywhere (all-constant) has p99.5 == 0, which
    # would make the floor 0 and the division undefined again -- fall back to
    # 1.0 there, which returns the (identically zero) numerator unchanged
    # rather than inf/NaN.
    floor = clip_fraction * float(np.percentile(range_l, 99.5))
    if not floor > 0:
        floor = 1.0
    return (image - min_l) / np.maximum(range_l, floor)


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
