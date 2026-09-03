import numpy as np

from piv_suite.processing.preprocess import (MIN_MAX_CLIP_FRACTION, apply_preprocess_pair,
                                             min_max_filter)


def test_min_max_filter_matches_manual_formula():
    rng = np.random.RandomState(0)
    image = rng.rand(20, 20) * 100 + 10  # background offset, so MinL isn't trivially 0
    length = 3

    from scipy.ndimage import minimum_filter, maximum_filter
    min_l = minimum_filter(image, size=length, mode="nearest")
    max_l = maximum_filter(image, size=length, mode="nearest")
    range_l = max_l - min_l
    floor = MIN_MAX_CLIP_FRACTION * np.percentile(range_l, 99.5)
    expected = (image - min_l) / np.maximum(range_l, floor)

    result = min_max_filter(image, length)
    assert np.allclose(result, expected)


def _synthetic_particle_field(seed):
    """Sparse bright particles on a low-amplitude noise floor -- the
    minimal stand-in for a PIV frame these two tests need."""
    rng = np.random.RandomState(seed)
    image = 40.0 + rng.rand(200, 200) * 4.0          # background: level 40, spread ~4
    particles = np.zeros_like(image, dtype=bool)
    ys, xs = rng.randint(5, 195, 60), rng.randint(5, 195, 60)
    for y, x in zip(ys, xs):
        image[y - 1:y + 2, x - 1:x + 2] += 400.0
        particles[y - 1:y + 2, x - 1:x + 2] = True
    return image, particles


def _particle_snr(image, particles):
    """How far the particles stand clear of the background, in units of
    the background's own spread. This is what a PIV correlation actually
    consumes, and what a preprocessing filter has to preserve or improve."""
    bg = image[~particles]
    return (image[particles].mean() - bg.mean()) / bg.std()


def test_min_max_filter_beats_the_unclipped_formulation_it_replaced():
    """The property whose absence was the bug, pinned against the exact
    expression this filter used to end with: `(I - MinL) * Max10L / MaxL`,
    whose gain is unbounded wherever the local dynamic range is small -- on
    a real 4096x3008 frame it amplified the darkest half of the image 24.4x
    on average against 2.28x for the brightest 1%, which is the wrong way
    round for a filter whose job is to make particles stand out from
    background.

    Deliberately a RELATIVE claim (new beats old on the same frame by the
    same measure), not an absolute "SNR must not fall" one: this synthetic
    field is far cleaner than a real particle image (its raw
    particle-to-background contrast is ~350, against 8.0 measured on a real
    frame), so an absolute floor here would be a statement about the
    fixture rather than about the filter. The absolute evidence is the real
    DaVis comparison quoted in min_max_filter's own docstring."""
    from scipy.ndimage import maximum_filter, minimum_filter

    image, particles = _synthetic_particle_field(3)
    min_l = minimum_filter(image, size=4, mode="nearest")
    tmp = image - min_l
    max_l = maximum_filter(tmp, size=4, mode="nearest")
    max_10l = maximum_filter(tmp, size=40, mode="nearest")
    old = np.where(max_l > 0, tmp * max_10l / max_l, 0.0)

    assert _particle_snr(min_max_filter(image, 4), particles) > _particle_snr(old, particles)


def test_min_max_filter_improves_particle_to_background_contrast():
    """The filter has to earn its place: a real particle image's
    p99.9/median contrast ratio should go UP, not down. The previous
    formulation took a real frame from 8.0 to 20.7 while merely
    subtracting the sliding minimum reached 40.1 -- i.e. its normalization
    step was destroying the contrast the subtraction had won."""
    rng = np.random.RandomState(11)
    image = 40.0 + rng.rand(200, 200) * 4.0
    ys, xs = rng.randint(5, 195, 60), rng.randint(5, 195, 60)
    for y, x in zip(ys, xs):
        image[y - 1:y + 2, x - 1:x + 2] += 400.0

    def contrast(a):
        return np.percentile(a, 99.9) / max(float(np.median(a)), 1e-12)

    assert contrast(min_max_filter(image, 4)) > contrast(image)


def test_min_max_filter_survives_a_constant_frame():
    """A frame with no contrast anywhere makes p99.5(RangeL) zero, which
    would put a 0 in the denominator."""
    out = min_max_filter(np.full((8, 8), 37.0), 3)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_min_max_filter_shape_and_dtype_preserved():
    image = (np.random.rand(15, 12) * 255).astype(np.uint8)
    result = min_max_filter(image, 4)
    assert result.shape == image.shape
    assert result.dtype == np.float64


def test_min_max_filter_removes_uniform_background():
    # A flat background plus one bright spot -- after the filter, the flat
    # region should be pulled toward 0 (background removed), which a
    # residual "add a constant" transform would not do.
    image = np.full((21, 21), 50.0)
    image[10, 10] = 250.0
    result = min_max_filter(image, 3)
    # Far from the spot, MinL/MaxL both see only background -> tmp is 0
    assert np.isclose(result[0, 0], 0.0)


def test_min_max_filter_rejects_invalid_length():
    import pytest
    with pytest.raises(ValueError):
        min_max_filter(np.zeros((5, 5)), 0)


def test_apply_preprocess_pair_no_op_when_disabled():
    class _Settings:
        min_max_filter_enabled = False
        min_max_filter_length = 5

    frame_a = np.random.rand(10, 10)
    frame_b = np.random.rand(10, 10)
    out_a, out_b = apply_preprocess_pair(frame_a, frame_b, _Settings())
    assert out_a is frame_a
    assert out_b is frame_b


def test_apply_preprocess_pair_applies_filter_when_enabled():
    class _Settings:
        min_max_filter_enabled = True
        min_max_filter_length = 3

    frame_a = np.random.rand(10, 10) * 100
    frame_b = np.random.rand(10, 10) * 100
    out_a, out_b = apply_preprocess_pair(frame_a, frame_b, _Settings())
    assert np.allclose(out_a, min_max_filter(frame_a, 3))
    assert np.allclose(out_b, min_max_filter(frame_b, 3))
