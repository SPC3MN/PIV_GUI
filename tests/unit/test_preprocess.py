import numpy as np

from piv_suite.processing.preprocess import apply_preprocess_pair, min_max_filter


def test_min_max_filter_matches_manual_5_step_formula():
    rng = np.random.RandomState(0)
    image = rng.rand(20, 20) * 100 + 10  # background offset, so MinL isn't trivially 0
    length = 3

    from scipy.ndimage import minimum_filter, maximum_filter
    min_l = minimum_filter(image, size=length, mode="nearest")
    tmp = image - min_l
    max_l = maximum_filter(tmp, size=length, mode="nearest")
    max_10l = maximum_filter(tmp, size=30, mode="nearest")
    expected = np.where(max_l > 0, tmp * max_10l / max_l, 0.0)

    result = min_max_filter(image, length)
    assert np.allclose(result, expected)


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
