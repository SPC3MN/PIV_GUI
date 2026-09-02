import numpy as np
import pytest

from piv_suite.processing.postprocess import (
    apply_calibration, global_outlier_mask, normalized_median_residual,
    range_filter, remove_small_groups, replace_invalid_vectors,
    smooth_vector_field,
)


def _reference_uod(u, v, window_size=3, eps=0.1):
    """Straightforward per-cell Westerweel & Scarano reference -- explicit
    Python loops, no strides/chunking. Deliberately the slowest, most
    obvious expression of the formula, so it can serve as ground truth
    for the vectorized/chunked implementation under test."""
    ny, nx = u.shape
    pad = window_size // 2
    out = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            nb_u, nb_v = [], []
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    if di == 0 and dj == 0:
                        continue  # W&S excludes the vector being judged
                    ii, jj = i + di, j + dj
                    if 0 <= ii < ny and 0 <= jj < nx:
                        nb_u.append(u[ii, jj])
                        nb_v.append(v[ii, jj])
            nb_u, nb_v = np.array(nb_u), np.array(nb_v)
            if np.all(np.isnan(nb_u)):
                continue
            med_u, med_v = np.nanmedian(nb_u), np.nanmedian(nb_v)
            mad_u = np.nanmedian(np.abs(nb_u - med_u))
            mad_v = np.nanmedian(np.abs(nb_v - med_v))
            r_u = abs(u[i, j] - med_u) / (mad_u + eps)
            r_v = abs(v[i, j] - med_v) / (mad_v + eps)
            out[i, j] = np.sqrt(r_u ** 2 + r_v ** 2)
    return out


@pytest.mark.parametrize("window_size", [3, 5])
def test_normalized_median_residual_matches_explicit_reference(window_size):
    rng = np.random.default_rng(11)
    u = rng.normal(size=(12, 14))
    v = rng.normal(size=(12, 14))
    got = normalized_median_residual(u, v, window_size=window_size)
    want = _reference_uod(u, v, window_size=window_size)
    both = np.isfinite(got) & np.isfinite(want)
    assert both.sum() > 100
    assert np.allclose(got[both], want[both])


def test_normalized_median_residual_chunking_is_exact():
    # Chunk boundaries must never change a result -- each cell's
    # statistics depend only on its own neighbourhood.
    rng = np.random.default_rng(12)
    u, v = rng.normal(size=(40, 23)), rng.normal(size=(40, 23))
    whole = normalized_median_residual(u, v, _max_bytes=10**9)
    chunked = normalized_median_residual(u, v, _max_bytes=1)  # forces 1 row per chunk
    assert np.array_equal(whole, chunked, equal_nan=True)


def test_normalized_median_residual_excludes_self_from_own_neighbourhood():
    # If the centre were included, a lone spike would drag the median it
    # is judged against toward itself and score far lower.
    u = np.ones((7, 7))
    v = np.zeros((7, 7))
    u[3, 3] = 50.0
    r = normalized_median_residual(u, v)
    assert r[3, 3] > 100  # judged against neighbours only -> enormous
    assert np.nanmax(np.delete(r.ravel(), 3 * 7 + 3)) < r[3, 3]


def test_normalized_median_residual_ignores_nan_neighbours():
    rng = np.random.default_rng(13)
    u, v = rng.normal(size=(9, 9)), rng.normal(size=(9, 9))
    u_holed, v_holed = u.copy(), v.copy()
    u_holed[2, 2] = np.nan
    v_holed[2, 2] = np.nan
    r = normalized_median_residual(u_holed, v_holed)
    # the hole itself has no score, but its neighbours still do (they are
    # not penalized for sitting next to a hole)
    assert np.isnan(r[2, 2])
    assert np.isfinite(r[2, 3]) and np.isfinite(r[3, 2])


def test_normalized_median_residual_is_scale_free_apart_from_eps():
    # The normalization is what lets ONE threshold work across regions of
    # different local dynamics -- the property the old absolute-distance
    # implementation lacked. With eps driven to ~0 the statistic is
    # invariant to a global rescaling of the field.
    rng = np.random.default_rng(14)
    u, v = rng.normal(size=(10, 10)), rng.normal(size=(10, 10))
    r1 = normalized_median_residual(u, v, eps=1e-12)
    r2 = normalized_median_residual(u * 1000.0, v * 1000.0, eps=1e-12)
    both = np.isfinite(r1) & np.isfinite(r2)
    assert np.allclose(r1[both], r2[both], rtol=1e-6)


def test_global_outlier_mask_none_disables():
    u = np.array([1.0, 2.0, 100.0])
    v = np.zeros(3)
    mask = global_outlier_mask(u, v, None)
    assert not mask.any()


def test_global_outlier_mask_flags_known_spike():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, size=1000)
    v = rng.normal(0, 1, size=1000)
    u[500] = 1000.0  # obvious spike
    mask = global_outlier_mask(u, v, n_std=4.0)
    assert mask[500]
    # most of the well-behaved normal samples should NOT be flagged
    assert mask.sum() < 10


def test_range_filter_residual_flags_spatial_spike():
    ny, nx = 9, 9
    u = np.ones((ny, nx)) * 2.0
    v = np.zeros((ny, nx))
    u[4, 4] = 50.0  # a single vector wildly different from its neighbors
    invalid = range_filter(u, v, residual_max=5.0, window_size=3)
    assert invalid[4, 4]
    assert invalid.sum() == 1


def test_range_filter_none_disables():
    u = np.ones((5, 5))
    v = np.zeros((5, 5))
    invalid = range_filter(u, v, residual_max=None)
    assert not invalid.any()


def test_range_filter_residual_requires_2d():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        range_filter(u, v, residual_max=1.0)


def test_range_filter_no_bounds_rejects_nothing():
    u = np.array([1.0, 1000.0])
    v = np.array([1.0, 1000.0])
    invalid = range_filter(u, v)
    assert not invalid.any()


def test_remove_small_groups_drops_island_keeps_cluster():
    valid = np.zeros((6, 6), dtype=bool)
    valid[0, 0] = True
    valid[0, 1] = True  # isolated 2-vector island
    valid[3:5, 3:5] = True  # 4-vector cluster
    valid[5, 4] = True  # touches cluster -> 5-vector cluster total
    out = remove_small_groups(valid, min_group_size=5)
    assert not out[0, 0] and not out[0, 1]
    assert out[3:5, 3:5].all()
    assert out[5, 4]


def test_remove_small_groups_none_or_le_one_disables():
    valid = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(remove_small_groups(valid, None), valid)
    np.testing.assert_array_equal(remove_small_groups(valid, 1), valid)


def test_replace_invalid_vectors_interpolates():
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    u = x.copy()
    v = y.copy()
    valid = np.ones_like(u, dtype=bool)
    valid[2, 2] = False
    u_out, v_out = replace_invalid_vectors(x.ravel(), y.ravel(), u.ravel(), v.ravel(), valid.ravel())
    # linear interpolation of a linear field should recover the true value
    assert np.isclose(u_out.reshape(5, 5)[2, 2], 2.0, atol=0.5)


def test_replace_invalid_vectors_no_crash_when_all_rejected():
    # Regression test: an overly strict filter combo (e.g. range_filter +
    # global_outlier_std) can reject every vector in a pair -- griddata
    # crashes with "No points given" if there's nothing left to interpolate
    # from; this must degrade gracefully (stay NaN) instead of crashing the
    # whole batch run.
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    u = np.full((5, 5), np.nan)
    v = np.full((5, 5), np.nan)
    valid = np.zeros((5, 5), dtype=bool)
    u_out, v_out = replace_invalid_vectors(x.ravel(), y.ravel(), u.ravel(), v.ravel(), valid.ravel())
    assert np.isnan(u_out).all()
    assert np.isnan(v_out).all()


def test_replace_invalid_vectors_noop_when_all_valid():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    valid = np.ones(3, dtype=bool)
    u_out, v_out = replace_invalid_vectors(x, y, u, v, valid)
    np.testing.assert_array_equal(u_out, u)
    np.testing.assert_array_equal(v_out, v)


def test_smooth_vector_field_preserves_shape_and_handles_nan():
    u = np.ones((10, 10))
    v = np.zeros((10, 10))
    u[5, 5] = np.nan
    u_s, v_s = smooth_vector_field(u, v, sigma=1.0)
    assert u_s.shape == u.shape
    assert not np.isnan(u_s).any()


def test_apply_calibration_noop_without_both_params():
    u = np.array([1.0])
    v = np.array([1.0])
    u_out, v_out = apply_calibration(u, v, pixel_pitch_mm=None, frame_dt_s=0.001)
    np.testing.assert_array_equal(u_out, u)


def test_apply_calibration_scales_correctly():
    u = np.array([10.0])  # px/frame
    v = np.array([0.0])
    # 0.01 mm/px, 0.002 s/frame -> scale = (0.01/1000) / 0.002 = 0.005 m/s per px/frame
    u_out, v_out = apply_calibration(u, v, pixel_pitch_mm=0.01, frame_dt_s=0.002)
    expected_scale = (0.01 / 1000.0) / 0.002
    assert np.isclose(u_out[0], 10.0 * expected_scale)


# ---- DaVis's "remove and iteratively replace" median filter ----
#
# DaVis runs its median filter with a REMOVAL factor of 2 and a separate,
# looser INSERTION factor of 3 (medianUniversalOutlierRemovalFactor /
# ...InsertionFactor in a real JobHistory.xml). This app implemented only the
# removal half, which costs real density: a good vector sitting beside a
# cluster of bad ones is judged against a neighbourhood median those bad ones
# have dragged around.

def _field_with_marooned_good_vectors():
    """A uniform field containing a patch of mutually-inconsistent noise, with
    two adjacent GOOD vectors marooned inside it.

    A lone outlier does NOT create collateral damage -- a 3x3 median is robust
    to one bad neighbour in eight, which is the whole point of using it. The
    damage happens where the neighbourhood is MOSTLY bad, so a good vector is
    judged against a median the noise owns. That is the case reinsertion
    exists for, and it is what this fixture builds."""
    rng = np.random.RandomState(0)
    u = np.full((11, 11), 1.0)
    v = np.zeros((11, 11))
    u[3:8, 3:8] = rng.uniform(-60, 60, (5, 5))
    v[3:8, 3:8] = rng.uniform(-60, 60, (5, 5))
    u[5, 5] = u[5, 6] = 1.0
    v[5, 5] = v[5, 6] = 0.0
    return u, v


def test_reinsertion_recovers_a_good_vector_the_single_shot_pass_rejects():
    u, v = _field_with_marooned_good_vectors()
    single = range_filter(u, v, residual_max=2.0, window_size=3)
    both = range_filter(u, v, residual_max=2.0, window_size=3, insertion_max=3.0)

    # A genuinely good vector, rejected only because the noise around it owned
    # the median, comes back once the noise is gone.
    assert single[5, 6]
    assert not both[5, 6]
    # Reinsertion can only ever shrink the rejected set.
    assert not (both & ~single).any()
    assert both.sum() < single.sum()


def test_reinsertion_does_not_readmit_genuinely_inconsistent_vectors():
    """At DaVis's own insertion factor, a vector that still disagrees with a
    cleaned neighbourhood stays out. (A large enough threshold would readmit
    anything -- the protection is the value, not the mechanism.)"""
    u, v = _field_with_marooned_good_vectors()
    single = range_filter(u, v, residual_max=2.0, window_size=3)
    both = range_filter(u, v, residual_max=2.0, window_size=3, insertion_max=3.0)
    # Most of the noise patch stays rejected: only the 2 marooned good vectors
    # had any business coming back.
    assert both.sum() >= single.sum() - 2


def test_insertion_max_none_is_the_old_single_shot_behaviour():
    u, v = _field_with_marooned_good_vectors()
    assert np.array_equal(
        range_filter(u, v, residual_max=2.0, window_size=3),
        range_filter(u, v, residual_max=2.0, window_size=3, insertion_max=None))


def test_min_neighbours_protects_a_vector_judged_on_no_evidence():
    """An isolated vector with too few valid neighbours must not be rejected:
    there is nothing to judge it against (DaVis's medianFilterMinNoNeighbours)."""
    u = np.full((7, 7), np.nan)
    v = np.full((7, 7), np.nan)
    # One vector plus a single neighbour -- 1 neighbour, below the minimum.
    u[3, 3], v[3, 3] = 99.0, 99.0
    u[3, 4], v[3, 4] = 0.0, 0.0
    assert range_filter(u, v, residual_max=2.0, window_size=3, min_neighbours=None)[3, 3]
    assert not range_filter(u, v, residual_max=2.0, window_size=3, min_neighbours=3)[3, 3]


def test_min_neighbours_still_rejects_where_there_is_evidence():
    u, v = _field_with_marooned_good_vectors()
    rejected = range_filter(u, v, residual_max=2.0, window_size=3, min_neighbours=3)
    assert rejected[3:5, 3:5].all()


def test_contributing_separates_the_judged_cell_from_its_neighbourhood():
    """normalized_median_residual's `contributing` must exclude a cell from
    OTHERS' neighbourhoods while still judging it -- the two roles the array
    plays are distinct, and conflating them makes a rejected cell's own
    residual NaN instead of re-testable."""
    u = np.full((5, 5), 1.0)
    v = np.zeros((5, 5))
    u[2, 2] = 99.0
    contributing = np.ones((5, 5), dtype=bool)
    contributing[2, 2] = False
    resid = normalized_median_residual(u, v, window_size=3, contributing=contributing)
    # The excluded cell is still judged (finite, and large -- it disagrees).
    assert np.isfinite(resid[2, 2]) and resid[2, 2] > 10
    # And its neighbours no longer see its value at all, so they look clean.
    assert resid[1, 1] < 1.0
