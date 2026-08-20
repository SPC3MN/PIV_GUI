"""Regression tests for engines/_openpiv_speedups.py -- every fast_*
function must match its original openpiv counterpart to within float
noise (not just "look reasonable"), since these feed a pipeline whose
whole point is exact reproducibility (structure functions, integral
length scale, dissipation rate estimates all derive from the resulting
u/v fields).

Synthetic-but-realistic data only, all small enough to run in a normal
CI unit-test budget -- the real, multi-minute, real-frame-pair
validation this module's docstrings reference was done separately during
development (see git history / commit message for this file) and is not
reproduced here, since it requires the reporting user's own real .set
data outside this repo. What IS reproduced here is every case that
validation actually caught a bug in: the "distance" method's int-kernel
truncation quirk, and the replace_nans convergence-check aliasing bug
that made naive vectorization diverge from real data at max_iter=4 (the
app's own default) -- both covered below with cases specifically
engineered to exercise them, not just generic random data.
"""

import warnings

import numpy as np
import pytest

from openpiv.lib import replace_nans as orig_replace_nans
from openpiv.pyprocess import correlation_to_displacement as orig_correlation_to_displacement
from openpiv.validation import local_median_val as orig_local_median_val
from openpiv.validation import local_norm_median_val as orig_local_norm_median_val
from openpiv.validation import typical_validation as orig_typical_validation

from piv_suite.engines._openpiv_speedups import (
    apply_speedups,
    fast_correlation_to_displacement,
    fast_local_median_val,
    fast_local_norm_median_val,
    fast_replace_nans,
    loose_typical_validation,
    use_loose_validation,
    use_real_validation,
)


def _assert_close(fast, orig, atol=1e-9, rtol=1e-9):
    fast = np.asarray(fast, dtype=np.float64)
    orig = np.asarray(orig, dtype=np.float64)
    nan_mismatch = np.isnan(fast) != np.isnan(orig)
    assert not nan_mismatch.any(), f"NaN pattern mismatch at {nan_mismatch.sum()} position(s)"
    both_real = ~np.isnan(fast)
    if both_real.any():
        assert np.allclose(fast[both_real], orig[both_real], atol=atol, rtol=rtol)


# ============================================================
# fast_replace_nans
# ============================================================

def _scattered_nan_field(rng, shape=(20, 25), frac=0.25):
    a = rng.rand(*shape)
    mask = rng.rand(*shape) < frac
    a[mask] = np.nan
    return a


@pytest.mark.parametrize("method", ["localmean", "disk", "distance"])
@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("max_iter", [1, 2, 4, 8])  # 4 is the app's real default
def test_replace_nans_matches_original_scattered(method, kernel_size, max_iter):
    rng = np.random.RandomState(42)
    a = _scattered_nan_field(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig = orig_replace_nans(a.copy(), max_iter=max_iter, tol=1e-3, kernel_size=kernel_size, method=method)
    fast = fast_replace_nans(a.copy(), max_iter=max_iter, tol=1e-3, kernel_size=kernel_size, method=method)
    _assert_close(fast, orig)


def test_replace_nans_needs_multiple_iterations_to_fully_fill():
    # A dense NaN cluster that CANNOT be fully filled in one iteration
    # with kernel_size=1 -- this is what exercises the replace_nans
    # convergence-check aliasing bug (see fast_replace_nans's docstring):
    # a naive vectorization that tries to replicate `tol` literally
    # diverges from the real openpiv output specifically once more than
    # ~2 iterations actually run, which only happens when filling takes
    # multiple hops like this.
    rng = np.random.RandomState(7)
    a = rng.rand(15, 15)
    a[5:11, 5:11] = np.nan  # a 6x6 block -- too big for kernel_size=1 to fill in one pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig = orig_replace_nans(a.copy(), max_iter=4, tol=1e-3, kernel_size=1, method="localmean")
    fast = fast_replace_nans(a.copy(), max_iter=4, tol=1e-3, kernel_size=1, method="localmean")
    _assert_close(fast, orig)
    # and confirm this case actually needed real diffusion (not a trivial
    # single-iteration fill) -- otherwise this test wouldn't be exercising
    # what it claims to
    assert np.isnan(a[7, 7])  # center of the block, farthest from any real neighbor
    assert not np.isnan(orig[7, 7])  # but got filled by iteration 4


def test_replace_nans_tol_value_does_not_matter_once_multiple_iterations_are_needed():
    # Documents (and locks in) the openpiv bug this module's fast version
    # deliberately replicates rather than "fixes": once a NaN region needs
    # 2+ iterations to fully fill (this test's 6x6 block -- too big for
    # kernel_size=1 to resolve in one pass), openpiv.lib.replace_nans's
    # real stopping rule from iteration 1 onward is "all originally-NaN
    # cells filled" -- NOT a genuine tolerance comparison -- so any two
    # positive tol values behave identically. This is DELIBERATELY NOT
    # true in general: for a NaN pattern shallow enough to fully fill in
    # a SINGLE iteration (see _scattered_nan_field's 25%-random case,
    # tested elsewhere), iteration 0's check runs before the aliasing
    # that causes this bug kicks in, so the tol VALUE genuinely matters
    # there -- fast_replace_nans handles that iteration separately for
    # exactly this reason (see its docstring/source).
    a = np.random.RandomState(7).rand(15, 15)
    a[5:11, 5:11] = np.nan  # same pattern as test_replace_nans_needs_multiple_iterations_to_fully_fill
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_small_tol = orig_replace_nans(a.copy(), max_iter=4, tol=1e-3, kernel_size=1, method="localmean")
        out_huge_tol = orig_replace_nans(a.copy(), max_iter=4, tol=1e10, kernel_size=1, method="localmean")
        out_impossible_tol = orig_replace_nans(a.copy(), max_iter=4, tol=-1.0, kernel_size=1, method="localmean")
    assert np.array_equal(out_small_tol, out_huge_tol, equal_nan=True)
    # an impossible tol forces every requested iteration to actually run,
    # which (for a field needing multi-hop diffusion) gives a genuinely
    # different result -- confirming tol=1e-3 was NOT running all 4
    # iterations "for real"
    assert not np.array_equal(out_small_tol, out_impossible_tol, equal_nan=True)


def test_replace_nans_shallow_pattern_iteration_zero_tol_check_is_genuine():
    # The complementary case to the test above: a NaN pattern shallow
    # enough that a SINGLE kernel_size=1 iteration fills every cell, so
    # openpiv's aliasing bug never gets a chance to kick in (it only
    # first applies to iteration 1's check onward) -- iteration 0's tol
    # comparison is against the real (not-yet-aliased) all-zeros
    # replaced_old, so the tol VALUE itself genuinely changes the
    # outcome here, and fast_replace_nans must still match on this path.
    rng = np.random.RandomState(3)
    a = _scattered_nan_field(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_default = orig_replace_nans(a.copy(), max_iter=4, tol=1e-3, kernel_size=1, method="localmean")
        orig_huge = orig_replace_nans(a.copy(), max_iter=4, tol=1e10, kernel_size=1, method="localmean")
    # confirms this test data actually exercises the claimed scenario:
    # the huge tol changes the outcome relative to the real default here
    # (unlike the deep-cluster case above), proving iteration 0 alone
    # doesn't already decide it regardless of tol.
    assert not np.array_equal(orig_default, orig_huge, equal_nan=True)

    fast_default = fast_replace_nans(a.copy(), max_iter=4, tol=1e-3, kernel_size=1, method="localmean")
    fast_huge = fast_replace_nans(a.copy(), max_iter=4, tol=1e10, kernel_size=1, method="localmean")
    _assert_close(fast_default, orig_default)
    _assert_close(fast_huge, orig_huge)


def test_replace_nans_distance_method_kernel_truncation_bug():
    # openpiv.lib.get_dist's distance-based weights are assigned into an
    # int-dtype kernel array, silently truncating every weight below 1.0
    # to zero -- confirmed a real (if obscure) bug in openpiv 0.25.4, not
    # a misunderstanding on this project's part. fast_replace_nans must
    # reproduce the truncation, not the "intended" float weighting, or
    # results diverge (this test would have caught the very first,
    # incorrect version of fast_replace_nans written during development).
    rng = np.random.RandomState(11)
    a = rng.rand(10, 10)
    a[4, 4] = np.nan
    a[4, 5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig = orig_replace_nans(a.copy(), max_iter=2, tol=1e-3, kernel_size=1, method="distance")
    fast = fast_replace_nans(a.copy(), max_iter=2, tol=1e-3, kernel_size=1, method="distance")
    _assert_close(fast, orig)


def test_replace_nans_isolated_block_larger_than_kernel_stays_nan():
    rng = np.random.RandomState(0)
    a = rng.rand(15, 15)
    a[5:10, 5:10] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig = orig_replace_nans(a.copy(), max_iter=1, tol=1e-3, kernel_size=1, method="localmean")
    fast = fast_replace_nans(a.copy(), max_iter=1, tol=1e-3, kernel_size=1, method="localmean")
    _assert_close(fast, orig)
    assert np.isnan(orig[7, 7])  # too far from any valid neighbor in 1 iteration


def test_replace_nans_no_nan_passthrough():
    rng = np.random.RandomState(1)
    a = rng.rand(10, 10)
    orig = orig_replace_nans(a.copy(), max_iter=2, tol=1e-3, kernel_size=1, method="localmean")
    fast = fast_replace_nans(a.copy(), max_iter=2, tol=1e-3, kernel_size=1, method="localmean")
    _assert_close(fast, orig, atol=0, rtol=0)


def test_replace_nans_unknown_method_raises():
    a = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        fast_replace_nans(a, max_iter=1, tol=1e-3, kernel_size=1, method="bogus")


# ============================================================
# fast_correlation_to_displacement
# ============================================================

def _gaussian_bump_corr(rng, n_windows=200, size=16, noise=0.02):
    """Correlation maps shaped like real ones: a smooth Gaussian peak at
    a random subpixel-ish location plus a little noise, all positive
    (matching normalized_correlation=True's [0,1]-clipped range) so the
    gaussian subpixel method's happy path is exercised realistically."""
    yy, xx = np.mgrid[0:size, 0:size]
    corr = np.empty((n_windows, size, size), dtype=np.float64)
    for i in range(n_windows):
        cy = rng.uniform(size * 0.3, size * 0.7)
        cx = rng.uniform(size * 0.3, size * 0.7)
        corr[i] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
    corr += rng.rand(n_windows, size, size) * noise
    return corr


@pytest.mark.parametrize("method", ["gaussian", "parabolic", "centroid"])
def test_correlation_to_displacement_matches_original(method):
    rng = np.random.RandomState(5)
    corr = _gaussian_bump_corr(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_orig, v_orig = orig_correlation_to_displacement(corr.copy(), 20, 10, subpixel_method=method)
    u_fast, v_fast = fast_correlation_to_displacement(corr.copy(), 20, 10, subpixel_method=method)
    _assert_close(u_fast, u_orig)
    _assert_close(v_fast, v_orig)


def test_correlation_to_displacement_negative_values_trigger_parabolic_fallback():
    # gaussian falls back to parabolic per-window when any of the 5
    # sampled points is negative -- exercised deliberately, not left to
    # chance, since openpiv's own "vectorized_*" alternative was found to
    # use a DIFFERENT (`<= 0`) threshold and a forced float32 cast; this
    # is what confirms fast_correlation_to_displacement uses the ORIGINAL
    # `< 0` threshold instead.
    rng = np.random.RandomState(2)
    corr = rng.randn(100, 12, 12)  # has negative values, unlike a real normalized corr map
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_orig, v_orig = orig_correlation_to_displacement(corr.copy(), 10, 10, subpixel_method="gaussian")
    u_fast, v_fast = fast_correlation_to_displacement(corr.copy(), 10, 10, subpixel_method="gaussian")
    _assert_close(u_fast, u_orig)
    _assert_close(v_fast, v_orig)


def test_correlation_to_displacement_border_peaks_are_nan():
    corr = np.zeros((4, 8, 8))
    corr[0, 0, 3] = 1.0
    corr[1, 7, 3] = 1.0
    corr[2, 3, 0] = 1.0
    corr[3, 3, 7] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_orig, v_orig = orig_correlation_to_displacement(corr.copy(), 2, 2, subpixel_method="gaussian")
    u_fast, v_fast = fast_correlation_to_displacement(corr.copy(), 2, 2, subpixel_method="gaussian")
    assert np.isnan(u_fast).all() and np.isnan(v_fast).all()
    assert np.isnan(u_orig).all() and np.isnan(v_orig).all()


def test_correlation_to_displacement_unknown_method_raises():
    corr = np.zeros((1, 8, 8))
    with pytest.raises(ValueError):
        fast_correlation_to_displacement(corr, 1, 1, subpixel_method="bogus")


# ============================================================
# fast_local_median_val
# ============================================================

@pytest.mark.parametrize("size", [1, 2, 3])
def test_local_median_val_matches_original(size):
    rng = np.random.RandomState(9)
    u = rng.rand(20, 25) * 5
    v = rng.rand(20, 25) * 5
    # sprinkle in some outliers and some NaNs
    u[3, 3] = 100.0
    v[10, 10] = -100.0
    u[0, 0] = np.nan
    v[19, 24] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ind_orig = orig_local_median_val(u, v, u_threshold=3.0, v_threshold=3.0, size=size)
    ind_fast = fast_local_median_val(u, v, u_threshold=3.0, v_threshold=3.0, size=size)
    assert np.array_equal(ind_orig, ind_fast)


def test_local_median_val_masked_array_input():
    rng = np.random.RandomState(4)
    u = rng.rand(15, 15) * 5
    v = rng.rand(15, 15) * 5
    mask = rng.rand(15, 15) < 0.1
    u_m = np.ma.masked_array(u, mask=mask)
    v_m = np.ma.masked_array(v, mask=mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ind_orig = orig_local_median_val(u_m, v_m, u_threshold=2.0, v_threshold=2.0, size=1)
    ind_fast = fast_local_median_val(u_m, v_m, u_threshold=2.0, v_threshold=2.0, size=1)
    assert np.array_equal(ind_orig, ind_fast)


# ============================================================
# fast_local_norm_median_val
# ============================================================

@pytest.mark.parametrize("size", [1, 2, 3])
def test_local_norm_median_val_matches_original(size):
    rng = np.random.RandomState(13)
    u = rng.rand(20, 25) * 5
    v = rng.rand(20, 25) * 5
    u[3, 3] = 100.0
    v[10, 10] = -100.0
    u[0, 0] = np.nan
    v[19, 24] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ind_orig = orig_local_norm_median_val(u, v, 0.2, threshold=2.0, size=size)
    ind_fast = fast_local_norm_median_val(u, v, 0.2, threshold=2.0, size=size)
    assert np.array_equal(ind_orig, ind_fast)


def test_local_norm_median_val_masked_array_input():
    rng = np.random.RandomState(5)
    u = rng.rand(15, 15) * 5
    v = rng.rand(15, 15) * 5
    mask = rng.rand(15, 15) < 0.1
    u_m = np.ma.masked_array(u, mask=mask)
    v_m = np.ma.masked_array(v, mask=mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ind_orig = orig_local_norm_median_val(u_m, v_m, 0.2, threshold=2.0, size=1)
    ind_fast = fast_local_norm_median_val(u_m, v_m, 0.2, threshold=2.0, size=1)
    assert np.array_equal(ind_orig, ind_fast)


# ============================================================
# apply_speedups() wiring
# ============================================================

def test_apply_speedups_patches_filters_not_just_lib():
    # openpiv.filters does `from openpiv.lib import replace_nans` at ITS
    # OWN import time, binding a private copy -- patching only
    # openpiv.lib.replace_nans would silently do nothing, since
    # openpiv.filters.replace_outliers (the actual caller) already holds
    # a reference to the original function. This is the regression test
    # for that exact gotcha.
    import openpiv.filters
    import openpiv.lib

    apply_speedups()
    assert openpiv.filters.replace_nans is fast_replace_nans
    assert openpiv.lib.replace_nans is fast_replace_nans


def test_apply_speedups_patches_correlation_to_displacement_and_typical_validation():
    import openpiv.pyprocess
    import openpiv.validation

    apply_speedups()
    assert openpiv.pyprocess.correlation_to_displacement is fast_correlation_to_displacement
    assert openpiv.validation.typical_validation is loose_typical_validation


def test_apply_speedups_patches_local_median_val_and_local_norm_median_val():
    # Both are only ever reached from inside REAL typical_validation
    # (openpiv.validation.typical_validation, not the default
    # loose_typical_validation patch) -- i.e. only when a caller opts into
    # CPUPIVProcess's per_pass_validation and switches the module
    # attribute back via use_real_validation(). Patched unconditionally
    # here anyway since it's harmless when unreached (apply_speedups()
    # only runs once) and avoids a second global-patch call site.
    import openpiv.validation

    apply_speedups()
    assert openpiv.validation.local_median_val is fast_local_median_val
    assert openpiv.validation.local_norm_median_val is fast_local_norm_median_val


def test_apply_speedups_does_not_patch_fft_correlate_images():
    # Deliberately excluded -- see fast_fft_correlate_images's own
    # docstring for why (isolated checks passed, but the float32 rounding
    # noise it introduces compounds across multi-pass runs into real
    # divergence). This test exists so nobody re-adds the patch line to
    # apply_speedups() without re-reading why it was removed.
    import openpiv.pyprocess
    from openpiv.pyprocess import fft_correlate_images as original_fft_correlate_images

    apply_speedups()
    assert openpiv.pyprocess.fft_correlate_images is original_fft_correlate_images


def test_apply_speedups_is_idempotent():
    apply_speedups()
    apply_speedups()  # must not raise or double-wrap anything
    import openpiv.filters
    assert openpiv.filters.replace_nans is fast_replace_nans


# ============================================================
# use_real_validation() / use_loose_validation() toggle
# ============================================================

def test_use_real_and_loose_validation_toggle():
    import openpiv.validation

    apply_speedups()
    use_real_validation()
    assert openpiv.validation.typical_validation is orig_typical_validation
    use_loose_validation()
    assert openpiv.validation.typical_validation is loose_typical_validation
    # leave the module on the default (loose) so later tests in this
    # file/session aren't affected by this test's own toggling
    use_loose_validation()


# ============================================================
# loose_typical_validation -- flags literal NaN only, never a real outlier
# ============================================================

def test_loose_typical_validation_only_flags_nan():
    rng = np.random.RandomState(7)
    u = rng.normal(size=(10, 10))
    v = rng.normal(size=(10, 10))
    # a huge, finite outlier -- would trip global_val/global_std/median
    # under real typical_validation, but must NOT be flagged here.
    u[3, 3] = 1e6
    v[3, 3] = -1e6
    # a literal NaN cell -- the only thing that should be flagged.
    u[7, 2] = np.nan

    flags = loose_typical_validation(u, v, s2n=None, settings=None)

    expected = np.zeros((10, 10), dtype=bool)
    expected[7, 2] = True
    assert np.array_equal(flags, expected)
    assert not flags[3, 3]  # the finite outlier must NOT be flagged


def test_loose_typical_validation_flags_nan_in_either_component():
    u = np.array([[1.0, np.nan], [3.0, 4.0]])
    v = np.array([[np.nan, 2.0], [3.0, 4.0]])
    flags = loose_typical_validation(u, v, s2n=None, settings=None)
    assert np.array_equal(flags, np.array([[True, True], [False, False]]))


# ============================================================
# small end-to-end sanity: CPUPIVProcess speedups' FAITHFUL patches
# (replace_nans, correlation_to_displacement) vs unpatched, with
# validation held constant. typical_validation's replacement is a
# deliberate BEHAVIOR CHANGE (see loose_typical_validation's docstring)
# and is intentionally NOT covered by a before/after equality assertion
# here -- see the dedicated tests above for its own (different) contract.
# ============================================================

def test_cpu_engine_end_to_end_matches_with_and_without_speedups():
    # A small (fast-to-run-twice) synthetic image pair through the real
    # multi-pass CPUPIVProcess body, comparing patched vs unpatched output
    # directly -- catches any interaction between the faithful patches
    # operating together inside the real pipeline, not just each function
    # in isolation. Deliberately small (unlike the real 3008x4096/4-pass/
    # multi-minute validation this was originally checked against, which
    # isn't reproducible in a normal test budget) but still multi-pass, so
    # replace_nans/correlation_to_displacement are genuinely exercised
    # together. Validation is held constant (loose_typical_validation,
    # both before and after) since it's a deliberate behavior change, not
    # a faithful speedup -- comparing it here would conflate "did the
    # faithful patches stay faithful" with "did validation change" (it
    # did, on purpose).
    import dataclasses

    from openpiv.settings import PIVSettings
    from openpiv import windef

    rng = np.random.RandomState(123)
    size = 256
    base = (rng.rand(size, size) * 200 + 20).astype(np.float32)
    shifted = np.roll(base, shift=(1, 2), axis=(0, 1))

    settings = PIVSettings()
    valid_fields = {f.name for f in dataclasses.fields(settings)}
    overrides = {
        "windowsizes": (64, 32, 32),
        "overlap": (32, 24, 24),
        "num_iterations": 3,
    }
    for key, val in overrides.items():
        if key in valid_fields:
            setattr(settings, key, val)

    def run_once():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y, u, v, s2n = windef.first_pass(base.copy(), shifted.copy(), settings)
            grid_mask = np.zeros_like(u, dtype=bool)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
            from openpiv import filters
            flags = loose_typical_validation(u, v, s2n, settings)
            u, v = filters.replace_outliers(
                u, v, flags, method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )
            for i in range(1, settings.num_iterations):
                x, y, u, v, grid_mask, flags = windef.multipass_img_deform(
                    base.copy(), shifted.copy(), i, x, y, u, v, settings)
                u = np.ma.masked_array(u, np.ma.nomask)
                v = np.ma.masked_array(v, np.ma.nomask)
            return np.ma.filled(u, 0.0), np.ma.filled(v, 0.0)

    # Hold validation constant (loose_typical_validation via apply_speedups,
    # which also patches windef's own internal multipass calls) for BOTH
    # runs -- only the two faithful patches' presence/absence differs.
    import openpiv.lib
    import openpiv.filters
    import openpiv.pyprocess

    apply_speedups()
    u_after, v_after = run_once()

    fast_replace_nans_lib, fast_replace_nans_filters = openpiv.lib.replace_nans, openpiv.filters.replace_nans
    fast_corr_to_disp = openpiv.pyprocess.correlation_to_displacement
    try:
        openpiv.lib.replace_nans = orig_replace_nans
        openpiv.filters.replace_nans = orig_replace_nans
        openpiv.pyprocess.correlation_to_displacement = orig_correlation_to_displacement
        u_before, v_before = run_once()
    finally:
        openpiv.lib.replace_nans = fast_replace_nans_lib
        openpiv.filters.replace_nans = fast_replace_nans_filters
        openpiv.pyprocess.correlation_to_displacement = fast_corr_to_disp

    assert np.allclose(u_before, u_after, atol=1e-6, rtol=1e-6)
    assert np.allclose(v_before, v_after, atol=1e-6, rtol=1e-6)
