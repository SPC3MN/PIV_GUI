"""Tests for processing/parallel_dual_planar.py -- Tier 3's cross-pair
ProcessPoolExecutor, dual-camera-planar counterpart to
test_parallel_planar.py/test_parallel_stereo.py. Real process_frames x2 ->
combine_dual_planar_pair wiring throughout (the same production code
path), not a stub -- but small/fast synthetic images, since what needs
verifying here is process-count equivalence and plumbing (ordering,
cancellation, error handling), NOT PIV/placement numerics (already
covered by test_stereo_pipeline.py/test_davis_set_dual_planar.py). This
spawns real OS processes per test, so keeping each pair's own work tiny
matters for the test suite's overall runtime.
"""

import numpy as np
import pytest

from piv_suite.config.legacy import to_cpu_settings
from piv_suite.config.schema import DualPlanarSettings, PassSettings, ProjectConfig
from piv_suite.engines.registry import get_engine_factory
from piv_suite.processing import pipeline
from piv_suite.processing.preprocess import apply_preprocess_pair
from piv_suite.processing.parallel_dual_planar import (
    _worker_init,
    process_one_pair_dual_planar_worker,
    run_dual_planar_batch_parallel,
)

RAW_SHAPE = (64, 64)


def _dual_planar_settings():
    """No-overlap placement (cam1 to the left of cam0), identity raw-to-
    canvas scale, simple mm scale -- same style as
    test_davis_set_dual_planar.py's own synthetic fixture. Placement
    accuracy isn't what these tests check (see module docstring); this
    only needs to be a real, valid DualPlanarSettings that
    combine_dual_planar_pair can run against without raising."""
    ny, nx = RAW_SHAPE
    dp = DualPlanarSettings(
        enabled=True,
        canvas_width=2 * nx, canvas_height=ny,
        scale_x_mm_per_px=2.0, scale_x_offset_mm=0.0,
        scale_y_mm_per_px=-2.0, scale_y_offset_mm=0.0,
    )
    dp.cam0.region_x, dp.cam0.region_y = nx, 0.0
    dp.cam0.region_width, dp.cam0.region_height = nx, ny
    dp.cam0.raw_width, dp.cam0.raw_height = nx, ny
    dp.cam1.region_x, dp.cam1.region_y = 0.0, 0.0
    dp.cam1.region_width, dp.cam1.region_height = nx, ny
    dp.cam1.raw_width, dp.cam1.raw_height = nx, ny
    return dp


def _fast_dual_planar_cfg():
    """A ProjectConfig sized to run in well under a second per pair --
    single small pass, minimal post-processing, no plot/csv output."""
    cfg = ProjectConfig()
    cfg.project.backend = "cpu"
    cfg.project.mode = "planar"
    cfg.project.dual_camera = True
    cfg.correlation.passes = [PassSettings(16, 0.5)]
    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.save_summary_csv = False
    cfg.output.verbose = False
    cfg.postprocess.range_filter.enabled = False
    cfg.postprocess.global_outlier_std = None
    cfg.postprocess.remove_small_groups_threshold = None
    cfg.dual_planar = _dual_planar_settings()
    return cfg


def _make_raw_pair(rng, shift):
    base = (rng.rand(*RAW_SHAPE) * 200 + 20).astype(np.float32)
    shifted = np.roll(base, shift=shift, axis=(0, 1))
    return base, shifted


def _make_dual_planar_pair(rng, shift0=(1, 2), shift1=(2, 1)):
    fa0, fb0 = _make_raw_pair(rng, shift0)
    fa1, fb1 = _make_raw_pair(rng, shift1)
    return fa0, fb0, fa1, fb1


def _serial_reference(cfg, fa0, fb0, fa1, fb1):
    """What the ORIGINAL (pre-Tier-3) serial per-pair code (cli.main.
    handle_pair_dual_planar / pipeline_worker._process_set_dual_planar)
    computes -- build an engine PER CAMERA (two separate instances,
    matching both serial loops' actual structure) + process_frames x2 +
    combine_dual_planar_pair, no parallelism involved at all. The
    reference every worker-process result must match exactly."""
    fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
    fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)

    cpu_settings = to_cpu_settings(cfg.correlation, cfg.validation)
    factory = get_engine_factory("cpu")
    post = cfg.postprocess.for_pipeline()

    # engine.coords is read AFTER process_frames, not the factory's own
    # directly-returned x/y -- combine_dual_planar_pair needs the RAW
    # (row-down, unflipped) grid, not the display-flipped one the factory
    # itself returns (see pipeline.combine_dual_planar_pair's docstring
    # and cli.main._run_dual_planar_camera, which this mirrors exactly).
    # Using the factory's own x/y directly here was a real bug in an
    # earlier version of THIS test -- it produced a Y grid that silently
    # differed from what both the real worker and every real call site
    # actually use, which fed a subtly wrong grid into griddata and
    # showed up as small-but-real velocity differences after resampling,
    # despite both cameras' raw u/v correlation output matching exactly.
    engine0, _x0_flipped, _y0_flipped = factory(fa0.shape, {"cpu_settings": cpu_settings})
    u0, v0, valid0, _, r0 = pipeline.process_frames(engine0, fa0, fb0, post)
    x0, y0 = engine0.coords
    engine1, _x1_flipped, _y1_flipped = factory(fa1.shape, {"cpu_settings": cpu_settings})
    u1, v1, valid1, _, r1 = pipeline.process_frames(engine1, fa1, fb1, post)
    x1, y1 = engine1.coords

    X, Y, U, V, valid = pipeline.combine_dual_planar_pair(
        (u0, v0, x0, y0, valid0), (u1, v1, x1, y1, valid1),
        cfg.dual_planar, cfg.calibration.frame_dt_s)
    n_range = r0["range_residual"] + r1["range_residual"]
    n_std = r0["std_dev"] + r1["std_dev"]
    return X, Y, U, V, valid, n_range, n_std


def test_process_one_pair_dual_planar_worker_matches_serial_reference(tmp_path):
    cfg = _fast_dual_planar_cfg()
    _worker_init()  # normally the ProcessPoolExecutor initializer -- called directly here
    # since this test invokes the worker function in-process, not via a pool.
    rng = np.random.RandomState(1)
    fa0, fb0, fa1, fb1 = _make_dual_planar_pair(rng)

    result = process_one_pair_dual_planar_worker(
        0, "pairA", fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy(), cfg, str(tmp_path))
    assert result["pair_id"] == "pairA"

    saved = np.load(tmp_path / "pairA_velocity.npz")
    X_ref, Y_ref, U_ref, V_ref, valid_ref, n_range_ref, n_std_ref = _serial_reference(
        cfg, fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy())

    np.testing.assert_array_equal(saved["x"], X_ref)
    np.testing.assert_array_equal(saved["y"], Y_ref)
    np.testing.assert_array_equal(saved["u"], U_ref)
    np.testing.assert_array_equal(saved["v"], V_ref)
    assert np.array_equal(saved["valid"], valid_ref)
    assert result["n_valid"] == int(valid_ref.sum())
    assert result["n_total"] == int(valid_ref.size)
    assert result["n_rejected_range_residual"] == n_range_ref
    assert result["n_rejected_std_dev"] == n_std_ref


def test_run_dual_planar_batch_parallel_matches_serial_for_every_pair(tmp_path):
    cfg = _fast_dual_planar_cfg()
    rng = np.random.RandomState(2)
    pairs = [(f"pair{i}", *_make_dual_planar_pair(rng, shift0=(i % 3 + 1, i % 2 + 1),
                                                   shift1=(i % 2 + 1, i % 3 + 1)))
             for i in range(5)]

    out_dir = tmp_path / "parallel"
    out_dir.mkdir()
    results, cancelled = run_dual_planar_batch_parallel(
        ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), n_workers=3)

    assert not cancelled
    assert [r["pair_id"] for r in results] == [pid for pid, *_ in pairs]  # submission order preserved

    for pair_id, fa0, fb0, fa1, fb1 in pairs:
        saved = np.load(out_dir / f"{pair_id}_velocity.npz")
        _, _, U_ref, V_ref, valid_ref, _, _ = _serial_reference(cfg, fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy())
        np.testing.assert_array_equal(saved["u"], U_ref)
        np.testing.assert_array_equal(saved["v"], V_ref)
        assert np.array_equal(saved["valid"], valid_ref)


def test_run_dual_planar_batch_parallel_output_independent_of_worker_count(tmp_path):
    cfg = _fast_dual_planar_cfg()
    rng = np.random.RandomState(3)
    pairs = [(f"pair{i}", *_make_dual_planar_pair(rng, shift0=(i + 1, i + 1), shift1=(i + 2, i + 1)))
             for i in range(4)]

    dirs = {}
    for n_workers in (1, 2, 4):
        # n_workers=1 here exercises run_dual_planar_batch_parallel's OWN
        # pool (max_workers=1), not the "skip the pool entirely" branch
        # the real call sites use for n_workers<=1 -- still must agree
        # byte-for-byte with n_workers>1, since a size-1 pool is exact too.
        out_dir = tmp_path / f"n{n_workers}"
        out_dir.mkdir()
        results, cancelled = run_dual_planar_batch_parallel(
            ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
            cfg, str(out_dir), n_workers=n_workers)
        assert not cancelled
        dirs[n_workers] = out_dir

    for pair_id, *_ in pairs:
        ref = np.load(dirs[1] / f"{pair_id}_velocity.npz")
        for n_workers in (2, 4):
            other = np.load(dirs[n_workers] / f"{pair_id}_velocity.npz")
            np.testing.assert_array_equal(ref["u"], other["u"])
            np.testing.assert_array_equal(ref["v"], other["v"])
            assert np.array_equal(ref["valid"], other["valid"])


def test_run_dual_planar_batch_parallel_stops_submitting_after_cancel(tmp_path):
    cfg = _fast_dual_planar_cfg()
    rng = np.random.RandomState(4)
    pairs = [(f"pair{i}", *_make_dual_planar_pair(rng)) for i in range(6)]

    submitted = []

    def cancel_after_two():
        return len(submitted) >= 2

    def on_started(pair_id):
        submitted.append(pair_id)

    out_dir = tmp_path / "cancelled"
    out_dir.mkdir()
    results, cancelled = run_dual_planar_batch_parallel(
        ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), n_workers=2,
        on_pair_started=on_started, cancel_check=cancel_after_two)

    assert cancelled
    assert len(submitted) == 2
    # in-flight pairs (already submitted before cancellation) still finish
    # and are returned -- never MORE than what was submitted, never fewer
    # than what had time to complete.
    assert len(results) <= len(submitted)


def test_run_dual_planar_batch_parallel_one_pair_erroring_does_not_lose_others(tmp_path):
    cfg = _fast_dual_planar_cfg()
    rng = np.random.RandomState(5)
    good0 = _make_dual_planar_pair(rng)
    good1 = _make_dual_planar_pair(rng)
    # A raw array shape smaller than the correlation window size reliably
    # raises inside process_frames -- per-pair and deterministic, same
    # trick test_parallel_planar.py's own error test uses.
    bad_a0 = np.zeros((4, 4), dtype=np.float32)
    bad = (bad_a0, good0[1], good0[2], good0[3])

    pairs = [("good1", *good0), ("bad", *bad), ("good2", *good1)]
    errors = []

    out_dir = tmp_path / "with_error"
    out_dir.mkdir()
    results, cancelled = run_dual_planar_batch_parallel(
        ((pid, a0, b0, a1, b1) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), n_workers=2,
        on_pair_error=lambda pair_id, exc: errors.append(pair_id))

    assert not cancelled
    assert errors == ["bad"]
    assert {r["pair_id"] for r in results} == {"good1", "good2"}
