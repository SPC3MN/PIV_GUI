"""Tests for processing/parallel_planar.py -- Tier 3's cross-pair
ProcessPoolExecutor. Pairs are independent by construction (no state
carries between them), so what needs verifying here is process-count
equivalence (same output regardless of worker count / completion order)
and plumbing (ordering, cancellation, error handling) -- NOT PIV
numerics, which engines/_openpiv_speedups.py's own tests already cover.

Small, fast synthetic images/single-pass settings throughout -- this
spawns real OS processes per test, so keeping each pair's own work tiny
matters for the test suite's overall runtime.
"""

import numpy as np
import pytest

from piv_suite.config.schema import CorrelationSettings, PassSettings, ProjectConfig
from piv_suite.engines.registry import get_engine_factory
from piv_suite.config.legacy import to_cpu_settings
from piv_suite.processing import pipeline
from piv_suite.processing.parallel_planar import (
    process_one_pair_planar_worker,
    run_planar_batch_parallel,
)


def _fast_cfg():
    """A ProjectConfig sized to run in well under a second per pair --
    single small pass, no plot/csv output."""
    cfg = ProjectConfig()
    cfg.project.backend = "cpu"
    cfg.correlation.passes = [PassSettings(16, 0.5)]
    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.save_summary_csv = False
    cfg.output.verbose = False
    cfg.postprocess.range_filter.enabled = False  # keep post-processing minimal/fast
    cfg.postprocess.global_outlier_std = None
    cfg.postprocess.remove_small_groups_threshold = None
    return cfg


def _make_pair(rng, shape=(64, 64), shift=(1, 2)):
    base = (rng.rand(*shape) * 200 + 20).astype(np.float32)
    shifted = np.roll(base, shift=shift, axis=(0, 1))
    return base, shifted


def _serial_reference(pair_id, frame_a, frame_b, cfg):
    """What the ORIGINAL (pre-Tier-3) serial per-pair code path computes
    -- engine build + pipeline.process_frames + calibration, no
    parallelism involved at all. The reference every worker-process
    result must match exactly."""
    from piv_suite.processing.preprocess import apply_preprocess_pair
    from piv_suite.processing.postprocess import apply_calibration

    frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, cfg.preprocess)
    cpu_settings = to_cpu_settings(cfg.correlation, cfg.validation)
    factory = get_engine_factory("cpu")
    engine, x, y = factory(frame_a.shape, {"cpu_settings": cpu_settings})
    post = cfg.postprocess.for_pipeline()
    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post)
    u, v = apply_calibration(u, v, cfg.calibration.pixel_pitch_mm, cfg.calibration.frame_dt_s)
    return x, y, u, v, valid


def test_process_one_pair_planar_worker_matches_serial_reference(tmp_path):
    cfg = _fast_cfg()
    rng = np.random.RandomState(1)
    frame_a, frame_b = _make_pair(rng)

    result = process_one_pair_planar_worker(0, "pairA", frame_a.copy(), frame_b.copy(), cfg, str(tmp_path))
    assert result["pair_id"] == "pairA"

    saved = np.load(tmp_path / "pairA_velocity.npz")
    x_ref, y_ref, u_ref, v_ref, valid_ref = _serial_reference("pairA", frame_a.copy(), frame_b.copy(), cfg)

    assert np.array_equal(saved["x"], x_ref)
    assert np.array_equal(saved["y"], y_ref)
    assert np.array_equal(saved["u"], u_ref)
    assert np.array_equal(saved["v"], v_ref)
    assert np.array_equal(saved["valid"], valid_ref)
    assert result["n_valid"] == int(valid_ref.sum())
    assert result["n_total"] == int(valid_ref.size)


def test_run_planar_batch_parallel_matches_serial_for_every_pair(tmp_path):
    cfg = _fast_cfg()
    rng = np.random.RandomState(2)
    pairs = [(f"pair{i}", *_make_pair(rng, shift=(i % 3 + 1, i % 2 + 1))) for i in range(5)]

    out_dir = tmp_path / "parallel"
    out_dir.mkdir()
    results, cancelled = run_planar_batch_parallel(
        ((pid, a.copy(), b.copy()) for pid, a, b in pairs), cfg, str(out_dir), n_workers=3)

    assert not cancelled
    assert [r["pair_id"] for r in results] == [pid for pid, _, _ in pairs]  # submission order preserved

    for pair_id, frame_a, frame_b in pairs:
        saved = np.load(out_dir / f"{pair_id}_velocity.npz")
        x_ref, y_ref, u_ref, v_ref, valid_ref = _serial_reference(pair_id, frame_a.copy(), frame_b.copy(), cfg)
        assert np.array_equal(saved["u"], u_ref)
        assert np.array_equal(saved["v"], v_ref)
        assert np.array_equal(saved["valid"], valid_ref)


def test_run_planar_batch_parallel_output_independent_of_worker_count(tmp_path):
    cfg = _fast_cfg()
    rng = np.random.RandomState(3)
    pairs = [(f"pair{i}", *_make_pair(rng, shift=(i + 1, i + 1))) for i in range(4)]

    dirs = {}
    for n_workers in (1, 2, 4):
        # n_workers=1 here exercises run_planar_batch_parallel's OWN pool
        # (max_workers=1), not the "skip the pool entirely" branch the
        # real call sites use for n_workers<=1 -- still must agree byte-
        # for-byte with n_workers>1, since a size-1 pool is still exact.
        out_dir = tmp_path / f"n{n_workers}"
        out_dir.mkdir()
        results, cancelled = run_planar_batch_parallel(
            ((pid, a.copy(), b.copy()) for pid, a, b in pairs), cfg, str(out_dir), n_workers=n_workers)
        assert not cancelled
        dirs[n_workers] = out_dir

    for pair_id, _, _ in pairs:
        ref = np.load(dirs[1] / f"{pair_id}_velocity.npz")
        for n_workers in (2, 4):
            other = np.load(dirs[n_workers] / f"{pair_id}_velocity.npz")
            assert np.array_equal(ref["u"], other["u"])
            assert np.array_equal(ref["v"], other["v"])
            assert np.array_equal(ref["valid"], other["valid"])


def test_run_planar_batch_parallel_never_exceeds_n_workers_in_flight(tmp_path):
    # Regression test for a real bug found via manual GUI testing on a
    # 100+-pair batch: on_pair_started used to fire for EVERY pair as
    # soon as it was submitted to the executor, not as it actually got a
    # worker -- ProcessPoolExecutor.submit() only queues a task, it
    # doesn't wait for a free slot, so the job table showed 100+ pairs
    # "running" at once (when at most n_workers ever could be), and the
    # resulting flood of near-simultaneous Qt model-insert signals was
    # enough to freeze the GUI thread ("Not Responding"). This directly
    # measures the actual in-flight count (started but not yet finished)
    # via on_pair_started/on_pair_finished, using a lock since they fire
    # from different threads -- not just inferring it from timing.
    import threading

    cfg = _fast_cfg()
    n_workers = 3
    n_pairs = 12  # well more than n_workers, so the bug (if reintroduced) would be obvious
    rng = np.random.RandomState(6)
    pairs = [(f"pair{i}", *_make_pair(rng, shift=(i % 3 + 1, i % 2 + 1))) for i in range(n_pairs)]

    lock = threading.Lock()
    concurrent_count = [0]
    max_concurrent = [0]

    def on_started(pair_id):
        with lock:
            concurrent_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])

    def on_finished(pair_id, result):
        with lock:
            concurrent_count[0] -= 1

    out_dir = tmp_path / "bounded"
    out_dir.mkdir()
    results, cancelled = run_planar_batch_parallel(
        ((pid, a.copy(), b.copy()) for pid, a, b in pairs), cfg, str(out_dir), n_workers=n_workers,
        on_pair_started=on_started, on_pair_finished=on_finished)

    assert not cancelled
    assert len(results) == n_pairs
    assert max_concurrent[0] <= n_workers, (
        f"{max_concurrent[0]} pairs were in flight at once with n_workers={n_workers} -- "
        "submission is not being throttled to the worker count"
    )


def test_run_planar_batch_parallel_stops_submitting_after_cancel(tmp_path):
    cfg = _fast_cfg()
    rng = np.random.RandomState(4)
    pairs = [(f"pair{i}", *_make_pair(rng)) for i in range(6)]

    submitted = []

    def cancel_after_two():
        return len(submitted) >= 2

    def on_started(pair_id):
        submitted.append(pair_id)

    out_dir = tmp_path / "cancelled"
    out_dir.mkdir()
    results, cancelled = run_planar_batch_parallel(
        ((pid, a.copy(), b.copy()) for pid, a, b in pairs), cfg, str(out_dir), n_workers=2,
        on_pair_started=on_started, cancel_check=cancel_after_two)

    assert cancelled
    assert len(submitted) == 2
    # in-flight pairs (already submitted before cancellation) still finish
    # and are returned -- never MORE than what was submitted, never fewer
    # than what had time to complete.
    assert len(results) <= len(submitted)


def test_run_planar_batch_parallel_one_pair_erroring_does_not_lose_others(tmp_path):
    cfg = _fast_cfg()
    rng = np.random.RandomState(5)
    good_a, good_b = _make_pair(rng)
    # A "bad" pair smaller than the 16px window -- extended_search_area_piv
    # raises ValueError("window size cannot be larger than the image"),
    # so process_one_pair_planar_worker reliably raises inside the worker.
    bad_a, bad_b = _make_pair(rng, shape=(8, 8))

    pairs = [("good1", good_a, good_b), ("bad", bad_a, bad_b), ("good2", good_a, good_b)]
    errors = []

    out_dir = tmp_path / "with_error"
    out_dir.mkdir()
    results, cancelled = run_planar_batch_parallel(
        ((pid, a.copy(), b.copy()) for pid, a, b in pairs), cfg, str(out_dir), n_workers=2,
        on_pair_error=lambda pair_id, exc: errors.append(pair_id))

    assert not cancelled
    assert errors == ["bad"]
    assert {r["pair_id"] for r in results} == {"good1", "good2"}
