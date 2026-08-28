"""Tests for processing/parallel_stereo.py -- Tier 3's cross-pair
ProcessPoolExecutor, stereo counterpart to test_parallel_planar.py. Real
dewarp_image -> process_frames x2 -> combine_stereo_pair wiring throughout
(via calibration.camera_mapping.build_camera_mapping, the same production
code path), not a stub -- but small/fast synthetic cameras and images,
since what needs verifying here is process-count equivalence and plumbing
(ordering, cancellation, error handling), NOT PIV/reconstruction numerics
(already covered by test_stereo_pipeline.py/test_reconstruction.py). This
spawns real OS processes per test, so keeping each pair's own work tiny
matters for the test suite's overall runtime.
"""

import os

import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import build_camera_mapping
from piv_suite.config.legacy import to_cpu_settings
from piv_suite.config.schema import (
    CameraMappingSettings, PassSettings, ProjectConfig, StereoSettings,
)
from piv_suite.engines.registry import get_engine_factory
from piv_suite.processing import pipeline
from piv_suite.processing.preprocess import apply_preprocess_pair
from piv_suite.processing.parallel_stereo import (
    _worker_init,
    process_one_pair_stereo_worker,
    run_stereo_batch_parallel,
)

REAL_SWIRL_SET = r"J:\Final_Stereo\Swirl\On Time=0.7_Burst On Time=0.0_Burst Off Time=0.0.set"

# World grid deliberately small/fast (single 16px correlation pass, same
# sizing philosophy as test_parallel_planar._fast_cfg). MARGIN pads the
# raw canvas cam0/cam1 dewarp FROM -- see CameraMappingSettings' "1"
# coefficient below, same "-MARGIN constant term recenters the world
# grid into the padded raw canvas" trick test_stereo_pipeline.py's own
# _make_camera uses.
WORLD_SHAPE = (48, 48)
MARGIN = 8
RAW_SHAPE = (WORLD_SHAPE[0] + 2 * MARGIN, WORLD_SHAPE[1] + 2 * MARGIN)
COEF_KEYS = ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")


def _camera_mapping_settings(dx_scale):
    ny, nx = WORLD_SHAPE
    dx_coefs = {k: 0.0 for k in COEF_KEYS}
    dx_coefs["1"], dx_coefs["s"] = -MARGIN, dx_scale
    dy_coefs = {k: 0.0 for k in COEF_KEYS}
    dy_coefs["1"] = -MARGIN
    return CameraMappingSettings(x0=nx / 2, x_span=nx, y0=ny / 2, y_span=ny,
                                  dx_coefs=dx_coefs, dy_coefs=dy_coefs, name="camX")


def _fast_stereo_cfg():
    """A ProjectConfig sized to run in well under a second per pair --
    single small pass, minimal post-processing, no plot/csv output.
    frame_dt_s left None so combine_stereo_pair's output stays in native
    mm/frame units -- irrelevant to exactness (a pure per-pair scalar
    divide either way), kept simple for these plumbing-focused tests."""
    cfg = ProjectConfig()
    cfg.project.backend = "cpu"
    cfg.project.mode = "stereo"
    cfg.correlation.passes = [PassSettings(16, 0.5)]
    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.save_summary_csv = False
    cfg.output.verbose = False
    cfg.postprocess.range_filter.enabled = False
    cfg.postprocess.global_outlier_std = None
    cfg.postprocess.remove_small_groups_threshold = None
    cfg.stereo = StereoSettings(
        cam0_mapping=_camera_mapping_settings(1.0),
        cam1_mapping=_camera_mapping_settings(-1.0),
        world_shape=WORLD_SHAPE,
        world_scale_px_per_mm=1.0,
        dewarp_order=1,
        alpha1_deg=-44.765, alpha2_deg=44.765, beta1_deg=0.0, beta2_deg=0.0,
    )
    return cfg


def _angles(cfg):
    return (np.deg2rad(cfg.stereo.alpha1_deg), np.deg2rad(cfg.stereo.alpha2_deg),
            np.deg2rad(cfg.stereo.beta1_deg), np.deg2rad(cfg.stereo.beta2_deg))


def _make_raw_pair(rng, shift):
    base = (rng.rand(*RAW_SHAPE) * 200 + 20).astype(np.float32)
    shifted = np.roll(base, shift=shift, axis=(0, 1))
    return base, shifted


def _make_stereo_pair(rng, shift0=(1, 2), shift1=(2, 1)):
    fa0, fb0 = _make_raw_pair(rng, shift0)
    fa1, fb1 = _make_raw_pair(rng, shift1)
    return fa0, fb0, fa1, fb1


def _serial_reference(cfg, angles, fa0, fb0, fa1, fb1):
    """What the ORIGINAL (pre-Tier-3) serial per-pair code (cli.main.
    handle_pair_stereo / pipeline_worker._process_set_stereo) computes --
    build cam0/cam1 + dewarp + build engine PER CAMERA (two separate
    instances, matching both serial loops' actual structure) +
    pipeline.process_stereo_pair (both engines -> combine -> validate the
    combined field once -- see that function's own docstring for why this
    replaced the earlier process_frames-x2-then-intersect approach), no
    parallelism involved at all. The reference every worker-process result
    must match exactly.

    This fixture's own postprocess settings (_fast_stereo_cfg) disable
    every validation mechanism (range_filter, global_outlier_std,
    remove_small_groups) and never sets raw_width/raw_height on the
    synthetic cameras (so raw_domain_valid is a no-op, all-True -- see
    CameraMapping.raw_domain_valid's own "unknown, no masking" contract)
    -- so old-vs-new architecture makes no numeric difference here either
    way; this reference is kept structurally faithful to the real
    production call sites anyway, not just simplified to whatever
    matches."""
    cam0 = build_camera_mapping(cfg.stereo.cam0_mapping, cfg.stereo.cam0_mapping_plane2,
                                 cfg.stereo.sheet_z_mm)
    cam1 = build_camera_mapping(cfg.stereo.cam1_mapping, cfg.stereo.cam1_mapping_plane2,
                                 cfg.stereo.sheet_z_mm)

    fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
    fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)
    dw_a0 = cam0.dewarp_image(fa0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_b0 = cam0.dewarp_image(fb0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_a1 = cam1.dewarp_image(fa1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_b1 = cam1.dewarp_image(fb1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)

    cpu_settings = to_cpu_settings(cfg.correlation, cfg.validation)
    factory = get_engine_factory("cpu")
    post = cfg.postprocess.for_pipeline()

    engine0, x, y = factory(dw_a0.shape, {"cpu_settings": cpu_settings})
    engine1, _, _ = factory(dw_a1.shape, {"cpu_settings": cpu_settings})
    y_row_down = cfg.stereo.world_shape[0] - y
    fov_valid = cam0.raw_domain_valid(x, y_row_down) & cam1.raw_domain_valid(x, y_row_down)

    U, V, W, valid, _, r = pipeline.process_stereo_pair(
        engine0, engine1, dw_a0, dw_b0, dw_a1, dw_b1, angles,
        cfg.stereo.world_scale_px_per_mm, cfg.calibration.frame_dt_s, fov_valid, post, x, y)
    n_range, n_std = r["range_residual"], r["std_dev"]
    return x, y, U, V, W, valid, n_range, n_std


def test_process_one_pair_stereo_worker_matches_serial_reference(tmp_path):
    cfg = _fast_stereo_cfg()
    angles = _angles(cfg)
    _worker_init(cfg)  # normally the ProcessPoolExecutor initializer -- called directly here
    # since this test invokes the worker function in-process, not via a pool.
    rng = np.random.RandomState(1)
    fa0, fb0, fa1, fb1 = _make_stereo_pair(rng)

    result = process_one_pair_stereo_worker(
        0, "pairA", fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy(), cfg, str(tmp_path), angles)
    assert result["pair_id"] == "pairA"

    saved = np.load(tmp_path / "pairA_stereo_velocity.npz")
    x_ref, y_ref, U_ref, V_ref, W_ref, valid_ref, n_range_ref, n_std_ref = _serial_reference(
        cfg, angles, fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy())

    assert np.array_equal(saved["x"], x_ref)
    assert np.array_equal(saved["y"], y_ref)
    np.testing.assert_array_equal(saved["U"], U_ref)
    np.testing.assert_array_equal(saved["V"], V_ref)
    np.testing.assert_array_equal(saved["W"], W_ref)
    assert np.array_equal(saved["valid"], valid_ref)
    assert result["n_valid"] == int(valid_ref.sum())
    assert result["n_total"] == int(valid_ref.size)
    assert result["n_rejected_range_residual"] == n_range_ref
    assert result["n_rejected_std_dev"] == n_std_ref


def test_run_stereo_batch_parallel_matches_serial_for_every_pair(tmp_path):
    cfg = _fast_stereo_cfg()
    angles = _angles(cfg)
    rng = np.random.RandomState(2)
    pairs = [(f"pair{i}", *_make_stereo_pair(rng, shift0=(i % 3 + 1, i % 2 + 1),
                                              shift1=(i % 2 + 1, i % 3 + 1)))
             for i in range(5)]

    out_dir = tmp_path / "parallel"
    out_dir.mkdir()
    results, cancelled = run_stereo_batch_parallel(
        ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), angles, n_workers=3)

    assert not cancelled
    assert [r["pair_id"] for r in results] == [pid for pid, *_ in pairs]  # submission order preserved

    for pair_id, fa0, fb0, fa1, fb1 in pairs:
        saved = np.load(out_dir / f"{pair_id}_stereo_velocity.npz")
        _, _, U_ref, V_ref, W_ref, valid_ref, _, _ = _serial_reference(
            cfg, angles, fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy())
        np.testing.assert_array_equal(saved["U"], U_ref)
        np.testing.assert_array_equal(saved["V"], V_ref)
        np.testing.assert_array_equal(saved["W"], W_ref)
        assert np.array_equal(saved["valid"], valid_ref)


def test_run_stereo_batch_parallel_output_independent_of_worker_count(tmp_path):
    cfg = _fast_stereo_cfg()
    angles = _angles(cfg)
    rng = np.random.RandomState(3)
    pairs = [(f"pair{i}", *_make_stereo_pair(rng, shift0=(i + 1, i + 1), shift1=(i + 2, i + 1)))
             for i in range(4)]

    dirs = {}
    for n_workers in (1, 2, 4):
        # n_workers=1 here exercises run_stereo_batch_parallel's OWN pool
        # (max_workers=1), not the "skip the pool entirely" branch the
        # real call sites use for n_workers<=1 -- still must agree byte-
        # for-byte with n_workers>1, since a size-1 pool is still exact.
        out_dir = tmp_path / f"n{n_workers}"
        out_dir.mkdir()
        results, cancelled = run_stereo_batch_parallel(
            ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
            cfg, str(out_dir), angles, n_workers=n_workers)
        assert not cancelled
        dirs[n_workers] = out_dir

    for pair_id, *_ in pairs:
        ref = np.load(dirs[1] / f"{pair_id}_stereo_velocity.npz")
        for n_workers in (2, 4):
            other = np.load(dirs[n_workers] / f"{pair_id}_stereo_velocity.npz")
            np.testing.assert_array_equal(ref["U"], other["U"])
            np.testing.assert_array_equal(ref["V"], other["V"])
            np.testing.assert_array_equal(ref["W"], other["W"])
            assert np.array_equal(ref["valid"], other["valid"])


def test_run_stereo_batch_parallel_stops_submitting_after_cancel(tmp_path):
    cfg = _fast_stereo_cfg()
    angles = _angles(cfg)
    rng = np.random.RandomState(4)
    pairs = [(f"pair{i}", *_make_stereo_pair(rng)) for i in range(6)]

    submitted = []

    def cancel_after_two():
        return len(submitted) >= 2

    def on_started(pair_id):
        submitted.append(pair_id)

    out_dir = tmp_path / "cancelled"
    out_dir.mkdir()
    results, cancelled = run_stereo_batch_parallel(
        ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), angles, n_workers=2,
        on_pair_started=on_started, cancel_check=cancel_after_two)

    assert cancelled
    assert len(submitted) == 2
    # in-flight pairs (already submitted before cancellation) still finish
    # and are returned -- never MORE than what was submitted, never fewer
    # than what had time to complete.
    assert len(results) <= len(submitted)


def test_run_stereo_batch_parallel_one_pair_erroring_does_not_lose_others(tmp_path):
    cfg = _fast_stereo_cfg()
    angles = _angles(cfg)
    rng = np.random.RandomState(5)
    good0 = _make_stereo_pair(rng)
    good1 = _make_stereo_pair(rng)
    # Stereo's world_shape (what process_frames actually correlates on) is
    # fixed by cfg for the whole batch, unlike planar where a per-pair raw
    # frame shape smaller than the window size reliably raises -- here a
    # per-pair raw-array dtype scipy's map_coordinates can't handle
    # reliably raises instead, inside dewarp_image, still per-pair and
    # still deterministic.
    bad_a0 = np.full(RAW_SHAPE, "not-a-number", dtype=object)
    bad = (bad_a0, good0[1], good0[2], good0[3])

    pairs = [("good1", *good0), ("bad", *bad), ("good2", *good1)]
    errors = []

    out_dir = tmp_path / "with_error"
    out_dir.mkdir()
    results, cancelled = run_stereo_batch_parallel(
        ((pid, a0, b0, a1, b1) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), angles, n_workers=2,
        on_pair_error=lambda pair_id, exc: errors.append(pair_id))

    assert not cancelled
    assert errors == ["bad"]
    assert {r["pair_id"] for r in results} == {"good1", "good2"}


# ---- real-data-gated: real stereo recording, real frame content ----

@pytest.mark.skipif(not os.path.exists(REAL_SWIRL_SET),
                     reason="real stereo project not available on this machine")
def test_run_stereo_batch_parallel_matches_serial_on_real_swirl_frames(tmp_path):
    """A handful of REAL pairs from J:\\Final_Stereo\\Swirl (read-only --
    never written to), through the real dewarp -> process_frames x2 ->
    combine_stereo_pair pipeline, comparing the TRUE pre-Tier-3 serial
    per-pair loop (_serial_reference, same structure as cli.main.
    handle_pair_stereo / pipeline_worker._process_set_stereo) against
    run_stereo_batch_parallel's output.

    This project's real calibration snapshot is a PinholeOpenCV type that
    this app's exact Polynomial3rdOrder extractor can't decode (see
    test_davis_set_stereo_calibration.py's
    test_read_stereo_calibration_from_real_swirl_set_raises_pinhole_not_exact,
    and SESSION_HANDOFF.md) -- a separate, already-understood limitation,
    not something to work around here. The calibration used below is a
    deliberately trivial IDENTITY mapping (all dx/dy coefs zero, so
    dewarp_image is a real map_coordinates call that happens to be a
    no-op resample) over a small cropped world region of each real raw
    frame -- calibration ACCURACY is irrelevant to what this test checks
    (worker-count/process-boundary exactness), only that the pipeline
    runs on real, non-synthetic, non-degenerate image content and real
    frame shapes. Frames are cropped to a small region purely to keep
    this test's runtime bounded, matching this suite's stated preference
    for small/fast synthetic images elsewhere in this file."""
    from piv_suite.io.davis_set import iter_stereo_from_set

    world_shape = (192, 192)
    cam_settings = CameraMappingSettings(
        x0=world_shape[1] / 2, x_span=world_shape[1], y0=world_shape[0] / 2, y_span=world_shape[0],
        dx_coefs={k: 0.0 for k in COEF_KEYS}, dy_coefs={k: 0.0 for k in COEF_KEYS}, name="identity")

    cfg = ProjectConfig()
    cfg.project.backend = "cpu"
    cfg.project.mode = "stereo"
    cfg.correlation.passes = [PassSettings(32, 0.5)]
    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.save_summary_csv = False
    cfg.output.verbose = False
    cfg.postprocess.range_filter.enabled = False
    cfg.postprocess.global_outlier_std = None
    cfg.postprocess.remove_small_groups_threshold = None
    cfg.stereo = StereoSettings(
        cam0_mapping=cam_settings, cam1_mapping=cam_settings,
        world_shape=world_shape, world_scale_px_per_mm=1.0, dewarp_order=1,
        alpha1_deg=-44.765, alpha2_deg=44.765, beta1_deg=0.0, beta2_deg=0.0,
    )
    angles = _angles(cfg)

    pairs = []
    for pair_id, fa0, fb0, fa1, fb1 in iter_stereo_from_set(REAL_SWIRL_SET):
        crop = tuple(f[:world_shape[0], :world_shape[1]].astype(np.float32) for f in (fa0, fb0, fa1, fb1))
        pairs.append((pair_id, *crop))
        if len(pairs) >= 3:
            break
    assert len(pairs) == 3

    out_dir = tmp_path / "real_parallel"
    out_dir.mkdir()
    results, cancelled = run_stereo_batch_parallel(
        ((pid, a0.copy(), b0.copy(), a1.copy(), b1.copy()) for pid, a0, b0, a1, b1 in pairs),
        cfg, str(out_dir), angles, n_workers=3)
    assert not cancelled
    assert len(results) == 3

    for pair_id, fa0, fb0, fa1, fb1 in pairs:
        saved = np.load(out_dir / f"{pair_id}_stereo_velocity.npz")
        _, _, U_ref, V_ref, W_ref, valid_ref, _, _ = _serial_reference(
            cfg, angles, fa0.copy(), fb0.copy(), fa1.copy(), fb1.copy())
        np.testing.assert_array_equal(saved["U"], U_ref)
        np.testing.assert_array_equal(saved["V"], V_ref)
        np.testing.assert_array_equal(saved["W"], W_ref)
        assert np.array_equal(saved["valid"], valid_ref)
