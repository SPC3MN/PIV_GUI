"""GPU tiling correctness -- needs real cupy + a CUDA device, so every
test here is skipped on machines without GPU hardware (matches
engines.gpu_engine's own lazy-import philosophy). Run on real Windows/CUDA
hardware, not simulated -- see default_tile_margin()'s docstring in
engines.gpu_engine for how these numbers were measured.
"""

import numpy as np
import pytest

from piv_suite.config.schema import CorrelationSettings, PostProcessSettings, ValidationSettings
from piv_suite.config.legacy import to_gpu_settings
from piv_suite.processing import pipeline

gpu = pytest.importorskip("piv_suite.engines.gpu_engine")

pytestmark = pytest.mark.skipif(
    not gpu.is_gpu_available(), reason="no CUDA-capable GPU / cupy / openpiv_gpu on this machine"
)


def _synthetic_shifted_pair(shape=(512, 512), shift=(2, 3), seed=1):
    rng = np.random.default_rng(seed)
    frame_a = rng.integers(0, 255, size=shape).astype(np.float64)
    frame_b = np.roll(frame_a, shift=shift, axis=(0, 1))
    return frame_a, frame_b


def test_tiled_lattice_matches_non_tiled_exactly():
    # The tiled path stitches an unstructured point set back together from
    # independently-built per-tile piv_gpu instances -- confirm that
    # produces the SAME window lattice as one non-tiled piv_gpu instance
    # over the whole frame (no gaps at tile seams, no duplicated vectors
    # from overlapping halos).
    correlation, validation = CorrelationSettings(), ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    margin = gpu.default_tile_margin(min_search_size, piv_settings)
    post = PostProcessSettings().for_pipeline()
    frame_a, frame_b = _synthetic_shifted_pair()

    init_fn = lambda shape: gpu._init_gpu_processor_raw(shape, min_search_size, piv_settings)
    x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
        frame_a, frame_b, post, init_fn, n_tiles_y=2, n_tiles_x=2, margin_px=margin,
        free_pools_fn=gpu.free_gpu_pools,
    )
    engine, x2, y2 = gpu.init_gpu_processor((512, 512), min_search_size, piv_settings)
    u2, v2, valid2, elapsed2, rejects2 = pipeline.process_frames(engine, frame_a, frame_b, post)
    del engine
    gpu.free_gpu_pools()

    tiled_pts = set(zip(np.round(x, 3), np.round(y, 3)))
    nontiled_pts = set(zip(np.round(x2.ravel(), 3), np.round(y2.ravel(), 3)))
    assert len(x) == len(tiled_pts), "duplicate vectors in tiled output"
    assert tiled_pts == nontiled_pts


def test_default_tile_margin_keeps_seam_error_at_noise_floor():
    # Regression test: default_tile_margin()'s old 1x-coarsest-window
    # value left vectors near tile seams up to ~0.22px off from what
    # non-tiled processing produces on the IDENTICAL frame pair (~8x the
    # algorithm's own ~0.01-0.03px noise floor) -- confirmed via a real
    # comparison run on Windows/CUDA hardware, not simulated. 3x brought
    # the max seam error down to ~0.001px. This pins the fixed default
    # against a strict noise-floor threshold so a future change to the
    # multiplier can't silently regress tiling accuracy.
    correlation, validation = CorrelationSettings(), ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    margin = gpu.default_tile_margin(min_search_size, piv_settings)
    post = PostProcessSettings().for_pipeline()
    frame_a, frame_b = _synthetic_shifted_pair()

    init_fn = lambda shape: gpu._init_gpu_processor_raw(shape, min_search_size, piv_settings)
    x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
        frame_a, frame_b, post, init_fn, n_tiles_y=2, n_tiles_x=2, margin_px=margin,
        free_pools_fn=gpu.free_gpu_pools,
    )
    engine, x2, y2 = gpu.init_gpu_processor((512, 512), min_search_size, piv_settings)
    u2, v2, valid2, elapsed2, rejects2 = pipeline.process_frames(engine, frame_a, frame_b, post)
    del engine
    gpu.free_gpu_pools()

    order_tiled = np.lexsort((y, x))
    order_nontiled = np.lexsort((y2.ravel(), x2.ravel()))
    ut, vt = u[order_tiled], v[order_tiled]
    un, vn = u2.ravel()[order_nontiled], v2.ravel()[order_nontiled]

    max_diff = max(np.nanmax(np.abs(ut - un)), np.nanmax(np.abs(vt - vn)))
    assert max_diff < 0.01, (
        f"seam error {max_diff:.4f}px exceeds the ~0.01px noise floor -- "
        "default_tile_margin()'s multiplier may have regressed"
    )


def test_run_tiled_rejects_core_smaller_than_finest_window():
    # Regression test: confirmed on real GPU hardware that too many tiles
    # for the frame size doesn't just lose accuracy or cleanly error --
    # a tile core below the finest pass's window size silently produced a
    # DIFFERENT-SIZED, misaligned stitched result (961 vectors instead of
    # the correct 841 on a 256x256 test frame at 20x20 tiles) instead of
    # a gap or an error. piv_gpu's own too-small-to-process assertion only
    # guards the padded region, not this lattice-alignment failure mode,
    # so run_tiled needs its own guard. 16x16 tiles on a 256x256 frame
    # (min_search_size=32) gives an 16px core, below the 32px window --
    # confirmed this exact combination silently broke before the fix.
    correlation, validation = CorrelationSettings(), ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    margin = gpu.default_tile_margin(min_search_size, piv_settings)
    post = PostProcessSettings().for_pipeline()
    frame_a, frame_b = _synthetic_shifted_pair(shape=(256, 256))

    init_fn = lambda shape: gpu._init_gpu_processor_raw(shape, min_search_size, piv_settings)
    with pytest.raises(ValueError, match="smallest tile core"):
        pipeline.process_frames_tiled(
            frame_a, frame_b, post, init_fn, n_tiles_y=16, n_tiles_x=16, margin_px=margin,
            free_pools_fn=gpu.free_gpu_pools,
        )


def test_smooth_setting_actually_changes_intermediate_passes():
    # Regression test for a bug in the third-party openpiv_gpu library
    # (patched locally, see gpu_process.py's smooth_fields -- not part of
    # this repo, so this test guards against the SYMPTOM rather than
    # calling the buggy code directly): smooth_fields() checked the
    # truthiness of the per-pass TUPLE attribute instead of the per-pass
    # bool just unpacked from it, and a non-empty tuple is always truthy
    # regardless of its contents -- so smoothing ran on every intermediate
    # pass no matter what `smooth` was set to. That means smooth=False and
    # smooth=True would have produced statistically IDENTICAL results
    # (both effectively smoothed); confirm they now measurably differ on
    # a noisy synthetic frame pair, where smoothing has a real effect.
    correlation, validation = CorrelationSettings(), ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)

    frame_shape = (256, 256)
    rng = np.random.default_rng(3)
    frame_a = rng.integers(0, 255, size=frame_shape).astype(np.float64)
    frame_b = np.roll(frame_a, shift=(2, 3), axis=(0, 1))
    # sprinkle noise so intermediate-pass smoothing has something to smooth
    noise_mask = rng.random(frame_shape) < 0.03
    frame_b = np.where(noise_mask, rng.integers(0, 255, size=frame_shape), frame_b).astype(np.float64)

    settings_off = dict(piv_settings, smooth=False)
    settings_on = dict(piv_settings, smooth=True, smoothing_par=1.0)

    engine_off, _, _ = gpu.init_gpu_processor(frame_shape, min_search_size, settings_off)
    u_off, v_off = engine_off(frame_a, frame_b)
    del engine_off
    gpu.free_gpu_pools()

    engine_on, _, _ = gpu.init_gpu_processor(frame_shape, min_search_size, settings_on)
    u_on, v_on = engine_on(frame_a, frame_b)
    del engine_on
    gpu.free_gpu_pools()

    max_diff = max(np.nanmax(np.abs(u_off - u_on)), np.nanmax(np.abs(v_off - v_on)))
    assert max_diff > 0.01, (
        f"smooth=False and smooth=True produced nearly identical output (max diff {max_diff:.5f}px) "
        "-- smoothing may be running unconditionally again regardless of the smooth setting"
    )
