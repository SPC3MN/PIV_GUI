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
