"""Confirms the core invariant of the validation restructure: the engines
themselves never reject a vector -- the final `valid` mask returned by
processing.pipeline.process_frames is determined ENTIRELY by
PostProcessSettings, not by anything that happens during calculation.
See engines/cpu_engine.py's and config/legacy.py's module docstrings for
the full rationale.
"""

import numpy as np
import pytest

from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.config.schema import CorrelationSettings, PostProcessSettings, ValidationSettings
from piv_suite.engines.cpu_engine import init_cpu_processor
from piv_suite.processing import pipeline

SIZE = 256  # large enough for the default 4-pass schedule's spline interpolation between passes


def _noisy_pair(seed_a, seed_b):
    rng_a = np.random.RandomState(seed_a)
    rng_b = np.random.RandomState(seed_b)
    frame_a = (rng_a.rand(SIZE, SIZE) * 255).astype(np.float32)
    frame_b = (rng_b.rand(SIZE, SIZE) * 255).astype(np.float32)  # uncorrelated
    return frame_a, frame_b


def test_process_frames_valid_is_all_true_with_postprocess_disabled_cpu():
    frame_a, frame_b = _noisy_pair(1, 2)
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    cpu_settings = to_cpu_settings(correlation, validation)
    engine, x, y = init_cpu_processor(frame_a.shape, cpu_settings)

    disabled_post = PostProcessSettings(global_outlier_std=None)
    disabled_post.range_filter.enabled = False
    disabled_post.range_filter.residual_max = None

    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, disabled_post.for_pipeline())

    assert valid.all()
    assert rejects["range_residual"] == 0
    assert rejects["std_dev"] == 0


def test_process_frames_valid_is_all_true_with_postprocess_disabled_gpu():
    gpu = pytest.importorskip("piv_suite.engines.gpu_engine")
    if not gpu.is_gpu_available():
        pytest.skip("no CUDA-capable GPU / cupy / openpiv_gpu on this machine")

    frame_a, frame_b = _noisy_pair(1, 2)
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    engine, x, y = gpu.init_gpu_processor(frame_a.shape, min_search_size, piv_settings)

    disabled_post = PostProcessSettings(global_outlier_std=None)
    disabled_post.range_filter.enabled = False
    disabled_post.range_filter.residual_max = None

    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, disabled_post.for_pipeline())
    gpu.free_gpu_pools()

    assert valid.all()
    assert rejects["range_residual"] == 0
    assert rejects["std_dev"] == 0


def test_per_pass_validation_defaults_on_and_switches_module_to_real():
    # ValidationSettings.per_pass_validation defaults True (matching DaVis's
    # own per-pass scheme -- confirmed via real-dataset comparison to raise
    # U/V correlation ~0.6->0.8) -- constructing a CPU engine with default
    # settings must switch openpiv.validation.typical_validation to the
    # real (non-loose) implementation.
    import openpiv.validation
    from piv_suite.engines import _openpiv_speedups

    correlation = CorrelationSettings()
    validation = ValidationSettings()
    assert validation.per_pass_validation is True
    cpu_settings = to_cpu_settings(correlation, validation)
    init_cpu_processor((64, 64), cpu_settings)
    assert openpiv.validation.typical_validation is _openpiv_speedups._REAL_TYPICAL_VALIDATION

    # Explicitly disabling it must restore the NaN-only patch.
    disabled = ValidationSettings(per_pass_validation=False)
    init_cpu_processor((64, 64), to_cpu_settings(correlation, disabled))
    assert openpiv.validation.typical_validation is _openpiv_speedups.loose_typical_validation


def test_per_pass_validation_enabled_switches_to_real_validation_with_tuned_thresholds():
    import openpiv.validation
    from piv_suite.engines import _openpiv_speedups

    correlation = CorrelationSettings()
    validation = ValidationSettings(per_pass_validation=True, per_pass_median_threshold=2.0,
                                     per_pass_median_size=1)
    cpu_settings = to_cpu_settings(correlation, validation)
    engine, x, y = init_cpu_processor((64, 64), cpu_settings)

    assert openpiv.validation.typical_validation is _openpiv_speedups._REAL_TYPICAL_VALIDATION
    assert engine._settings.median_normalized is True
    assert engine._settings.median_threshold == 2.0
    assert engine._settings.median_size == 1

    # Explicitly disabling it must restore the loose patch -- confirms the
    # toggle is live, not sticky.
    disabled = ValidationSettings(per_pass_validation=False)
    init_cpu_processor((64, 64), to_cpu_settings(correlation, disabled))
    assert openpiv.validation.typical_validation is _openpiv_speedups.loose_typical_validation


def test_to_gpu_settings_hardcodes_validation_tolerances_to_none():
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    _min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    for key in ("s2n_tol", "median_tol", "mad_tol", "mean_tol", "rms_tol"):
        assert piv_settings[key] is None
    assert piv_settings["num_replacing_iters"] == 0
    assert piv_settings["revalidate"] is False
