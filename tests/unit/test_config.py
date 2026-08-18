import json
import os

import pytest

from piv_suite.config.io import from_dict, load_project, save_project, to_dict
from piv_suite.config.legacy import passes_to_cpu, passes_to_gpu, to_cpu_settings, to_gpu_settings
from piv_suite.config.schema import CorrelationSettings, PassSettings, ProjectConfig, ValidationSettings


def test_json_roundtrip_stability(tmp_path):
    cfg = ProjectConfig()
    cfg.project.input_path = "/some/data.set"
    cfg.postprocess.global_outlier_std = 4.0
    cfg.postprocess.range_filter.enabled = True
    cfg.postprocess.range_filter.residual_max = 5.0
    cfg.postprocess.range_filter.window_size = 5

    d1 = to_dict(cfg)
    cfg2 = from_dict(d1)
    d2 = to_dict(cfg2)
    assert d1 == d2
    assert cfg2.project.input_path == "/some/data.set"
    assert cfg2.postprocess.global_outlier_std == 4.0
    assert cfg2.postprocess.range_filter.residual_max == 5.0
    assert cfg2.postprocess.range_filter.window_size == 5


def test_load_project_writes_defaults_when_missing(tmp_path):
    path = tmp_path / "proj.pivproj"
    assert not path.exists()
    cfg = load_project(str(path))
    assert path.exists()
    assert cfg.project.backend == "cpu"

    with open(path) as f:
        written = json.load(f)
    assert written["project"]["backend"] == "cpu"


def test_load_project_user_keys_override_only_specified(tmp_path):
    path = tmp_path / "proj.pivproj"
    # user file only sets input_path and one correlation field -- everything
    # else should still fall back to defaults, matching the original
    # load_controls() "only include what you're changing" UX
    with open(path, "w") as f:
        json.dump({"project": {"input_path": "/my/data"}, "correlation": {"dt": 2.0}}, f)

    cfg = load_project(str(path))
    assert cfg.project.input_path == "/my/data"
    assert cfg.correlation.dt == 2.0
    # untouched defaults still present
    assert cfg.project.backend == "cpu"
    assert cfg.project.output_dir == "piv_output"
    assert len(cfg.correlation.passes) == 4


def test_save_then_load_is_stable(tmp_path):
    path = tmp_path / "proj.pivproj"
    cfg = ProjectConfig()
    cfg.project.backend = "gpu"
    cfg.project.mode = "stereo"
    save_project(str(path), cfg)

    reloaded = load_project(str(path))
    assert reloaded.project.backend == "gpu"
    assert reloaded.project.mode == "stereo"


# ---- legacy adapter <-> exact original defaults ----

def test_passes_to_cpu_matches_original_default():
    corr = CorrelationSettings()
    windowsizes, overlap = passes_to_cpu(corr.passes)
    assert windowsizes == [64, 32, 32, 32]
    assert overlap == [32, 24, 24, 24]


def test_passes_to_gpu_matches_original_default():
    corr = CorrelationSettings()
    min_search_size, search_size_iters, overlap_ratio = passes_to_gpu(corr.passes)
    assert min_search_size == 32
    assert search_size_iters == (1, 3)
    assert overlap_ratio == (0.5, 0.75)


def test_passes_to_gpu_rejects_non_finest_minimum():
    passes = [PassSettings(32, 0.5), PassSettings(64, 0.75)]  # finest pass isn't smallest
    with pytest.raises(ValueError):
        passes_to_gpu(passes)


def test_to_cpu_settings_maps_validation_fields():
    corr, val = CorrelationSettings(), ValidationSettings()
    val.sig2noise_threshold = 1.3
    settings = to_cpu_settings(corr, val)
    assert settings["sig2noise_threshold"] == 1.3
    assert settings["windowsizes"] == [64, 32, 32, 32]


def test_to_gpu_settings_maps_validation_fields():
    corr, val = CorrelationSettings(), ValidationSettings()
    val.sig2noise_threshold = 1.3
    min_search_size, piv_settings = to_gpu_settings(corr, val)
    assert piv_settings["s2n_tol"] == 1.3
    assert piv_settings["search_size_iters"] == (1, 3)
