"""Regression test for scripts/migrate_legacy_config.py against real
legacy default configs (transcribed from the original repos' DEFAULT_CONFIG
dicts) -- both a GPU stereo config and a CPU planar config."""

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_legacy_config.py"
spec = importlib.util.spec_from_file_location("migrate_legacy_config", SCRIPT_PATH)
migrate_legacy_config = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))
spec.loader.exec_module(migrate_legacy_config)

CAM_COEFS = {k: 0.0 for k in ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")}


def test_migrate_gpu_stereo_legacy_config():
    legacy = {
        "input_mode": "set", "input_path": "D:\\x.set",
        "cam0_mapping": {"x0": 1.0, "x_span": 100.0, "y0": 1.0, "y_span": 100.0,
                         "dx_coefs": CAM_COEFS, "dy_coefs": CAM_COEFS, "name": "cam0"},
        "cam1_mapping": {"x0": 1.0, "x_span": 100.0, "y0": 1.0, "y_span": 100.0,
                         "dx_coefs": CAM_COEFS, "dy_coefs": CAM_COEFS, "name": "cam1"},
        "world_shape": [500, 600], "world_scale_px_per_mm": 10.0,
        "min_search_size": 32,
        "piv_settings": {"search_size_iters": [1, 3], "overlap_ratio": [0.5, 0.75], "dt": 1.0},
        "alpha1_deg": -45.0, "alpha2_deg": 45.0, "beta1_deg": 0.0, "beta2_deg": 0.0,
        "apply_v_sign_flip": True, "replace_invalid": True,
    }
    cfg = migrate_legacy_config.migrate(legacy)
    assert cfg.project.backend == "gpu"
    assert cfg.project.mode == "stereo"
    passes = [(p.window_size, p.overlap_fraction) for p in cfg.correlation.passes]
    assert passes == [(64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75)]
    assert cfg.stereo.cam0_mapping.name == "cam0"
    assert cfg.stereo.world_shape == (500, 600)
    assert cfg.stereo.alpha1_deg == -45.0
    # apply_v_sign_flip was removed from the schema entirely -- a legacy
    # config that had it set is silently dropped, not an error
    assert not hasattr(cfg.postprocess, "apply_v_sign_flip")
    assert cfg.postprocess.replace_invalid is True


def test_migrate_cpu_planar_legacy_config():
    legacy = {
        "input_mode": "loose", "input_path": "/data",
        "cpu_settings": {
            "windowsizes": [64, 32, 32, 32], "overlap": [32, 24, 24, 24],
            "dt": 1.0, "sig2noise_threshold": 1.2,
        },
        "global_outlier_std": 4.0, "smooth_field": True, "smooth_sigma": 2.0,
        "pixel_pitch_mm": 0.01, "frame_dt_s": 0.002,
    }
    cfg = migrate_legacy_config.migrate(legacy)
    assert cfg.project.backend == "cpu"
    assert cfg.project.mode == "planar"
    passes = [(p.window_size, p.overlap_fraction) for p in cfg.correlation.passes]
    assert passes == [(64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75)]
    assert cfg.validation.sig2noise_threshold == 1.2
    assert cfg.postprocess.global_outlier_std == 4.0
    assert cfg.postprocess.smooth_field is True
    assert cfg.calibration.pixel_pitch_mm == 0.01
