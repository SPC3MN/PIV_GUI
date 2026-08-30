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


def test_dual_planar_settings_roundtrip(tmp_path):
    cfg = ProjectConfig()
    cfg.project.dual_camera = True
    cfg.dual_planar.enabled = True
    cfg.dual_planar.cam0.region_x = 3865.0
    cfg.dual_planar.cam0.region_width = 4144.0
    cfg.dual_planar.cam0.raw_width = 4096
    cfg.dual_planar.cam1.region_x = 0.0
    cfg.dual_planar.canvas_width = 8009
    cfg.dual_planar.canvas_height = 3046
    cfg.dual_planar.scale_x_mm_per_px = 0.0392775752732
    cfg.dual_planar.scale_y_mm_per_px = -0.0392775752732

    d1 = to_dict(cfg)
    cfg2 = from_dict(d1)
    d2 = to_dict(cfg2)
    assert d1 == d2
    assert cfg2.project.dual_camera is True
    assert cfg2.dual_planar.enabled is True
    assert cfg2.dual_planar.cam0.region_x == 3865.0
    assert cfg2.dual_planar.cam1.region_x == 0.0
    assert cfg2.dual_planar.canvas_width == 8009
    assert cfg2.dual_planar.scale_y_mm_per_px == pytest.approx(-0.0392775752732)


def test_remove_small_groups_threshold_default_and_roundtrip(tmp_path):
    cfg = ProjectConfig()
    assert cfg.postprocess.remove_small_groups_threshold == 5  # on by default, matching DaVis
    cfg.postprocess.remove_small_groups_threshold = None
    d1 = to_dict(cfg)
    cfg2 = from_dict(d1)
    assert cfg2.postprocess.remove_small_groups_threshold is None


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


def test_load_project_raises_clear_error_for_malformed_json(tmp_path):
    # .pivproj files are meant to be hand-edited -- a syntax slip (here: a
    # trailing comma) is a realistic mistake, and should fail with an
    # actionable message pointing at the file and the syntax problem, not
    # a raw json.JSONDecodeError traceback.
    path = tmp_path / "proj.pivproj"
    path.write_text('{"project": {"backend": "cpu",}}')  # trailing comma -- invalid JSON

    with pytest.raises(ValueError, match=r"isn't valid JSON"):
        load_project(str(path))


def test_save_project_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "proj.pivproj"
    save_project(str(path), ProjectConfig())
    leftover = [p for p in os.listdir(tmp_path) if p != "proj.pivproj"]
    assert leftover == []


def test_save_project_never_truncates_existing_file_on_write_failure(tmp_path, monkeypatch):
    # A process killed (or a disk that fills up) mid-write must never
    # leave a partially-written .pivproj in place -- save_project() writes
    # to a temp file and os.replace()s it into place, so a failure before
    # that replace can only ever leave the OLD file (if any) untouched.
    path = tmp_path / "proj.pivproj"
    original_cfg = ProjectConfig()
    original_cfg.project.input_path = "/original/data.set"
    save_project(str(path), original_cfg)
    original_bytes = path.read_bytes()

    def _boom(*a, **k):
        raise OSError("simulated disk failure mid-write")

    monkeypatch.setattr(json, "dump", _boom)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_project(str(path), ProjectConfig())

    assert path.read_bytes() == original_bytes  # untouched, not truncated/corrupted
    leftover_tmp = [p for p in os.listdir(tmp_path) if p != "proj.pivproj"]
    assert leftover_tmp == []  # the failed temp file was not left behind either


def test_performance_settings_default_is_auto():
    cfg = ProjectConfig()
    assert cfg.performance.n_workers is None  # None = auto (perf.autotune.recommended_workers())


def test_performance_settings_roundtrip(tmp_path):
    path = tmp_path / "proj.pivproj"
    cfg = ProjectConfig()
    cfg.performance.n_workers = 4
    save_project(str(path), cfg)

    reloaded = load_project(str(path))
    assert reloaded.performance.n_workers == 4


def test_performance_settings_missing_from_older_pivproj_file_falls_back_to_auto(tmp_path):
    # An older .pivproj saved before PerformanceSettings existed has no
    # "performance" key at all -- from_dict must not choke on that.
    path = tmp_path / "proj.pivproj"
    with open(path, "w") as f:
        json.dump({"project": {"backend": "cpu"}}, f)
    cfg = load_project(str(path))
    assert cfg.performance.n_workers is None


def test_per_pass_validation_thresholds_default_to_davis_combined_values_and_roundtrip(tmp_path):
    # per_pass_sig2noise_threshold's MEANING changed (peak-ratio, not
    # peak2mean -- see ValidationSettings' own docstring for the real-
    # dataset evidence) alongside its default value (1.0 -> 1.5, matching
    # DaVis's own peakRatioThreshold), and per_pass_correlation_threshold
    # is a new field (0.5, matching DaVis's own correlationThreshold).
    cfg = ProjectConfig()
    assert cfg.validation.per_pass_sig2noise_threshold == 1.5
    assert cfg.validation.per_pass_correlation_threshold == 0.5

    path = tmp_path / "proj.pivproj"
    cfg.validation.per_pass_sig2noise_threshold = 2.0
    cfg.validation.per_pass_correlation_threshold = 0.6
    save_project(str(path), cfg)

    reloaded = load_project(str(path))
    assert reloaded.validation.per_pass_sig2noise_threshold == 2.0
    assert reloaded.validation.per_pass_correlation_threshold == 0.6


def test_per_pass_correlation_threshold_missing_from_older_pivproj_file_falls_back_to_default(tmp_path):
    # An older .pivproj saved before this field existed has no
    # "per_pass_correlation_threshold" key at all -- from_dict must not
    # choke on that (matches the existing pattern for every other field
    # added to this schema after its own initial release).
    path = tmp_path / "proj.pivproj"
    with open(path, "w") as f:
        json.dump({"project": {"backend": "cpu"}, "validation": {"per_pass_median_threshold": 3.0}}, f)
    cfg = load_project(str(path))
    assert cfg.validation.per_pass_median_threshold == 3.0
    assert cfg.validation.per_pass_correlation_threshold == 0.5


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
    # ValidationSettings' sig2noise_*/validation_first_pass/replace_vectors
    # fields were removed when validation moved entirely to
    # PostProcessSettings (see schema.py's ValidationSettings docstring) --
    # what's left (filter_method/max_filter_iteration/filter_kernel_size/
    # smoothn/smoothn_p) is purely the internal per-pass NaN-safety-fill
    # mechanism, still mapped through as-is.
    corr, val = CorrelationSettings(), ValidationSettings()
    val.max_filter_iteration = 7
    settings = to_cpu_settings(corr, val)
    assert settings["max_filter_iteration"] == 7
    assert settings["windowsizes"] == [64, 32, 32, 32]
    assert "sig2noise_threshold" not in settings
    assert "sig2noise_validate" not in settings
    # per_pass_sig2noise_threshold/per_pass_correlation_threshold ARE
    # forwarded, though (popped by CPUPIVProcess itself, not PIVSettings
    # fields) -- see this function's own docstring.
    assert settings["per_pass_sig2noise_threshold"] == 1.5
    assert settings["per_pass_correlation_threshold"] == 0.5


def test_to_gpu_settings_maps_validation_fields():
    # s2n_tol/median_tol/mad_tol/mean_tol/rms_tol are ALWAYS hard-coded to
    # None regardless of ValidationSettings' content -- validation moved
    # entirely to PostProcessSettings (see to_gpu_settings' docstring).
    corr, val = CorrelationSettings(), ValidationSettings()
    val.filter_kernel_size = 5
    min_search_size, piv_settings = to_gpu_settings(corr, val)
    assert piv_settings["s2n_tol"] is None
    assert piv_settings["replacing_size"] == 5
    assert piv_settings["search_size_iters"] == (1, 3)


def test_to_gpu_settings_replacing_method_matches_piv_gpu_contract():
    # Regression test: piv_gpu.__init__ itself asserts replacing_method
    # is a tuple, one entry per pass (>= len(search_size_iters)), drawn
    # from {'median', 'spring', 'mean'} -- confirmed via a real
    # AssertionError from openpiv_gpu on Windows/CUDA hardware. CPU's
    # filter_method vocabulary ('localmean', 'disk', 'distance') must
    # never be passed straight through to the GPU backend.
    corr = CorrelationSettings()  # 4 passes by default
    val = ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(corr, val)
    _, search_size_iters, _ = passes_to_gpu(corr.passes)

    replacing_method = piv_settings["replacing_method"]
    assert isinstance(replacing_method, tuple)
    assert len(replacing_method) >= len(search_size_iters)
    assert all(m in {"median", "spring", "mean"} for m in replacing_method)


def test_to_gpu_settings_remaining_fields_match_piv_gpu_contract():
    # Confirmed on real Windows/CUDA hardware (constructing piv_gpu AND
    # running a full frame pair through it, including forced outliers to
    # exercise replacement and smoothing) that -- unlike replacing_method
    # -- these fields do NOT need vocabulary translation: piv_gpu.__init__
    # broadcasts a bare str/bool straight into a per-pass tuple itself
    # (`(x,) * self.num_passes if isinstance(x, str/bool) else x`), and
    # canonical schema's default values already fall inside piv_gpu's
    # allowed sets. Pinning the contract here so a future default change
    # (e.g. a new GUI dropdown option) can't silently drift into a value
    # piv_gpu rejects.
    corr = CorrelationSettings()
    val = ValidationSettings()
    _, piv_settings = to_gpu_settings(corr, val)

    assert piv_settings["subpixel_method"] in {"gaussian", "parabolic", "centroid"}
    assert isinstance(piv_settings["revalidate"], bool)
    assert isinstance(piv_settings["smooth"], bool)
