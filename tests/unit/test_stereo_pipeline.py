"""End-to-end stereo smoke test: two distinctly-distorted synthetic
cameras -> real calibration dewarping -> real PIV correlation per camera
-> real 3-component reconstruction, checked against a known synthetic 3D
displacement. Exercises the actual wiring cli.main/pipeline_worker use
(CameraMapping.dewarp_image -> engines.*_engine -> processing.pipeline
.combine_stereo_pair), not just each piece in isolation -- those already
have their own focused tests (test_camera_mapping.py, test_reconstruction
.py). GPU variant is skipped without real GPU hardware, matching
test_gpu_tiling.py's pattern.
"""

import numpy as np
import pytest

from piv_suite.calibration.camera_mapping import CameraMapping
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.config.schema import CorrelationSettings, PostProcessSettings, ValidationSettings
from piv_suite.engines.cpu_engine import init_cpu_processor
from piv_suite.processing import pipeline

WORLD_SHAPE = (256, 256)  # (ny, nx) -- small/fast but enough for multi-pass PIV
MARGIN = 20
# matches test_reconstruction.py's documented original-rig config
ALPHA1, ALPHA2 = np.deg2rad(-44.765), np.deg2rad(44.765)
BETA1, BETA2 = 0.0, 0.0
WORLD_SCALE_PX_PER_MM = 1.0  # keep units simple; reconstruction math has its own tests
TARGET_DX, TARGET_DY, TARGET_DZ = 2.0, 1.5, 1.0


def _make_camera(dx_scale):
    ny, nx = WORLD_SHAPE
    # nonzero on every term, two DISTINCT cameras (dx_scale flips sign on
    # one) -- not just an identity/no-op dewarp call
    coefs_dx = {"1": -MARGIN, "s": 1.0 * dx_scale, "s2": 0.5, "s3": 0.0, "t": -0.2,
                "t2": 0.15, "t3": 0.0, "st": 0.3, "s2t": 0.0, "t2s": 0.0}
    coefs_dy = {"1": -MARGIN, "s": -0.15, "s2": 0.1, "s3": 0.0, "t": 0.6,
                "t2": 0.3, "t3": 0.0, "st": -0.2, "s2t": 0.0, "t2s": 0.0}
    return CameraMapping(x0=nx / 2, x_span=nx, y0=ny / 2, y_span=ny,
                          dx_coefs=coefs_dx, dy_coefs=coefs_dy)


def _synthetic_particle_raw(seed, raw_shape, shift_x=0.0, shift_y=0.0):
    rng = np.random.default_rng(seed)
    raw_ny, raw_nx = raw_shape
    img = np.zeros(raw_shape)
    ys = rng.uniform(0, raw_ny, 900)
    xs = rng.uniform(0, raw_nx, 900)
    yy, xx = np.mgrid[0:raw_ny, 0:raw_nx]
    for y, x in zip(ys, xs):
        img += 200 * np.exp(-(((xx - (x + shift_x)) ** 2 + (yy - (y + shift_y)) ** 2) / (2 * 1.2 ** 2)))
    return np.clip(img, 0, 255)


def _build_dewarped_pairs():
    """-> (cam0, cam1, dw_a0, dw_b0, dw_a1, dw_b1) with a known synthetic
    3D displacement (TARGET_DX, TARGET_DY, TARGET_DZ) baked in via each
    camera's own raw-pixel-space shift, computed from the SAME formula
    reconstruct_stereo inverts (dx_i = dX - dZ*tan(alpha_i), etc.)."""
    cam0, cam1 = _make_camera(1.0), _make_camera(-1.0)
    ny, nx = WORLD_SHAPE
    raw_shape = (ny + 2 * MARGIN, nx + 2 * MARGIN)

    dx1 = TARGET_DX - TARGET_DZ * np.tan(ALPHA1)
    dx2 = TARGET_DX - TARGET_DZ * np.tan(ALPHA2)
    dy1 = TARGET_DY - TARGET_DZ * np.tan(BETA1)
    dy2 = TARGET_DY - TARGET_DZ * np.tan(BETA2)

    raw_a0 = _synthetic_particle_raw(101, raw_shape)
    raw_b0 = _synthetic_particle_raw(101, raw_shape, shift_x=dx1, shift_y=dy1)
    raw_a1 = _synthetic_particle_raw(202, raw_shape)
    raw_b1 = _synthetic_particle_raw(202, raw_shape, shift_x=dx2, shift_y=dy2)

    dw_a0 = cam0.dewarp_image(raw_a0, WORLD_SHAPE, order=3)
    dw_b0 = cam0.dewarp_image(raw_b0, WORLD_SHAPE, order=3)
    dw_a1 = cam1.dewarp_image(raw_a1, WORLD_SHAPE, order=3)
    dw_b1 = cam1.dewarp_image(raw_b1, WORLD_SHAPE, order=3)
    return dw_a0, dw_b0, dw_a1, dw_b1


def _run_and_reconstruct(dw_a0, dw_b0, dw_a1, dw_b1, engine0, x, engine1):
    correlation = CorrelationSettings()
    # Post-processing's std-dev/residual filters default ON now (see
    # config/schema.py's PostProcessSettings docstring) -- explicitly
    # disabled here since this test's purpose is checking 3D-reconstruction
    # accuracy against a known displacement, not validation behavior
    # (which has its own coverage in test_validation_restructure.py). A
    # clean synthetic field can still have a few boundary-window vectors
    # with a locally elevated residual from spline-deformation edge
    # effects -- real signal, not something this test should reject.
    disabled_post = PostProcessSettings(global_outlier_std=None)
    disabled_post.range_filter.enabled = False
    disabled_post.range_filter.residual_max = None
    post = disabled_post.for_pipeline()
    u1, v1, valid1, _, _ = pipeline.process_frames(engine0, dw_a0, dw_b0, post)
    u2, v2, valid2, _, _ = pipeline.process_frames(engine1, dw_a1, dw_b1, post)
    valid = valid1 & valid2

    U, V, W = pipeline.combine_stereo_pair(
        u1, v1, u2, v2, (ALPHA1, ALPHA2, BETA1, BETA2), WORLD_SCALE_PX_PER_MM, frame_dt_s=None)
    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    W = np.where(valid, W, np.nan)
    return U, V, W, valid


def test_stereo_pipeline_recovers_known_3d_displacement_cpu():
    dw_a0, dw_b0, dw_a1, dw_b1 = _build_dewarped_pairs()
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    cpu_settings = to_cpu_settings(correlation, validation)

    engine0, x, y = init_cpu_processor(WORLD_SHAPE, cpu_settings)
    engine1, x2, y2 = init_cpu_processor(WORLD_SHAPE, cpu_settings)
    U, V, W, valid = _run_and_reconstruct(dw_a0, dw_b0, dw_a1, dw_b1, engine0, x, engine1)

    assert valid.all()  # clean synthetic particle field -- nothing should be flagged invalid
    assert abs(np.nanmean(U) - TARGET_DX) < 0.15
    assert abs(np.nanmean(V) - TARGET_DY) < 0.15
    assert abs(np.nanmean(W) - TARGET_DZ) < 0.3


def test_stereo_pipeline_recovers_known_3d_displacement_gpu():
    gpu = pytest.importorskip("piv_suite.engines.gpu_engine")
    if not gpu.is_gpu_available():
        pytest.skip("no CUDA-capable GPU / cupy / openpiv_gpu on this machine")

    dw_a0, dw_b0, dw_a1, dw_b1 = _build_dewarped_pairs()
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)

    engine0, x, y = gpu.init_gpu_processor(WORLD_SHAPE, min_search_size, piv_settings)
    engine1, x2, y2 = gpu.init_gpu_processor(WORLD_SHAPE, min_search_size, piv_settings)
    U, V, W, valid = _run_and_reconstruct(dw_a0, dw_b0, dw_a1, dw_b1, engine0, x, engine1)
    gpu.free_gpu_pools()

    assert valid.all()
    assert abs(np.nanmean(U) - TARGET_DX) < 0.15
    assert abs(np.nanmean(V) - TARGET_DY) < 0.15
    assert abs(np.nanmean(W) - TARGET_DZ) < 0.3
