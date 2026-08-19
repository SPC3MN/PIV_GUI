"""Regression test for a real bug found while auditing GUI parameter
coverage: piv_suite_gui.workers.pipeline_worker.PipelineWorker's planar
batch path never called pipeline.process_frames_tiled -- checking the
GUI's "GPU tiling" box left `engine` as None and called
pipeline.process_frames(engine=None, ...) instead, which crashes
immediately (engine(frame_a, frame_b) on None). Needs real GPU hardware
to actually exercise the tiled path (matches test_gpu_tiling.py's
skip-without-hardware pattern), since the bug is specifically in how the
GUI worker wires the tiling call, not in tiling itself (already covered
by test_gpu_tiling.py).
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
gpu = pytest.importorskip("piv_suite.engines.gpu_engine")

pytestmark = pytest.mark.skipif(
    not gpu.is_gpu_available(), reason="no CUDA-capable GPU / cupy / openpiv_gpu on this machine"
)

from piv_suite.config.schema import ProjectConfig
from piv_suite_gui.workers.pipeline_worker import PipelineWorker


def test_planar_batch_with_tiling_enabled_does_not_crash(qtbot, tmp_path):
    cfg = ProjectConfig()
    cfg.project.backend = "gpu"
    cfg.correlation.use_tiling = True
    cfg.correlation.n_tiles_y = 2
    cfg.correlation.n_tiles_x = 2
    cfg.output.save_npz = False  # keep the test filesystem-free

    rng = np.random.default_rng(1)
    frame_shape = (256, 256)
    frame_a = rng.integers(0, 255, size=frame_shape).astype(np.float64)
    frame_b = np.roll(frame_a, shift=(2, 3), axis=(0, 1))
    pair_source = [("pair0", frame_a, frame_b)]

    worker = PipelineWorker(cfg)
    summary_rows, cancelled = worker._process_set_planar(pair_source, cfg, str(tmp_path))

    assert cancelled is False
    assert len(summary_rows) == 1
    pair_id, elapsed, n_valid, n_total, n_range, n_std = summary_rows[0]
    assert pair_id == "pair0"
    assert n_valid > 0
    assert n_total > 0
