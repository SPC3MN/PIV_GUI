"""Preview panel: runs the pipeline on just the first pair and renders it
inline, replacing piv_common.preview_first_snapshot()'s blocking terminal
y/N prompt. Settings can be tweaked and re-previewed as many times as
needed before committing to a full batch run (gates run_panel's Run
button via the `previewed` signal -- see main_window.py). Supports both
planar (single camera) and stereo (two cameras, dewarped and combined via
reconstruct_stereo) preview.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from piv_suite.calibration.camera_mapping import CameraMapping
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.davis_set import iter_pairs_from_set, iter_stereo_from_set, resolve_set_paths
from piv_suite.io.loose_files import iter_pairs_from_loose_files, iter_stereo_from_loose_files
from piv_suite.plotting.preview import make_preview_figure
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration


def _build_engine(backend, frame_shape, correlation, validation):
    factory = get_engine_factory(backend)
    if backend == "cpu":
        settings = {"cpu_settings": to_cpu_settings(correlation, validation)}
    else:
        min_search_size, piv_settings = to_gpu_settings(correlation, validation)
        settings = {"min_search_size": min_search_size, "piv_settings": piv_settings}
    return factory(frame_shape, settings)


class PreviewPanel(QWidget):
    previewed = Signal(bool)  # emits True on a successful preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.preview_btn = QPushButton("Preview first pair")
        self.preview_btn.clicked.connect(self._do_preview)
        layout.addWidget(self.preview_btn)

        self.status_label = QLabel("No preview yet.")
        layout.addWidget(self.status_label)

        self.canvas_container = QVBoxLayout()
        layout.addLayout(self.canvas_container)
        layout.addStretch(1)

    def _first_pair_planar(self, project):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return next(iter(iter_pairs_from_set(set_paths[0], project.multiset_index)))
        return next(iter(iter_pairs_from_loose_files(
            project.input_path, project.loose_glob, project.suffix_a, project.suffix_b)))

    def _first_pair_stereo(self, project):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return next(iter(iter_stereo_from_set(
                set_paths[0], project.multiset_index, project.stereo_frame_order)))
        return next(iter(iter_stereo_from_loose_files(
            project.input_path, project.loose_glob,
            project.suffix_cam0, project.suffix_cam1, project.stereo_frame_order)))

    def _set_canvas(self, fig):
        if self.canvas is not None:
            self.canvas.setParent(None)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas_container.addWidget(self.canvas)

    def _do_preview(self):
        main_window = self.window()
        try:
            project = main_window.project_panel.get_project_settings()
            correlation = main_window.settings_panel.get_correlation_settings()
            validation = main_window.settings_panel.get_validation_settings()
            post = main_window.settings_panel.get_postprocess_settings()

            self.status_label.setText("Running preview...")
            if project.mode == "stereo":
                self._preview_stereo(main_window, project, correlation, validation, post)
            else:
                self._preview_planar(project, correlation, validation, post)
            self.previewed.emit(True)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")
            self.previewed.emit(False)
            raise

    def _preview_planar(self, project, correlation, validation, post):
        pair_id, frame_a, frame_b = self._first_pair_planar(project)
        engine, x, y = _build_engine(project.backend, frame_a.shape, correlation, validation)

        u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post.for_pipeline())
        u, v = apply_calibration(u, v, None, None)

        n_valid, n_total = int(valid.sum()), int(valid.size)
        self.status_label.setText(
            f"Pair '{pair_id}': {elapsed:.3f}s, {n_valid}/{n_total} valid "
            f"(range/residual rejected {rejects['range_residual']}, "
            f"std-dev rejected {rejects['std_dev']})"
        )
        fig = make_preview_figure("planar", x, y, u, v, valid, title=f"Preview -- {pair_id}")
        self._set_canvas(fig)

    def _preview_stereo(self, main_window, project, correlation, validation, post):
        stereo_settings = main_window.calibration_panel.get_settings()
        cam0 = CameraMapping(
            stereo_settings.cam0_mapping.x0, stereo_settings.cam0_mapping.x_span,
            stereo_settings.cam0_mapping.y0, stereo_settings.cam0_mapping.y_span,
            stereo_settings.cam0_mapping.dx_coefs, stereo_settings.cam0_mapping.dy_coefs,
            stereo_settings.cam0_mapping.name,
        )
        cam1 = CameraMapping(
            stereo_settings.cam1_mapping.x0, stereo_settings.cam1_mapping.x_span,
            stereo_settings.cam1_mapping.y0, stereo_settings.cam1_mapping.y_span,
            stereo_settings.cam1_mapping.dx_coefs, stereo_settings.cam1_mapping.dy_coefs,
            stereo_settings.cam1_mapping.name,
        )

        pair_id, fa0, fb0, fa1, fb1 = self._first_pair_stereo(project)
        dw_a0 = cam0.dewarp_image(fa0, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_b0 = cam0.dewarp_image(fb0, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_a1 = cam1.dewarp_image(fa1, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_b1 = cam1.dewarp_image(fb1, stereo_settings.world_shape, stereo_settings.dewarp_order)

        engine0, x, y = _build_engine(project.backend, dw_a0.shape, correlation, validation)
        u1, v1, valid1, elapsed1, r1 = pipeline.process_frames(engine0, dw_a0, dw_b0, post.for_pipeline())
        engine1, _, _ = _build_engine(project.backend, dw_a1.shape, correlation, validation)
        u2, v2, valid2, elapsed2, r2 = pipeline.process_frames(engine1, dw_a1, dw_b1, post.for_pipeline())

        valid = valid1 & valid2
        angles = (np.deg2rad(stereo_settings.alpha1_deg), np.deg2rad(stereo_settings.alpha2_deg),
                  np.deg2rad(stereo_settings.beta1_deg), np.deg2rad(stereo_settings.beta2_deg))
        U, V, W = pipeline.combine_stereo_pair(u1, v1, u2, v2, angles, stereo_settings.world_scale_px_per_mm)
        U = np.where(valid, U, np.nan)
        V = np.where(valid, V, np.nan)
        W = np.where(valid, W, np.nan)

        n_valid, n_total = int(valid.sum()), int(valid.size)
        self.status_label.setText(
            f"Pair '{pair_id}': {elapsed1 + elapsed2:.3f}s, {n_valid}/{n_total} valid "
            f"(range/residual rejected {r1['range_residual'] + r2['range_residual']}, "
            f"std-dev rejected {r1['std_dev'] + r2['std_dev']})"
        )
        fig = make_preview_figure("stereo", x, y, U, V, valid, w=W, title=f"Stereo preview -- {pair_id}")
        self._set_canvas(fig)
