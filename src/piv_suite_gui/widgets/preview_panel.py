"""Preview panel: runs the pipeline on one selected pair (defaulting to the
first) and renders it inline, replacing piv_common.preview_first_snapshot()'s
blocking terminal y/N prompt. Settings can be tweaked and re-previewed as
many times as needed before committing to a full batch run (gates
run_panel's Run button via the `previewed` signal -- see main_window.py).
Supports both planar (single camera) and stereo (two cameras, dewarped and
combined via reconstruct_stereo) preview.

The plot itself (make_preview_figure) supports per-component (U, V, and W
for stereo) filled contours with auto or manually-scaled colorbars, an
optional vector overlay on top, and a choice of colormap -- see
piv_suite.plotting.preview.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)
from PySide6.QtCore import QObject, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np

from piv_suite.calibration.camera_mapping import build_camera_mapping
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.davis_set import (
    get_dual_planar_from_set, get_pair_from_set, get_stereo_from_set,
    list_pair_ids_from_set, resolve_set_paths,
)
from piv_suite.io.loose_files import (
    get_pair_from_loose_files, get_stereo_from_loose_files,
    list_pair_ids_from_loose_files, list_pair_ids_stereo_from_loose_files,
)
from piv_suite.plotting.preview import AVAILABLE_COLORMAPS, make_preview_figure
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration
from piv_suite.processing.preprocess import apply_preprocess_pair

from ._util import style_spin


def _build_engine(backend, frame_shape, correlation, validation):
    factory = get_engine_factory(backend)
    if backend == "cpu":
        settings = {"cpu_settings": to_cpu_settings(correlation, validation)}
    else:
        min_search_size, piv_settings = to_gpu_settings(correlation, validation)
        settings = {"min_search_size": min_search_size, "piv_settings": piv_settings}
    return factory(frame_shape, settings)


class _PreviewWorker(QObject):
    """Runs one preview's COMPUTE off the GUI thread.

    Correlating a single full-resolution pair is not fast: a real
    3008x4096 pair through the default 4-pass schedule takes ~45s on a
    24-core workstation. This used to run inline on the GUI thread (with
    a comment asserting "a single pair is fast enough that a background
    thread wasn't worth the complexity"), which meant the window stopped
    answering Windows' paint/ping messages for that whole time and the OS
    painted it "(Not Responding)" -- reported from real use, and
    reproducible on any full-resolution dataset.

    Only the computation moves here. Figure construction stays on the GUI
    thread (matplotlib's Qt canvas must be built there), so this emits
    plain arrays and the panel renders them in its own slot."""

    finished = Signal(object)   # dict payload -> PreviewPanel._render
    failed = Signal(str)

    def __init__(self, compute_fn):
        super().__init__()
        self._compute_fn = compute_fn

    def run(self):
        try:
            self.finished.emit(self._compute_fn())
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc))


class PreviewPanel(QWidget):
    previewed = Signal(bool)  # emits True on a successful preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self._range_rows = {}
        self._preview_thread = None
        self._preview_worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        pair_row = QHBoxLayout()
        pair_row.addWidget(QLabel("Pair:"))
        self.pair_combo = QComboBox()
        self.pair_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pair_combo.setToolTip("Which pair (by index/id) to preview -- click Refresh pairs after changing the input path/glob/suffixes.")
        pair_row.addWidget(self.pair_combo, 1)
        self.refresh_pairs_btn = QPushButton("Refresh pairs")
        self.refresh_pairs_btn.setToolTip("Re-scan the current input settings and repopulate the Pair list above.")
        self.refresh_pairs_btn.clicked.connect(self._refresh_pairs)
        pair_row.addWidget(self.refresh_pairs_btn)
        layout.addLayout(pair_row)

        self.preview_btn = QPushButton("Preview selected pair")
        self.preview_btn.setProperty("accent", True)
        self.preview_btn.clicked.connect(self._do_preview)
        layout.addWidget(self.preview_btn)

        # Indeterminate ("busy") mode -- there's no meaningful percentage
        # for a single preview pair, just a running/not-running state.
        # Hidden except while a preview is in progress.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("No preview yet.")
        layout.addWidget(self.status_label)

        layout.addWidget(self._build_plot_options_box())

        self.canvas_container = QVBoxLayout()
        layout.addLayout(self.canvas_container)
        layout.addStretch(1)

    def _build_plot_options_box(self):
        box = QGroupBox("PLOT OPTIONS")
        box_layout = QVBoxLayout(box)

        toggle_row = QHBoxLayout()
        self.show_contour_check = QCheckBox("Contours")
        self.show_contour_check.setChecked(True)
        self.show_contour_check.setToolTip("Filled color contour of each component (U, V, and W for stereo), one subplot per component with its own colorbar.")
        self.show_vectors_check = QCheckBox("Vectors")
        self.show_vectors_check.setChecked(False)
        self.show_vectors_check.setToolTip("Quiver arrow overlay on every subplot, drawn on top of the contour when both are on.")
        toggle_row.addWidget(self.show_contour_check)
        toggle_row.addWidget(self.show_vectors_check)
        toggle_row.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(AVAILABLE_COLORMAPS)
        self.cmap_combo.setToolTip("Color map used for the filled contours.")
        toggle_row.addWidget(self.cmap_combo)
        toggle_row.addStretch(1)
        box_layout.addLayout(toggle_row)

        # Per-component (U, V, and stereo-only W) color-range controls --
        # auto-scaled (min/max of that pair's data) by default, but each
        # component can be pinned to a manual range independently (e.g. to
        # compare pairs on a fixed scale instead of one that jumps around
        # per-pair).
        range_grid = QGridLayout()
        range_grid.addWidget(QLabel("Component"), 0, 0)
        range_grid.addWidget(QLabel("Auto range"), 0, 1)
        range_grid.addWidget(QLabel("Min"), 0, 2)
        range_grid.addWidget(QLabel("Max"), 0, 3)
        for row, name in enumerate(("U", "V", "W"), start=1):
            label = QLabel(name)
            auto_check = QCheckBox()
            auto_check.setChecked(True)
            auto_check.setToolTip(f"Auto-scale {name}'s colorbar to this pair's own min/max. Uncheck to pin a fixed Min/Max instead -- useful for comparing pairs on the same scale.")
            min_spin = style_spin(QDoubleSpinBox())
            min_spin.setRange(-1e6, 1e6)
            min_spin.setEnabled(False)
            min_spin.setToolTip(f"Manual lower bound for {name}'s colorbar (enabled when Auto range is off).")
            max_spin = style_spin(QDoubleSpinBox())
            max_spin.setRange(-1e6, 1e6)
            max_spin.setEnabled(False)
            max_spin.setToolTip(f"Manual upper bound for {name}'s colorbar (enabled when Auto range is off).")
            auto_check.toggled.connect(
                lambda checked, mn=min_spin, mx=max_spin: (mn.setEnabled(not checked), mx.setEnabled(not checked)))
            range_grid.addWidget(label, row, 0)
            range_grid.addWidget(auto_check, row, 1)
            range_grid.addWidget(min_spin, row, 2)
            range_grid.addWidget(max_spin, row, 3)
            self._range_rows[name] = (label, auto_check, min_spin, max_spin)
        box_layout.addLayout(range_grid)

        self.set_stereo_mode(False)
        return box

    def set_stereo_mode(self, is_stereo):
        """Called by main_window when the project Mode (planar/stereo)
        changes -- the W range row only applies to stereo."""
        for w in self._range_rows["W"]:
            w.setVisible(is_stereo)

    def _get_ranges(self):
        ranges = {}
        for name, (_label, auto_check, min_spin, max_spin) in self._range_rows.items():
            if not auto_check.isChecked():
                ranges[name] = (min_spin.value(), max_spin.value())
        return ranges

    def _list_pair_ids(self, project):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            # a .set's pair ids are index-based off the same underlying
            # dataset for both planar and stereo -- one listing works for
            # either mode.
            return list_pair_ids_from_set(set_paths[0], project.multiset_index)
        if project.mode == "stereo":
            return list_pair_ids_stereo_from_loose_files(
                project.input_path, project.loose_glob, project.suffix_cam0, project.suffix_cam1)
        return list_pair_ids_from_loose_files(
            project.input_path, project.loose_glob, project.suffix_a, project.suffix_b)

    def _refresh_pairs(self):
        main_window = self.window()
        project = main_window.project_panel.get_project_settings()
        try:
            pair_ids = self._list_pair_ids(project)
        except Exception as e:
            self.status_label.setText(f"Couldn't list pairs: {e}")
            return
        self.pair_combo.clear()
        self.pair_combo.addItems(pair_ids)
        if pair_ids:
            self.pair_combo.setCurrentIndex(0)
            self.status_label.setText(f"{len(pair_ids)} pair(s) found.")
        else:
            self.status_label.setText("No pairs found for the current input settings.")

    def _first_pair_planar(self, project, index):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return get_pair_from_set(set_paths[0], index, project.multiset_index)
        return get_pair_from_loose_files(
            project.input_path, index, project.loose_glob, project.suffix_a, project.suffix_b)

    def _first_pair_stereo(self, project, index):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return get_stereo_from_set(set_paths[0], index, project.multiset_index, project.stereo_frame_order)
        return get_stereo_from_loose_files(
            project.input_path, index, project.loose_glob,
            project.suffix_cam0, project.suffix_cam1, project.stereo_frame_order)

    def _first_pair_dual_planar(self, project, index):
        # .set-mode-only -- see cli.main/pipeline_worker's identical
        # restriction, a SideBySide2D combined 4-frame buffer has no
        # loose-file equivalent in this app.
        set_paths, _ = resolve_set_paths(project.input_path)
        return get_dual_planar_from_set(set_paths[0], index, project.multiset_index)

    def _set_canvas(self, fig):
        if self.canvas is not None:
            self.canvas.setParent(None)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas_container.addWidget(self.canvas)

    def _do_preview(self):
        main_window = self.window()
        if self.pair_combo.count() == 0:
            self._refresh_pairs()
        if self.pair_combo.count() == 0:
            self.status_label.setText("Preview failed: no pairs found for the current input settings.")
            self.previewed.emit(False)
            return
        index = self.pair_combo.currentIndex()

        # Settings are read HERE, on the GUI thread -- they touch widgets,
        # which is not safe from a worker thread. Only the correlation
        # itself is handed off (see _PreviewWorker for why it has to be).
        try:
            project = main_window.project_panel.get_project_settings()
            preprocess = main_window.project_panel.get_preprocess_settings()
            correlation = main_window.settings_panel.get_correlation_settings()
            validation = main_window.settings_panel.get_validation_settings()
            post = main_window.settings_panel.get_postprocess_settings()
            calibration = main_window.settings_panel.get_calibration_settings()
            if project.mode == "stereo":
                stereo_settings = main_window.calibration_panel.get_settings()
                def compute():
                    return self._compute_stereo(project, preprocess, correlation, validation,
                                                 post, calibration, stereo_settings, index)
            elif project.dual_camera:
                dual_planar_settings = main_window.project_panel.get_dual_planar_settings()
                def compute():
                    return self._compute_dual_planar(project, preprocess, correlation, validation,
                                                       post, calibration, dual_planar_settings, index)
            else:
                def compute():
                    return self._compute_planar(project, preprocess, correlation, validation,
                                                 post, calibration, index)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")
            self.previewed.emit(False)
            return

        # Disabling the button also blocks double-clicks from queuing a
        # second preview while one is still running.
        self.preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Running preview...")

        self._preview_thread = QThread()
        self._preview_worker = _PreviewWorker(compute)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_thread.start()

    def _teardown_preview_thread(self):
        self.progress_bar.setVisible(False)
        self.preview_btn.setEnabled(True)
        thread = getattr(self, "_preview_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait()
            self._preview_thread = None
            self._preview_worker = None

    def _on_preview_failed(self, message):
        self._teardown_preview_thread()
        self.status_label.setText(f"Preview failed: {message}")
        self.previewed.emit(False)

    def _on_preview_finished(self, result):
        self._teardown_preview_thread()
        try:
            self._render(result)
        except Exception as e:
            self.status_label.setText(f"Preview failed while rendering: {e}")
            self.previewed.emit(False)
            raise
        self.previewed.emit(True)

    def _render(self, r):
        """Build the figure and status line from a worker payload. Runs on
        the GUI thread (matplotlib's Qt canvas requires it)."""
        self.status_label.setText(
            f"Pair '{r['pair_id']}': {r['elapsed']:.3f}s, {r['n_valid']}/{r['n_total']} valid "
            f"(range/residual rejected {r['n_range']}, std-dev rejected {r['n_std']})"
        )
        common = dict(
            show_contour=self.show_contour_check.isChecked(),
            show_vectors=self.show_vectors_check.isChecked(),
            cmap=self.cmap_combo.currentText(), ranges=self._get_ranges(),
        )
        if r["kind"] == "stereo":
            fig = make_preview_figure(
                "stereo", r["x"], r["y"], r["u"], r["v"], r["valid"], w=r["w"],
                title=f"Stereo preview -- {r['pair_id']}", **common)
        else:
            fig = make_preview_figure(
                "planar", r["x"], r["y"], r["u"], r["v"], r["valid"],
                title=f"Preview -- {r['pair_id']}", **common)
        self._set_canvas(fig)

    def _compute_planar(self, project, preprocess, correlation, validation, post, calibration, index):
        pair_id, frame_a, frame_b = self._first_pair_planar(project, index)
        frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, preprocess)
        engine, x, y = _build_engine(project.backend, frame_a.shape, correlation, validation)

        u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post.for_pipeline())
        u, v = apply_calibration(u, v, calibration.pixel_pitch_mm, calibration.frame_dt_s)

        return dict(kind="planar", pair_id=pair_id, x=x, y=y, u=u, v=v, valid=valid,
                    elapsed=elapsed, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=rejects["range_residual"], n_std=rejects["std_dev"])

    def _compute_dual_planar(self, project, preprocess, correlation, validation, post, calibration,
                              dual_planar_settings, index):
        pair_id, fa0, fb0, fa1, fb1 = self._first_pair_dual_planar(project, index)
        fa0, fb0 = apply_preprocess_pair(fa0, fb0, preprocess)
        fa1, fb1 = apply_preprocess_pair(fa1, fb1, preprocess)

        # RAW (row-down, unflipped) engine.coords -- NOT the display-
        # flipped x, y _build_engine's factory returns -- see
        # pipeline.combine_dual_planar_pair's docstring for why.
        engine0, _x0f, _y0f = _build_engine(project.backend, fa0.shape, correlation, validation)
        u0, v0, valid0, elapsed0, r0 = pipeline.process_frames(engine0, fa0, fb0, post.for_pipeline())
        x0, y0 = engine0.coords

        engine1, _x1f, _y1f = _build_engine(project.backend, fa1.shape, correlation, validation)
        u1, v1, valid1, elapsed1, r1 = pipeline.process_frames(engine1, fa1, fb1, post.for_pipeline())
        x1, y1 = engine1.coords

        X, Y, U, V, valid = pipeline.combine_dual_planar_pair(
            (u0, v0, x0, y0, valid0), (u1, v1, x1, y1, valid1),
            dual_planar_settings, calibration.frame_dt_s)

        return dict(kind="planar", pair_id=pair_id, x=X, y=Y, u=U, v=V, valid=valid,
                    elapsed=elapsed0 + elapsed1, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=r0["range_residual"] + r1["range_residual"],
                    n_std=r0["std_dev"] + r1["std_dev"])

    def _compute_stereo(self, project, preprocess, correlation, validation, post, calibration,
                         stereo_settings, index):
        cam0 = build_camera_mapping(stereo_settings.cam0_mapping, stereo_settings.cam0_mapping_plane2,
                                    stereo_settings.sheet_z_mm)
        cam1 = build_camera_mapping(stereo_settings.cam1_mapping, stereo_settings.cam1_mapping_plane2,
                                    stereo_settings.sheet_z_mm)

        pair_id, fa0, fb0, fa1, fb1 = self._first_pair_stereo(project, index)
        fa0, fb0 = apply_preprocess_pair(fa0, fb0, preprocess)
        fa1, fb1 = apply_preprocess_pair(fa1, fb1, preprocess)
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
        U, V, W = pipeline.combine_stereo_pair(u1, v1, u2, v2, angles, stereo_settings.world_scale_px_per_mm,
                                                calibration.frame_dt_s)
        U = np.where(valid, U, np.nan)
        V = np.where(valid, V, np.nan)
        W = np.where(valid, W, np.nan)

        return dict(kind="stereo", pair_id=pair_id, x=x, y=y, u=U, v=V, w=W, valid=valid,
                    elapsed=elapsed1 + elapsed2, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=r1["range_residual"] + r2["range_residual"],
                    n_std=r1["std_dev"] + r2["std_dev"])
