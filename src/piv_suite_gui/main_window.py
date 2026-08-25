"""Main window: a branded header across the top, then the project/settings
panels (left) and preview/run panels (right, as tabs) below it, with a
status bar along the bottom.

The window itself is freely resizable (no fixed width) -- only the left
panel has a fixed, comfortable width with its QScrollArea's horizontal
scroll bar disabled, so growing the window just gives the preview/run
side (stretch=1) more room rather than ever needing horizontal scrolling
anywhere.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from piv_suite import __version__
from piv_suite.io.davis_set import read_calibration_from_set, resolve_set_paths
from piv_suite_gui.widgets.calibration_panel import CalibrationPanel
from piv_suite_gui.widgets.header_bar import HeaderBar
from piv_suite_gui.widgets.preview_panel import PreviewPanel
from piv_suite_gui.widgets.project_panel import ProjectPanel
from piv_suite_gui.widgets.run_panel import RunPanel
from piv_suite_gui.widgets.settings_panel import SettingsPanel

LEFT_PANEL_WIDTH = 440
RIGHT_PANEL_WIDTH = 640
INITIAL_WINDOW_WIDTH = LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PIV Testing")
        self.resize(INITIAL_WINDOW_WIDTH, 860)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        # Named so the stylesheet can paint the window ground here rather
        # than on a blanket QWidget rule, which would also fill plain
        # container widgets sitting inside the white cards (see theme.py).
        central.setObjectName("centralSurface")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = HeaderBar(version=__version__)
        outer.addWidget(self.header)

        body = QWidget()
        outer.addWidget(body, stretch=1)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- left: project + settings, vertically scrollable only ----
        left_container = QWidget()
        left_container.setMaximumWidth(LEFT_PANEL_WIDTH - 20)  # leaves room for the scrollbar
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self.project_panel = ProjectPanel()
        self.settings_panel = SettingsPanel()
        self.calibration_panel = CalibrationPanel()
        left_layout.addWidget(self.project_panel)
        left_layout.addWidget(self.calibration_panel)
        left_layout.addWidget(self.settings_panel)

        # calibration only matters in stereo mode -- hide it otherwise
        self.calibration_panel.setVisible(self.project_panel.is_stereo)
        self.project_panel.planar_radio.toggled.connect(
            lambda checked: self.calibration_panel.setVisible(not checked))

        # GPU-only settings (batch size, tiling) are greyed out on CPU and
        # vice versa (correlation method, deformation method, etc.) --
        # see settings_panel.set_backend()'s docstring for which fields
        # are in which group.
        self.settings_panel.set_backend(self.project_panel.backend)
        self.project_panel.cpu_radio.toggled.connect(
            lambda checked: self.settings_panel.set_backend("cpu" if checked else "gpu"))

        # .set input: auto-extract pixel pitch / frame Δt straight off the
        # DaVis project the moment it's selected, instead of requiring
        # manual entry (see _on_input_path_changed).
        self.project_panel.input_path_changed.connect(self._on_input_path_changed)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setFixedWidth(LEFT_PANEL_WIDTH)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll, stretch=0)

        # ---- right: preview / run, as tabs ----
        self.tabs = QTabWidget()
        self.preview_panel = PreviewPanel()
        self.run_panel = RunPanel()
        self.tabs.addTab(self.preview_panel, "Preview")
        self.tabs.addTab(self.run_panel, "Run")
        layout.addWidget(self.tabs, stretch=1)

        self._build_status_bar()

        # Run is gated behind a successful preview, matching the original
        # preview_first_snapshot()'s "confirm before batch" UX -- just
        # inline instead of a blocking terminal prompt (see preview_panel.py).
        self.preview_panel.previewed.connect(self.run_panel.set_run_enabled)

        # The header's Run button is a second surface for the SAME action,
        # not a second code path: run_panel stays the single source of
        # truth for whether running is allowed (it also disables while a
        # batch is in flight), and the header mirrors its state.
        self.run_panel.run_enabled_changed.connect(self.header.set_run_enabled)
        self.header.run_requested.connect(self._run_from_header)

        # the preview plot's W color-range row only makes sense in stereo
        # mode (planar has no W component at all)
        self.preview_panel.set_stereo_mode(self.project_panel.is_stereo)
        self.project_panel.planar_radio.toggled.connect(
            lambda checked: self.preview_panel.set_stereo_mode(not checked))

    def _on_input_path_changed(self, path):
        """.set-mode-only. Re-extracts and overwrites the Calibration
        panel's fields every time the input path (or multiset sub-index)
        changes -- always trust the newly selected project's real
        calibration over any prior manual edit. Never crashes the GUI:
        any failure (missing file, bad path, corrupt .set, lvpyio error)
        is caught and surfaced via the status bar."""
        if not self.project_panel.mode_set.isChecked():
            return
        if not path or not os.path.exists(path):
            return  # nothing to read yet, e.g. a fresh/partial manual path
        set_paths, _ = resolve_set_paths(path)
        if not set_paths:
            return
        try:
            calibration = read_calibration_from_set(
                set_paths[0], self.project_panel.multiset_index_spin.value())
        except Exception as e:
            self.statusBar().showMessage(
                f"Couldn't auto-extract calibration from '{os.path.basename(path)}': {e}", 8000)
            return
        self.settings_panel.set_calibration_settings(calibration)
        parts = []
        if calibration.pixel_pitch_mm is not None:
            parts.append(f"pixel pitch {calibration.pixel_pitch_mm:.6g} mm/px")
        if calibration.frame_dt_s is not None:
            parts.append(f"Δt {calibration.frame_dt_s:.6g} s")
        if parts:
            self.statusBar().showMessage("Auto-extracted from DaVis .set: " + ", ".join(parts), 8000)
        else:
            self.statusBar().showMessage(
                "Couldn't auto-extract calibration from this .set -- fill in manually.", 8000)

    def _build_status_bar(self):
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.status_backend_label = QLabel(self.header.status_text())
        bar.addWidget(self.status_backend_label)
        bar.addPermanentWidget(QLabel(self.header.version_text()))

    def _run_from_header(self):
        """Bring the Run tab forward so the user can see progress, then
        click run_panel's own button -- forwarding rather than calling
        _start_run() keeps the enabled-state check in exactly one place."""
        self.tabs.setCurrentWidget(self.run_panel)
        self.run_panel.run_btn.click()

    def _free_backend_resources(self):
        """GPU memory pools + matplotlib Figures -- the two OS-visible
        things a real user reported as still resident after closing the
        window (no closeEvent existed at all before this, so nothing
        ever ran on window close). Split out from closeEvent so app.py's
        QApplication.aboutToQuit can call this SAME cleanup as a second
        safety net (see app.main()'s comment for why aboutToQuit does
        NOT also repeat closeEvent's running-batch stop_and_wait() logic)
        without duplicating it. Safe to call more than once (both paths
        normally fire on an ordinary close) -- free_gpu_pools() on
        already-freed pools and plt.close('all') with no open figures
        are both harmless no-ops, not errors.

        Both cleanup calls are best-effort (never let shutdown cleanup
        itself crash the app): free_gpu_pools() imports cupy internally
        and raises ImportError outright if the GPU backend was never
        installed/available this session (see engines/gpu_engine.py's
        own module docstring for why cupy is lazy-imported everywhere in
        this codebase); matplotlib is always installed but closing figures
        that don't exist is guarded the same way for consistency."""
        try:
            from piv_suite.engines.gpu_engine import free_gpu_pools
            free_gpu_pools()
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass

    def closeEvent(self, event):
        """Stop a running batch and free backend-held memory before the
        window actually closes -- addresses two real user-reported bugs:
        (1) closing the window while a batch is running previously left
        its QThread/PipelineWorker (and, for GPU/parallel-CPU runs, its
        held resources) running underneath the now-destroyed widgets,
        with zero cleanup, since no closeEvent existed at all; (2)
        closing after ANY run (even a finished one) left GPU memory pools
        and matplotlib Figure objects still resident -- nothing on the
        normal end-of-batch path frees those at the APPLICATION level
        (pipeline_worker.py only frees per-engine GPU state between
        pairs/at batch end, not on window close).

        run_panel.stop_and_wait()'s timeout is a bounded wait, not an
        unbounded block -- a worker that genuinely can't stop within it
        (stuck deep inside one blocking correlation call; see
        pipeline_worker.PipelineWorker.force_stop's docstring for why
        that can't always be pre-empted instantly) must not be able to
        hang window close forever. Logged (print(), not a status-bar
        message -- there's no GUI left to see a status bar once the
        window is gone) rather than silently ignored, since it's a real
        signal something would need harder termination to fix."""
        if self.run_panel.is_running():
            stopped = self.run_panel.stop_and_wait(5000)
            if not stopped:
                print("[warn] closeEvent: batch worker didn't stop within 5s -- "
                      "closing anyway, but its thread (and anything it holds, e.g. "
                      "a GPU context) may still be alive after this process exits")

        self._free_backend_resources()
        super().closeEvent(event)
