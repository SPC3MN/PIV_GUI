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
