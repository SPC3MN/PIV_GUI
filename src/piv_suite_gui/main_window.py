"""Main window: assembles the project/settings panels (left), preview and
run panels (right, as tabs) into one window.

The window itself is freely resizable (no fixed width) -- only the left
panel has a fixed, comfortable width with its QScrollArea's horizontal
scroll bar disabled, so growing the window just gives the preview/run
side (stretch=1) more room rather than ever needing horizontal scrolling
anywhere.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from piv_suite_gui.widgets.calibration_panel import CalibrationPanel
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
        self.setWindowTitle("PIV Suite")
        self.resize(INITIAL_WINDOW_WIDTH, 800)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
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

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setFixedWidth(LEFT_PANEL_WIDTH)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll, stretch=0)

        # ---- right: preview / run, as tabs ----
        tabs = QTabWidget()
        self.preview_panel = PreviewPanel()
        self.run_panel = RunPanel()
        tabs.addTab(self.preview_panel, "Preview")
        tabs.addTab(self.run_panel, "Run")
        layout.addWidget(tabs, stretch=1)

        # Run is gated behind a successful preview, matching the original
        # preview_first_snapshot()'s "confirm before batch" UX -- just
        # inline instead of a blocking terminal prompt (see preview_panel.py).
        self.preview_panel.previewed.connect(self.run_panel.set_run_enabled)

        # the preview plot's W color-range row only makes sense in stereo
        # mode (planar has no W component at all)
        self.preview_panel.set_stereo_mode(self.project_panel.is_stereo)
        self.project_panel.planar_radio.toggled.connect(
            lambda checked: self.preview_panel.set_stereo_mode(not checked))
