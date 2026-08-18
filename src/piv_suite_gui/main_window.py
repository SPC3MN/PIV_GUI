"""Main window: assembles the project/settings panels (left), preview and
run panels (right, as tabs) into one window."""

from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from piv_suite_gui.widgets.calibration_panel import CalibrationPanel
from piv_suite_gui.widgets.preview_panel import PreviewPanel
from piv_suite_gui.widgets.project_panel import ProjectPanel
from piv_suite_gui.widgets.run_panel import RunPanel
from piv_suite_gui.widgets.settings_panel import SettingsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PIV Suite")
        self.resize(1200, 800)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ---- left: project + settings, scrollable ----
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
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

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setMinimumWidth(420)
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
