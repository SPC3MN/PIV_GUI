"""Project panel: input source (labeled image pairs / .set DaVis project),
planar vs stereo mode, CPU vs GPU backend, output directory.
"""

from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import ProjectSettings
from piv_suite.engines.registry import is_gpu_available


class ProjectPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpu_available = is_gpu_available()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- input source ----
        input_box = QGroupBox("Input")
        grid = QGridLayout(input_box)

        self.mode_set = QRadioButton(".set (DaVis project)")
        self.mode_loose = QRadioButton("Labeled image pairs (folder)")
        self.mode_set.setChecked(True)
        input_mode_group = QButtonGroup(self)
        input_mode_group.addButton(self.mode_set)
        input_mode_group.addButton(self.mode_loose)

        self.input_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_input)

        grid.addWidget(self.mode_set, 0, 0)
        grid.addWidget(self.mode_loose, 0, 1)
        grid.addWidget(QLabel("Path:"), 1, 0)
        grid.addWidget(self.input_path_edit, 1, 1)
        grid.addWidget(browse_btn, 1, 2)

        self.loose_glob_edit = QLineEdit("*.im7")
        self.suffix_a_edit = QLineEdit("_a.im7")
        self.suffix_b_edit = QLineEdit("_b.im7")
        grid.addWidget(QLabel("Glob (loose mode):"), 2, 0)
        grid.addWidget(self.loose_glob_edit, 2, 1)
        grid.addWidget(QLabel("Frame A / B suffix (planar):"), 3, 0)
        grid.addWidget(self.suffix_a_edit, 3, 1)
        grid.addWidget(self.suffix_b_edit, 3, 2)

        self.suffix_cam0_edit = QLineEdit("_cam1.im7")
        self.suffix_cam1_edit = QLineEdit("_cam2.im7")
        self.stereo_frame_order_combo = QComboBox()
        self.stereo_frame_order_combo.addItems(["camera_major", "frame_major"])
        grid.addWidget(QLabel("Cam0 / Cam1 suffix (stereo):"), 4, 0)
        grid.addWidget(self.suffix_cam0_edit, 4, 1)
        grid.addWidget(self.suffix_cam1_edit, 4, 2)
        grid.addWidget(QLabel("Stereo frame order:"), 5, 0)
        grid.addWidget(self.stereo_frame_order_combo, 5, 1)

        layout.addWidget(input_box)

        # ---- mode / backend ----
        mode_box = QGroupBox("Mode / Backend")
        mode_grid = QGridLayout(mode_box)

        self.planar_radio = QRadioButton("Planar")
        self.stereo_radio = QRadioButton("Stereo")
        self.planar_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.planar_radio)
        mode_group.addButton(self.stereo_radio)

        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio.setChecked(True)
        if not self._gpu_available:
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip(
                "GPU backend unavailable on this machine -- cupy/openpiv-python-gpu "
                "not importable, or no CUDA device detected."
            )
        backend_group = QButtonGroup(self)
        backend_group.addButton(self.cpu_radio)
        backend_group.addButton(self.gpu_radio)

        mode_grid.addWidget(self.planar_radio, 0, 0)
        mode_grid.addWidget(self.stereo_radio, 0, 1)
        mode_grid.addWidget(self.cpu_radio, 1, 0)
        mode_grid.addWidget(self.gpu_radio, 1, 1)

        layout.addWidget(mode_box)

        # ---- output ----
        out_box = QGroupBox("Output")
        out_grid = QGridLayout(out_box)
        self.output_dir_edit = QLineEdit("piv_output")
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_output)
        out_grid.addWidget(QLabel("Output directory:"), 0, 0)
        out_grid.addWidget(self.output_dir_edit, 0, 1)
        out_grid.addWidget(out_browse_btn, 0, 2)
        layout.addWidget(out_box)

        layout.addStretch(1)

    def _browse_input(self):
        if self.mode_set.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Select a DaVis .set project", filter="DaVis set (*.set);;All files (*)")
            if not path:
                path = QFileDialog.getExistingDirectory(self, "Or select a folder of .set files / a raw project folder")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select a folder of labeled image pairs")
        if path:
            self.input_path_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir_edit.setText(path)

    @property
    def is_stereo(self):
        return self.stereo_radio.isChecked()

    @property
    def backend(self):
        return "gpu" if self.gpu_radio.isChecked() else "cpu"

    def get_project_settings(self) -> ProjectSettings:
        return ProjectSettings(
            input_mode="set" if self.mode_set.isChecked() else "loose",
            input_path=self.input_path_edit.text(),
            output_dir=self.output_dir_edit.text(),
            backend=self.backend,
            mode="stereo" if self.is_stereo else "planar",
            loose_glob=self.loose_glob_edit.text(),
            suffix_a=self.suffix_a_edit.text(),
            suffix_b=self.suffix_b_edit.text(),
            suffix_cam0=self.suffix_cam0_edit.text(),
            suffix_cam1=self.suffix_cam1_edit.text(),
            stereo_frame_order=self.stereo_frame_order_combo.currentText(),
        )

    def set_from(self, project: ProjectSettings):
        self.mode_set.setChecked(project.input_mode == "set")
        self.mode_loose.setChecked(project.input_mode == "loose")
        self.input_path_edit.setText(project.input_path)
        self.output_dir_edit.setText(project.output_dir)
        self.cpu_radio.setChecked(project.backend == "cpu")
        if project.backend == "gpu" and self._gpu_available:
            self.gpu_radio.setChecked(True)
        self.planar_radio.setChecked(project.mode == "planar")
        self.stereo_radio.setChecked(project.mode == "stereo")
        self.loose_glob_edit.setText(project.loose_glob)
        self.suffix_a_edit.setText(project.suffix_a)
        self.suffix_b_edit.setText(project.suffix_b)
