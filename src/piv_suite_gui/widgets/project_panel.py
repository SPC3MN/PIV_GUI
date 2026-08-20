"""Project panel: input source (labeled image pairs / .set DaVis project),
planar vs stereo mode, CPU vs GPU backend, output directory.

Layout notes:
- Mode (planar/stereo) and Backend (CPU/GPU) are separate group boxes,
  side by side, rather than one combined box.
- The loose-folder-only fields (glob, frame suffixes, stereo frame order)
  are hidden entirely when ".set" input is selected -- they don't apply
  to DaVis project ingestion -- and further narrowed to planar-only vs
  stereo-only suffix fields depending on the Mode selection.
"""

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import PreprocessSettings, ProjectSettings
from piv_suite.engines.registry import is_gpu_available

from ._util import style_spin


def _glob_extension(glob_text):
    """Extract the file extension (e.g. '*.tif' -> '.tif') from the Glob
    field -- the single source of truth for the suffix fields' extension,
    so it only needs to be typed once instead of once per suffix."""
    idx = glob_text.rfind(".")
    return glob_text[idx:] if idx != -1 else ""


def _strip_extension(text):
    """Strip whatever trailing '.xxx' extension is present, if any -- NOT
    specifically the current glob's extension, since the whole point is
    to correctly normalize a marker typed (or pasted from an old-style
    full suffix) with a stale/different extension."""
    idx = text.rfind(".")
    return text[:idx] if idx != -1 else text


def _apply_glob_extension(marker, glob_text):
    """Combine a frame-marker suffix (e.g. '_a') with the extension taken
    from Glob (e.g. '*.tif' -> '.tif'). Strips any extension already
    present in `marker` first, so pasting a full old-style suffix like
    '_a.im7' doesn't end up with two extensions -- Glob's extension always
    wins."""
    return _strip_extension(marker.strip()) + _glob_extension(glob_text)


class ProjectPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpu_available = is_gpu_available()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- input source ----
        input_box = QGroupBox("INPUT")
        input_layout = QVBoxLayout(input_box)
        input_layout.setSpacing(4)

        mode_row = QHBoxLayout()
        self.mode_set = QRadioButton(".set (DaVis project)")
        self.mode_set.setToolTip(
            "A single DaVis .set file, or a folder containing multiple "
            ".set files to batch through one after another.")
        self.mode_loose = QRadioButton("Labeled image pairs (folder)")
        self.mode_loose.setToolTip(
            "A plain folder of image files (not a DaVis .set project) -- "
            "either double-frame buffers (one file per pair) or separate "
            "frame A/B files matched by the suffix fields below.")
        self.mode_set.setChecked(True)
        input_mode_group = QButtonGroup(self)
        input_mode_group.addButton(self.mode_set)
        input_mode_group.addButton(self.mode_loose)
        mode_row.addWidget(self.mode_set)
        mode_row.addWidget(self.mode_loose)
        input_layout.addLayout(mode_row)

        path_grid = QGridLayout()
        path_grid.setColumnStretch(1, 1)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setMinimumWidth(0)
        self.input_path_edit.setToolTip(
            "The .set file/folder (.set mode) or the folder of image "
            "files (loose mode) to read pairs from.")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_input)
        path_grid.addWidget(QLabel("Path:"), 0, 0)
        path_grid.addWidget(self.input_path_edit, 0, 1)
        path_grid.addWidget(browse_btn, 0, 2)

        # .set-mode-only: which sub-dataset inside a multi-set .set file to
        # read -- without this control it's silently always sub-dataset 0.
        self.multiset_index_label = QLabel("Sub-dataset index:")
        self.multiset_index_spin = style_spin(QSpinBox())
        self.multiset_index_spin.setRange(0, 100000)
        self.multiset_index_spin.setToolTip(
            "Which sub-dataset inside a multi-set .set file to process -- "
            "0 is the first. Only applies to .set input.")
        path_grid.addWidget(self.multiset_index_label, 1, 0)
        path_grid.addWidget(self.multiset_index_spin, 1, 1)
        input_layout.addLayout(path_grid)

        # loose-mode-only fields -- hidden entirely in .set mode
        self.loose_options = QWidget()
        loose_grid = QGridLayout(self.loose_options)
        loose_grid.setContentsMargins(0, 4, 0, 0)
        loose_grid.setColumnStretch(1, 1)
        loose_grid.setColumnStretch(2, 1)

        self.loose_glob_edit = QLineEdit("*.im7")
        self.loose_glob_edit.setToolTip(
            "Glob pattern selecting which files in the folder to consider, "
            "e.g. '*.tif' or '*.im7'. Its extension is reused automatically "
            "by the suffix fields below -- type it once here, not again per "
            "suffix.")
        loose_grid.addWidget(QLabel("Glob:"), 0, 0)
        loose_grid.addWidget(self.loose_glob_edit, 0, 1, 1, 2)

        # planar-only suffix fields -- just the distinguishing marker, e.g.
        # "_a"/"_b"; the extension comes from loose_glob_edit above (see
        # _apply_glob_extension), so it's typed once instead of twice.
        self.planar_suffix_label = QLabel("Frame A / B marker:")
        self.suffix_a_edit = QLineEdit("_a")
        self.suffix_a_edit.setToolTip(
            "The part of the filename that marks frame A, e.g. '_a' for "
            "'0001_a.tif' -- no extension needed, that's taken from Glob "
            "above and appended automatically.")
        self.suffix_b_edit = QLineEdit("_b")
        self.suffix_b_edit.setToolTip(
            "The part of the filename that marks frame B, e.g. '_b' for "
            "'0001_b.tif' -- no extension needed, that's taken from Glob "
            "above and appended automatically.")
        loose_grid.addWidget(self.planar_suffix_label, 1, 0)
        loose_grid.addWidget(self.suffix_a_edit, 1, 1)
        loose_grid.addWidget(self.suffix_b_edit, 1, 2)

        # stereo-only suffix / frame-order fields
        self.stereo_suffix_label = QLabel("Cam0 / Cam1 marker:")
        self.suffix_cam0_edit = QLineEdit("_cam1")
        self.suffix_cam0_edit.setToolTip(
            "The part of the filename that marks camera 0, e.g. '_cam1' -- "
            "no extension needed, that's taken from Glob above and "
            "appended automatically.")
        self.suffix_cam1_edit = QLineEdit("_cam2")
        self.suffix_cam1_edit.setToolTip(
            "The part of the filename that marks camera 1, e.g. '_cam2' -- "
            "no extension needed, that's taken from Glob above and "
            "appended automatically.")
        loose_grid.addWidget(self.stereo_suffix_label, 2, 0)
        loose_grid.addWidget(self.suffix_cam0_edit, 2, 1)
        loose_grid.addWidget(self.suffix_cam1_edit, 2, 2)

        self.stereo_frame_order_label = QLabel("Frame order:")
        self.stereo_frame_order_combo = QComboBox()
        self.stereo_frame_order_combo.addItems(["camera_major", "frame_major"])
        self.stereo_frame_order_combo.setToolTip(
            "How a combined 4-frame stereo buffer's frames are ordered: "
            "camera_major = [cam0_A, cam0_B, cam1_A, cam1_B], frame_major "
            "= [cam0_A, cam1_A, cam0_B, cam1_B]. Flip this if the wrong "
            "pair of frames ends up dewarped by each camera's mapping.")
        loose_grid.addWidget(self.stereo_frame_order_label, 3, 0)
        loose_grid.addWidget(self.stereo_frame_order_combo, 3, 1)

        input_layout.addWidget(self.loose_options)
        layout.addWidget(input_box)

        # ---- pre-processing (applied to raw frames, before correlation --
        # for stereo, before dewarping too) ----
        prep_box = QGroupBox("PRE-PROCESSING")
        prep_layout = QHBoxLayout(prep_box)
        self.min_max_check = QCheckBox("Min/max filter (L px):")
        self.min_max_check.setToolTip(
            "LaVision-style sliding min/max background removal + local "
            "contrast normalization, applied to each raw frame before "
            "correlation. For stereo, applied per-camera BEFORE "
            "dewarping.")
        self.min_max_length_spin = style_spin(QSpinBox())
        self.min_max_length_spin.setRange(1, 10000)
        self.min_max_length_spin.setValue(5)
        self.min_max_length_spin.setToolTip(
            "L, in pixels -- the sliding window size for the min/max "
            "filter's background-removal and local-contrast steps.")
        self.min_max_length_spin.setEnabled(False)
        self.min_max_check.toggled.connect(self.min_max_length_spin.setEnabled)
        prep_layout.addWidget(self.min_max_check)
        prep_layout.addWidget(self.min_max_length_spin)
        prep_layout.addStretch(1)
        layout.addWidget(prep_box)

        # ---- mode + backend, separate boxes side by side ----
        mode_backend_row = QHBoxLayout()
        mode_backend_row.setSpacing(6)

        mode_box = QGroupBox("MODE")
        mode_layout = QHBoxLayout(mode_box)
        self.planar_radio = QRadioButton("Planar")
        self.planar_radio.setToolTip(
            "Single-camera PIV -- produces in-plane (u, v) velocity only.")
        self.stereo_radio = QRadioButton("Stereo")
        self.stereo_radio.setToolTip(
            "Two-camera PIV, dewarped and combined into 3-component "
            "(U, V, W) velocity -- requires the Calibration tab to be "
            "filled in.")
        self.planar_radio.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.planar_radio)
        mode_group.addButton(self.stereo_radio)
        mode_layout.addWidget(self.planar_radio)
        mode_layout.addWidget(self.stereo_radio)
        mode_backend_row.addWidget(mode_box)

        backend_box = QGroupBox("BACKEND")
        backend_layout = QHBoxLayout(backend_box)
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setToolTip("Runs on openpiv-python. Always available, no GPU/CUDA needed.")
        self.gpu_radio = QRadioButton("GPU")
        self.gpu_radio.setToolTip(
            "Runs on cupy/openpiv-python-gpu -- much faster on large frames "
            "or batches. Every GPU-only setting in the Correlation group "
            "below is greyed out unless this is selected.")
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
        backend_layout.addWidget(self.cpu_radio)
        backend_layout.addWidget(self.gpu_radio)
        mode_backend_row.addWidget(backend_box)

        layout.addLayout(mode_backend_row)

        # ---- output ----
        out_box = QGroupBox("OUTPUT")
        out_grid = QGridLayout(out_box)
        out_grid.setColumnStretch(1, 1)
        self.output_dir_edit = QLineEdit("piv_output")
        self.output_dir_edit.setToolTip(
            "Where per-pair results (NPZ fields, summary CSV, saved plots) "
            "are written.")
        out_browse_btn = QPushButton("Browse...")
        out_browse_btn.clicked.connect(self._browse_output)
        out_grid.addWidget(QLabel("Output directory:"), 0, 0)
        out_grid.addWidget(self.output_dir_edit, 0, 1)
        out_grid.addWidget(out_browse_btn, 0, 2)
        layout.addWidget(out_box)

        layout.addStretch(1)

        # wire up conditional visibility now that every widget involved exists
        self.mode_set.toggled.connect(self._update_input_field_visibility)
        self.planar_radio.toggled.connect(self._update_input_field_visibility)
        self._update_input_field_visibility()

    def _update_input_field_visibility(self):
        """.set mode needs none of the loose-folder fields at all; loose
        mode needs either the planar suffix pair or the stereo suffix/
        frame-order fields, depending on Mode, never both."""
        is_loose = self.mode_loose.isChecked()
        is_stereo = self.stereo_radio.isChecked()

        self.loose_options.setVisible(is_loose)
        self.multiset_index_label.setVisible(not is_loose)
        self.multiset_index_spin.setVisible(not is_loose)

        for w in (self.planar_suffix_label, self.suffix_a_edit, self.suffix_b_edit):
            w.setVisible(is_loose and not is_stereo)
        for w in (self.stereo_suffix_label, self.suffix_cam0_edit, self.suffix_cam1_edit,
                  self.stereo_frame_order_label, self.stereo_frame_order_combo):
            w.setVisible(is_loose and is_stereo)

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
        glob_text = self.loose_glob_edit.text()
        return ProjectSettings(
            input_mode="set" if self.mode_set.isChecked() else "loose",
            input_path=self.input_path_edit.text(),
            output_dir=self.output_dir_edit.text(),
            backend=self.backend,
            mode="stereo" if self.is_stereo else "planar",
            multiset_index=self.multiset_index_spin.value(),
            loose_glob=glob_text,
            suffix_a=_apply_glob_extension(self.suffix_a_edit.text(), glob_text),
            suffix_b=_apply_glob_extension(self.suffix_b_edit.text(), glob_text),
            suffix_cam0=_apply_glob_extension(self.suffix_cam0_edit.text(), glob_text),
            suffix_cam1=_apply_glob_extension(self.suffix_cam1_edit.text(), glob_text),
            stereo_frame_order=self.stereo_frame_order_combo.currentText(),
        )

    def get_preprocess_settings(self) -> PreprocessSettings:
        return PreprocessSettings(
            min_max_filter_enabled=self.min_max_check.isChecked(),
            min_max_filter_length=self.min_max_length_spin.value(),
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
        self.multiset_index_spin.setValue(project.multiset_index)
        self.loose_glob_edit.setText(project.loose_glob)
        # strip the extension back off -- these fields only ever display
        # the frame-marker part, extension comes from loose_glob_edit
        self.suffix_a_edit.setText(_strip_extension(project.suffix_a))
        self.suffix_b_edit.setText(_strip_extension(project.suffix_b))
        self.suffix_cam0_edit.setText(_strip_extension(project.suffix_cam0))
        self.suffix_cam1_edit.setText(_strip_extension(project.suffix_cam1))
        idx = self.stereo_frame_order_combo.findText(project.stereo_frame_order)
        if idx >= 0:
            self.stereo_frame_order_combo.setCurrentIndex(idx)
        self._update_input_field_visibility()
