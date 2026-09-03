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

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import CalibrationSettings, DualPlanarSettings, ProjectSettings
from piv_suite.engines.registry import is_gpu_available

from ._util import section_label, style_spin


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
    # Emitted whenever input_path_edit's value is finalized (Browse... or
    # manual editingFinished) or the multiset sub-dataset index changes --
    # main_window uses this to re-trigger .set calibration auto-extraction.
    # Fires regardless of mode_set/mode_loose; main_window decides whether
    # the new path is worth reacting to.
    input_path_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpu_available = is_gpu_available()
        # Auto-extracted (never hand-edited) DaVis SideBySide2D placement/
        # scale data -- see main_window._on_input_path_changed, which
        # overwrites this via set_dual_planar_settings() the moment a
        # matching .set is selected. Empty/disabled default means "not a
        # dual-camera project" until extraction succeeds.
        self._dual_planar = DualPlanarSettings()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ONE section, not four. Input / Pre-processing / Mode / Backend used
        # to be four separate titled group boxes stacked down the panel, which
        # spent four headers and four card borders on what is really a single
        # question: where does the data come from and how should it be read.
        # ---- source ----
        input_box = QGroupBox("SOURCE")
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
        self.input_path_edit.editingFinished.connect(
            lambda: self.input_path_changed.emit(self.input_path_edit.text()))

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
        # a different sub-dataset is effectively a different recording --
        # re-emit so calibration auto-extraction re-runs for it too.
        self.multiset_index_spin.valueChanged.connect(
            lambda _: self.input_path_changed.emit(self.input_path_edit.text()))
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

        # ---- mode + backend, one row inside the same section ----
        mode_backend_row = QHBoxLayout()
        mode_backend_row.setSpacing(10)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)
        mode_layout.addWidget(section_label("Mode"))
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

        # Read-only stand-in for the two radios above, shown INSTEAD of
        # them for .set input -- see _update_input_field_visibility. A
        # .set project always auto-detects its own real acquisition
        # geometry the moment it's selected (main_window._on_input_path_
        # changed, off the project's own calibration), so a manual
        # override there was never actually choosing anything -- it just
        # was left as whatever the last click set until the next .set
        # selection silently overwrote it again. Loose-file input has no
        # calibration to detect from, so it keeps the radios as the only
        # way to say which mode applies -- same "state a value the user
        # cannot usefully change" reasoning as settings_panel's
        # correlation_method_value label.
        self.mode_detected_label = QLabel()
        self.mode_detected_label.setToolTip(
            "Detected from the selected .set project's own calibration. "
            "Switch to 'Labeled image pairs' input to choose Planar/Stereo "
            "manually.")
        mode_layout.addWidget(self.mode_detected_label)
        mode_backend_row.addLayout(mode_layout)
        mode_backend_row.addSpacing(18)

        # planar-only: DaVis "SideBySide2D" dual-camera stitching (see
        # config.schema.DualPlanarSettings) -- auto-checked/auto-extracted
        # when the selected .set's own calibration says SideBySide2D
        # (main_window._on_input_path_changed), but always user-toggleable
        # like every other auto-extracted field in this app.
        self.dual_camera_check = QCheckBox("Dual camera (SideBySide)")
        self.dual_camera_check.setToolTip(
            "DaVis 'SideBySide2D' acquisition: two coplanar cameras, each imaging a "
            "different (overlapping) region of one larger flat plane, stitched into a "
            "wider planar field -- NOT stereo (no triangulation, no 3rd velocity "
            "component). Auto-detected/extracted from the selected .set's own "
            "calibration; both cameras run through the ordinary planar PIV path and "
            "are combined afterward.")
        self.dual_camera_status_label = QLabel()
        self.dual_camera_status_label.setWordWrap(True)
        self.dual_camera_status_label.setVisible(False)

        backend_layout = QHBoxLayout()
        backend_layout.setSpacing(6)
        backend_layout.addWidget(section_label("Backend"))
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
        mode_backend_row.addLayout(backend_layout)
        mode_backend_row.addStretch(1)

        input_layout.addLayout(mode_backend_row)

        # AFTER the Mode row, not before it: dual-camera is planar-only and is
        # cleared and hidden when Stereo is selected, so a reader meets the
        # control that governs it first.
        dual_row = QHBoxLayout()
        dual_row.addWidget(self.dual_camera_check)
        dual_row.addWidget(self.dual_camera_status_label, 1)
        input_layout.addLayout(dual_row)

        layout.addWidget(input_box)

        # ---- physical units (calibration) -- right below project
        # selection, not gated behind an on/off toggle. Both fields
        # matter for every real workflow: a .set project auto-fills them
        # the moment it's selected (see main_window._extract_calibration_
        # for_current_mode), and a loose-file project needs them typed in
        # before the output means anything -- there is no "I deliberately
        # want px/frame" use case common enough to earn a checkbox click
        # ahead of it. ----
        units_box = QGroupBox("PHYSICAL UNITS")
        units_grid = QGridLayout(units_box)
        units_grid.setContentsMargins(6, 6, 6, 6)
        units_grid.setSpacing(4)
        units_grid.setColumnStretch(1, 1)

        self.pixel_pitch_spin = style_spin(QDoubleSpinBox(), decimals=6)
        self.pixel_pitch_spin.setRange(1e-9, 1e6)
        self.pixel_pitch_spin.setValue(1.0)
        self.pixel_pitch_spin.setToolTip(
            "Physical size of one pixel, in mm -- multiplies px/frame "
            "displacements into mm/frame. Auto-filled from the selected "
            ".set project's own scale where available.")
        units_grid.addWidget(QLabel("Pixel pitch (mm/px):"), 0, 0)
        units_grid.addWidget(self.pixel_pitch_spin, 0, 1)

        self.frame_dt_spin = style_spin(QDoubleSpinBox(), decimals=6)
        self.frame_dt_spin.setRange(1e-9, 1e6)
        self.frame_dt_spin.setValue(1.0)
        self.frame_dt_spin.setToolTip(
            "Real time between frame A and frame B, in seconds -- divides "
            "mm/frame into a true velocity per second. Auto-filled from "
            "the selected .set project's own acquisition timing where "
            "available.")
        units_grid.addWidget(QLabel("Frame Δt (s):"), 1, 0)
        units_grid.addWidget(self.frame_dt_spin, 1, 1)
        layout.addWidget(units_box)

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
        frame-order fields, depending on Mode, never both. Dual camera is
        planar-only (stereo already uses both cameras, for triangulation
        instead of stitching).

        Mode itself is only USER-CHOOSABLE in loose mode: a .set project
        always auto-detects its own real acquisition geometry the moment
        it's selected (main_window._on_input_path_changed), so the radios
        are hidden in favor of mode_detected_label there -- see its own
        comment at construction."""
        is_loose = self.mode_loose.isChecked()
        is_stereo = self.stereo_radio.isChecked()

        self.planar_radio.setVisible(is_loose)
        self.stereo_radio.setVisible(is_loose)
        self.mode_detected_label.setVisible(not is_loose)
        self.mode_detected_label.setText("Stereo" if is_stereo else "Planar")

        self.loose_options.setVisible(is_loose)
        self.multiset_index_label.setVisible(not is_loose)
        self.multiset_index_spin.setVisible(not is_loose)

        for w in (self.planar_suffix_label, self.suffix_a_edit, self.suffix_b_edit):
            w.setVisible(is_loose and not is_stereo)
        for w in (self.stereo_suffix_label, self.suffix_cam0_edit, self.suffix_cam1_edit,
                  self.stereo_frame_order_label, self.stereo_frame_order_combo):
            w.setVisible(is_loose and is_stereo)

        self.dual_camera_check.setVisible(not is_stereo)
        if is_stereo:
            self.dual_camera_check.setChecked(False)
        self.dual_camera_status_label.setVisible(
            not is_stereo and self.dual_camera_status_label.text() != "")

    def _browse_input(self):
        if self.mode_set.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Select a DaVis .set project", filter="DaVis set (*.set);;All files (*)")
            if not path:
                path = QFileDialog.getExistingDirectory(self, "Or select a folder of .set files / a raw project folder")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select a folder of labeled image pairs")
        if path:
            self.input_path_edit.setText(path)
            self.input_path_changed.emit(path)  # setText() alone doesn't fire editingFinished

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
            dual_camera=self.dual_camera_check.isChecked() and not self.is_stereo,
            multiset_index=self.multiset_index_spin.value(),
            loose_glob=glob_text,
            suffix_a=_apply_glob_extension(self.suffix_a_edit.text(), glob_text),
            suffix_b=_apply_glob_extension(self.suffix_b_edit.text(), glob_text),
            suffix_cam0=_apply_glob_extension(self.suffix_cam0_edit.text(), glob_text),
            suffix_cam1=_apply_glob_extension(self.suffix_cam1_edit.text(), glob_text),
            stereo_frame_order=self.stereo_frame_order_combo.currentText(),
        )

    def get_calibration_settings(self) -> CalibrationSettings:
        return CalibrationSettings(
            pixel_pitch_mm=self.pixel_pitch_spin.value(),
            frame_dt_s=self.frame_dt_spin.value(),
        )

    def set_calibration_settings(self, settings: CalibrationSettings):
        """Push auto-extracted calibration into the form -- called by
        main_window after a .set path is (re)selected and
        davis_set.read_calibration_from_set runs. A field the .set
        couldn't supply is left at whatever value was already showing --
        there is no "unset" state to fall back to any more (see this
        panel's PHYSICAL UNITS section comment), so the honest thing is to
        leave the last real number in place and let main_window's own
        status-bar message ("couldn't auto-extract ... fill in manually")
        say so, rather than silently reset to the 1.0 placeholder."""
        if settings.pixel_pitch_mm is not None:
            self.pixel_pitch_spin.setValue(settings.pixel_pitch_mm)
        if settings.frame_dt_s is not None:
            self.frame_dt_spin.setValue(settings.frame_dt_s)

    def get_dual_planar_settings(self) -> DualPlanarSettings:
        """The last auto-extracted DaVis SideBySide2D calibration (see
        set_dual_planar_settings, pushed by main_window's .set-selection
        auto-extraction) -- never hand-edited in the GUI, unlike
        StereoSettings' manual-entry form."""
        return self._dual_planar

    def set_dual_planar_settings(self, settings: DualPlanarSettings, status_text=""):
        """Called by main_window after a successful (or failed) dual-
        camera calibration extraction -- always overwrites, matching this
        app's other auto-extraction fields' "always trust the newly
        selected project" convention (see main_window._on_input_path_
        changed's own docstring)."""
        self._dual_planar = settings
        self.dual_camera_status_label.setText(status_text)
        self.dual_camera_status_label.setVisible(bool(status_text) and not self.is_stereo)

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
        self.dual_camera_check.setChecked(project.dual_camera)
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
