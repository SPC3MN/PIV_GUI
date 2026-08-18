"""Settings panel: multi-pass window/overlap schedule, correlation/
validation controls, and post-processing (std-dev spurious-vector filter +
range/residual filter) -- all the "general PIV controls" the project asks
for, bound to the canonical config.schema dataclasses.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import (
    CorrelationSettings, PassSettings, PostProcessSettings, RangeFilterSettings,
    ValidationSettings,
)

SPINBOX_WIDTH = 80


def _spin_width(spin):
    spin.setMaximumWidth(SPINBOX_WIDTH)
    return spin


class _PassesTable(QGroupBox):
    """Multi-pass window/overlap schedule editor -- an ordered (coarse to
    fine) list of (window_size, overlap_fraction) rows. Both engines'
    original per-pass overlap defaults were isotropic per pass, not
    per-axis, so one overlap_fraction column is enough to match real
    capability (see config.legacy's grouping)."""

    def __init__(self, parent=None):
        super().__init__("Window schedule (coarse -> fine)", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Window size (px)", "Overlap fraction"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add pass")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("Remove last pass")
        remove_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        for size, frac in ((64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75)):
            self._add_row(size, frac)

    def _add_row(self, window_size=32, overlap_fraction=0.5):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(window_size)))
        self.table.setItem(row, 1, QTableWidgetItem(str(overlap_fraction)))

    def _remove_row(self):
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)

    def get_passes(self):
        passes = []
        for row in range(self.table.rowCount()):
            size = int(self.table.item(row, 0).text())
            frac = float(self.table.item(row, 1).text())
            passes.append(PassSettings(size, frac))
        return passes

    def set_passes(self, passes):
        self.table.setRowCount(0)
        for p in passes:
            self._add_row(p.window_size, p.overlap_fraction)


class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.passes_table = _PassesTable()
        layout.addWidget(self.passes_table)

        # ---- correlation ----
        corr_box = QGroupBox("Correlation")
        corr_grid = QGridLayout(corr_box)
        corr_grid.setContentsMargins(6, 6, 6, 6)
        corr_grid.setSpacing(4)
        corr_grid.setColumnStretch(1, 1)
        self.dt_spin = _spin_width(QDoubleSpinBox())
        self.dt_spin.setRange(1e-6, 1e6)
        self.dt_spin.setValue(1.0)
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["gaussian", "centroid", "parabolic"])
        corr_grid.addWidget(QLabel("dt:"), 0, 0)
        corr_grid.addWidget(self.dt_spin, 0, 1)
        corr_grid.addWidget(QLabel("Subpixel method:"), 1, 0)
        corr_grid.addWidget(self.subpixel_combo, 1, 1)

        self.tiling_check = QCheckBox("GPU tiling (large frames)")
        self.tiling_check.setToolTip("Split large frames into a grid of tiles to bound peak GPU memory.")
        self.n_tiles_y_spin = _spin_width(QSpinBox()); self.n_tiles_y_spin.setRange(1, 64); self.n_tiles_y_spin.setValue(1)
        self.n_tiles_x_spin = _spin_width(QSpinBox()); self.n_tiles_x_spin.setRange(1, 64); self.n_tiles_x_spin.setValue(1)
        corr_grid.addWidget(self.tiling_check, 2, 0, 1, 2)
        corr_grid.addWidget(QLabel("n_tiles_y:"), 3, 0)
        corr_grid.addWidget(self.n_tiles_y_spin, 3, 1)
        corr_grid.addWidget(QLabel("n_tiles_x:"), 4, 0)
        corr_grid.addWidget(self.n_tiles_x_spin, 4, 1)
        layout.addWidget(corr_box)

        # ---- validation ----
        val_box = QGroupBox("Validation")
        val_grid = QGridLayout(val_box)
        val_grid.setContentsMargins(6, 6, 6, 6)
        val_grid.setSpacing(4)
        val_grid.setColumnStretch(1, 1)
        self.s2n_threshold_spin = _spin_width(QDoubleSpinBox())
        self.s2n_threshold_spin.setRange(0.0, 100.0)
        self.s2n_threshold_spin.setValue(1.05)
        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItems(["localmean", "disk", "distance"])
        self.max_filter_iter_spin = _spin_width(QSpinBox())
        self.max_filter_iter_spin.setRange(0, 100)
        self.max_filter_iter_spin.setValue(4)
        self.smoothn_check = QCheckBox("smoothn")
        self.smoothn_check.setToolTip("Apply openpiv's smoothn() between passes.")
        self.smoothn_p_spin = _spin_width(QDoubleSpinBox())
        self.smoothn_p_spin.setRange(0.0, 10.0)
        self.smoothn_p_spin.setValue(0.05)

        val_grid.addWidget(QLabel("sig2noise threshold:"), 0, 0)
        val_grid.addWidget(self.s2n_threshold_spin, 0, 1)
        val_grid.addWidget(QLabel("Outlier replace method:"), 1, 0)
        val_grid.addWidget(self.filter_method_combo, 1, 1)
        val_grid.addWidget(QLabel("Max replace iterations:"), 2, 0)
        val_grid.addWidget(self.max_filter_iter_spin, 2, 1)
        val_grid.addWidget(self.smoothn_check, 3, 0)
        val_grid.addWidget(self.smoothn_p_spin, 3, 1)
        layout.addWidget(val_box)

        # ---- post-processing ----
        post_box = QGroupBox("Post-processing")
        post_grid = QGridLayout(post_box)
        post_grid.setContentsMargins(6, 6, 6, 6)
        post_grid.setSpacing(4)

        self.sign_flip_check = QCheckBox("Flip v sign")
        post_grid.addWidget(self.sign_flip_check, 0, 0)

        self.std_filter_check = QCheckBox("Std-dev filter")
        self.std_filter_check.setToolTip("Reject vectors more than n_std standard deviations from the field mean.")
        self.std_filter_check.toggled.connect(self._on_std_filter_toggled)
        self.n_std_spin = _spin_width(QDoubleSpinBox())
        self.n_std_spin.setRange(0.1, 100.0)
        self.n_std_spin.setValue(4.0)
        self.n_std_spin.setEnabled(False)
        post_grid.addWidget(self.std_filter_check, 1, 0)
        post_grid.addWidget(QLabel("n_std:"), 1, 1)
        post_grid.addWidget(self.n_std_spin, 1, 2)

        self.replace_invalid_check = QCheckBox("Interpolate invalid vectors")
        post_grid.addWidget(self.replace_invalid_check, 2, 0, 1, 2)

        self.smooth_check = QCheckBox("Gaussian smooth")
        self.smooth_sigma_spin = _spin_width(QDoubleSpinBox())
        self.smooth_sigma_spin.setRange(0.1, 100.0)
        self.smooth_sigma_spin.setValue(1.0)
        post_grid.addWidget(self.smooth_check, 3, 0)
        post_grid.addWidget(QLabel("sigma:"), 3, 1)
        post_grid.addWidget(self.smooth_sigma_spin, 3, 2)

        layout.addWidget(post_box)

        # ---- range/residual filter ----
        range_box = QGroupBox("Range / residual filter")
        range_grid = QGridLayout(range_box)
        range_grid.setContentsMargins(6, 6, 6, 6)
        range_grid.setSpacing(4)
        self.range_enabled_check = QCheckBox("Enabled")
        range_grid.addWidget(self.range_enabled_check, 0, 0, 1, 3)

        self.mag_min_spin = _spin_width(QDoubleSpinBox()); self.mag_min_spin.setRange(-1e6, 1e6); self.mag_min_spin.setValue(0.0)
        self.mag_max_spin = _spin_width(QDoubleSpinBox()); self.mag_max_spin.setRange(-1e6, 1e6); self.mag_max_spin.setValue(1e6)
        self.mag_enabled_check = QCheckBox("Magnitude range:")
        self.mag_enabled_check.setToolTip("Reject vectors whose displacement magnitude (px/frame) falls outside [min, max].")
        range_grid.addWidget(self.mag_enabled_check, 1, 0)
        range_grid.addWidget(self.mag_min_spin, 1, 1)
        range_grid.addWidget(self.mag_max_spin, 1, 2)

        self.residual_enabled_check = QCheckBox("Local residual max:")
        self.residual_enabled_check.setToolTip(
            "Reject vectors whose distance from their local neighborhood median "
            "displacement (px/frame) exceeds this value.")
        self.residual_max_spin = _spin_width(QDoubleSpinBox()); self.residual_max_spin.setRange(0.0, 1e6); self.residual_max_spin.setValue(5.0)
        range_grid.addWidget(self.residual_enabled_check, 2, 0)
        range_grid.addWidget(self.residual_max_spin, 2, 1)

        self.neighborhood_spin = _spin_width(QSpinBox())
        self.neighborhood_spin.setRange(3, 21)
        self.neighborhood_spin.setSingleStep(2)
        self.neighborhood_spin.setValue(3)
        range_grid.addWidget(QLabel("Neighborhood:"), 3, 0)
        range_grid.addWidget(self.neighborhood_spin, 3, 1)

        layout.addWidget(range_box)
        layout.addStretch(1)

    def _on_std_filter_toggled(self, checked):
        self.n_std_spin.setEnabled(checked)

    def get_correlation_settings(self) -> CorrelationSettings:
        return CorrelationSettings(
            passes=self.passes_table.get_passes(),
            dt=self.dt_spin.value(),
            subpixel_method=self.subpixel_combo.currentText(),
            use_tiling=self.tiling_check.isChecked(),
            n_tiles_y=self.n_tiles_y_spin.value(),
            n_tiles_x=self.n_tiles_x_spin.value(),
        )

    def get_validation_settings(self) -> ValidationSettings:
        return ValidationSettings(
            sig2noise_threshold=self.s2n_threshold_spin.value(),
            filter_method=self.filter_method_combo.currentText(),
            max_filter_iteration=self.max_filter_iter_spin.value(),
            smoothn=self.smoothn_check.isChecked(),
            smoothn_p=self.smoothn_p_spin.value(),
        )

    def get_postprocess_settings(self) -> PostProcessSettings:
        range_filter = RangeFilterSettings(
            enabled=self.range_enabled_check.isChecked(),
            magnitude_range=(self.mag_min_spin.value(), self.mag_max_spin.value())
                if self.mag_enabled_check.isChecked() else None,
            residual_max=self.residual_max_spin.value() if self.residual_enabled_check.isChecked() else None,
            neighborhood_size=self.neighborhood_spin.value(),
        )
        return PostProcessSettings(
            apply_v_sign_flip=self.sign_flip_check.isChecked(),
            global_outlier_std=self.n_std_spin.value() if self.std_filter_check.isChecked() else None,
            range_filter=range_filter,
            replace_invalid=self.replace_invalid_check.isChecked(),
            smooth_field=self.smooth_check.isChecked(),
            smooth_sigma=self.smooth_sigma_spin.value(),
        )
