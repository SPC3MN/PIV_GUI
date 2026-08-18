"""Settings panel: multi-pass window/overlap schedule, correlation
controls, and the "remove invalid vectors" post-processing step.

Removing invalid vectors uses exactly two detection methods -- "remove if
difference to standard deviation [exceeds n*sigma]" and "remove if
residual [from the local window median exceeds a threshold]", with a
window-size control for the latter -- see config.schema.PostProcessSettings'
docstring for why the engines' own internal per-pass validation
(sig2noise threshold, outlier replace method, smoothn, ...) isn't exposed
here anymore; ValidationSettings still exists internally with fixed
defaults, engines need SOME threshold to run their multi-pass loop, it's
just no longer user-facing.
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

from ._util import fit_table_to_rows, style_spin


class _PassesTable(QGroupBox):
    """Multi-pass window/overlap schedule editor -- an ordered (coarse to
    fine) list of (window_size, overlap_fraction) rows. Both engines'
    original per-pass overlap defaults were isotropic per pass, not
    per-axis, so one overlap_fraction column is enough to match real
    capability (see config.legacy's grouping). Sized to always show every
    row -- no scroll bar of its own."""

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
        self.table.setItem(row, 1, QTableWidgetItem(f"{overlap_fraction:.2f}"))
        fit_table_to_rows(self.table)

    def _remove_row(self):
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)
            fit_table_to_rows(self.table)

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
        fit_table_to_rows(self.table)


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
        self.dt_spin = style_spin(QDoubleSpinBox())
        self.dt_spin.setRange(1e-6, 1e6)
        self.dt_spin.setValue(1.0)
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["gaussian", "centroid", "parabolic"])
        corr_grid.addWidget(QLabel("Δt:"), 0, 0)  # Δt
        corr_grid.addWidget(self.dt_spin, 0, 1)
        corr_grid.addWidget(QLabel("Subpixel method:"), 1, 0)
        corr_grid.addWidget(self.subpixel_combo, 1, 1)

        self.tiling_check = QCheckBox("GPU tiling (large frames)")
        self.tiling_check.setToolTip("Split large frames into a grid of tiles to bound peak GPU memory.")
        self.n_tiles_y_spin = style_spin(QSpinBox()); self.n_tiles_y_spin.setRange(1, 64); self.n_tiles_y_spin.setValue(1)
        self.n_tiles_x_spin = style_spin(QSpinBox()); self.n_tiles_x_spin.setRange(1, 64); self.n_tiles_x_spin.setValue(1)
        corr_grid.addWidget(self.tiling_check, 2, 0, 1, 2)
        corr_grid.addWidget(QLabel("n_tiles_y:"), 3, 0)
        corr_grid.addWidget(self.n_tiles_y_spin, 3, 1)
        corr_grid.addWidget(QLabel("n_tiles_x:"), 4, 0)
        corr_grid.addWidget(self.n_tiles_x_spin, 4, 1)
        layout.addWidget(corr_box)

        # ---- remove invalid vectors ----
        post_box = QGroupBox("Remove invalid vectors")
        post_grid = QGridLayout(post_box)
        post_grid.setContentsMargins(6, 6, 6, 6)
        post_grid.setSpacing(4)

        self.std_filter_check = QCheckBox("Remove if |value - mean| exceeds:")
        self.std_filter_check.setToolTip(
            "Reject vectors more than n·σ (n times the standard "
            "deviation) from the field mean.")
        self.std_filter_check.toggled.connect(self._on_std_filter_toggled)
        self.n_std_spin = style_spin(QDoubleSpinBox())
        self.n_std_spin.setRange(0.1, 100.0)
        self.n_std_spin.setValue(4.0)
        self.n_std_spin.setEnabled(False)
        post_grid.addWidget(self.std_filter_check, 0, 0, 1, 2)
        post_grid.addWidget(self.n_std_spin, 0, 2)
        post_grid.addWidget(QLabel("·σ"), 0, 3)  # ·σ

        self.residual_enabled_check = QCheckBox("Remove if residual exceeds:")
        self.residual_enabled_check.setToolTip(
            "Reject vectors whose distance from their local window median "
            "displacement (px/frame) exceeds this value.")
        self.residual_enabled_check.toggled.connect(self._on_residual_filter_toggled)
        self.residual_max_spin = style_spin(QDoubleSpinBox())
        self.residual_max_spin.setRange(0.0, 1e6)
        self.residual_max_spin.setValue(5.0)
        self.residual_max_spin.setEnabled(False)
        post_grid.addWidget(self.residual_enabled_check, 1, 0, 1, 2)
        post_grid.addWidget(self.residual_max_spin, 1, 2)

        self.window_size_spin = style_spin(QSpinBox())
        self.window_size_spin.setRange(3, 21)
        self.window_size_spin.setSingleStep(2)
        self.window_size_spin.setValue(3)
        self.window_size_spin.setEnabled(False)
        post_grid.addWidget(QLabel("  N (window size):"), 2, 1)
        post_grid.addWidget(self.window_size_spin, 2, 2)

        self.replace_invalid_check = QCheckBox("Interpolate removed vectors")
        post_grid.addWidget(self.replace_invalid_check, 3, 0, 1, 3)

        self.smooth_check = QCheckBox("Gaussian smooth")
        self.smooth_sigma_spin = style_spin(QDoubleSpinBox())
        self.smooth_sigma_spin.setRange(0.1, 100.0)
        self.smooth_sigma_spin.setValue(1.0)
        post_grid.addWidget(self.smooth_check, 4, 0)
        post_grid.addWidget(QLabel("σ:"), 4, 1)  # σ
        post_grid.addWidget(self.smooth_sigma_spin, 4, 2)

        layout.addWidget(post_box)
        layout.addStretch(1)

    def _on_std_filter_toggled(self, checked):
        self.n_std_spin.setEnabled(checked)

    def _on_residual_filter_toggled(self, checked):
        self.residual_max_spin.setEnabled(checked)
        self.window_size_spin.setEnabled(checked)

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
        """The engines' own internal per-pass validation is no longer
        user-facing (see module docstring) -- fixed defaults every time."""
        return ValidationSettings()

    def get_postprocess_settings(self) -> PostProcessSettings:
        range_filter = RangeFilterSettings(
            enabled=self.residual_enabled_check.isChecked(),
            residual_max=self.residual_max_spin.value() if self.residual_enabled_check.isChecked() else None,
            window_size=self.window_size_spin.value(),
        )
        return PostProcessSettings(
            global_outlier_std=self.n_std_spin.value() if self.std_filter_check.isChecked() else None,
            range_filter=range_filter,
            replace_invalid=self.replace_invalid_check.isChecked(),
            smooth_field=self.smooth_check.isChecked(),
            smooth_sigma=self.smooth_sigma_spin.value(),
        )
