"""Settings panel: multi-pass window/overlap schedule, correlation
controls, per-pass internal stability fill, physical-unit calibration,
and the "remove invalid vectors" post-processing step.

Two DIFFERENT groups look superficially similar (both have a "smoothn"/
"smooth" and a fill/replace concept) but serve entirely different
purposes -- kept visually separate:
- "Per-pass stability (internal)" group's "Per-pass smoothn"/fill
  filter/kernel run INSIDE the multi-pass engine loop, between passes --
  they affect what the NEXT (finer) pass sees, not just the final
  reported field, and critically do NOT reject or validate any vector
  (see the group's own tooltip) -- they only prevent numerically
  unstable (NaN) cells from poisoning the next pass's deformation.
- "Remove invalid vectors (post-processing)" group is the SOLE place a
  vector is ever marked invalid -- its std-dev filter and universal-
  outlier-detection (residual) filter run ONCE, after the engine has
  already produced its final field. "Gaussian smooth"/"Interpolate
  removed vectors" here are a separate, later, purely cosmetic/gap-
  filling step, not a third detection method.

CPU/GPU-specific fields (correlation_method, deformation_method,
interpolation_order, filter_method are CPU-only; batch_size, tiling/
n_tiles/tile_margin are GPU-only) stay visible regardless of the
selected backend -- set_backend() (called by main_window whenever
project_panel's CPU/GPU radio changes) greys out whichever group doesn't
apply, on top of the tooltip on each already noting this.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import (
    CalibrationSettings, CorrelationSettings, PassSettings, PerformanceSettings,
    PostProcessSettings, RangeFilterSettings, ValidationSettings,
)
from piv_suite.perf.autotune import recommended_workers

from ._util import fit_table_to_rows, style_spin


class _PassesTable(QGroupBox):
    """Multi-pass window/overlap schedule editor -- an ordered (coarse to
    fine) list of (window_size, overlap_fraction) rows. Both engines'
    original per-pass overlap defaults were isotropic per pass, not
    per-axis, so one overlap_fraction column is enough to match real
    capability (see config.legacy's grouping). Sized to always show every
    row -- no scroll bar of its own."""

    def __init__(self, parent=None):
        super().__init__("WINDOW SCHEDULE (COARSE → FINE)", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Window size (px)", "Overlap fraction"])
        self.table.setToolTip(
            "Ordered coarse-to-fine interrogation window schedule. Each "
            "row is one pass: 'Window size' is the interrogation window's "
            "side length in pixels, 'Overlap fraction' is how much "
            "consecutive windows overlap (0-1). The LAST row's window "
            "size must be the SMALLEST across all rows -- required by the "
            "GPU backend's convention, and the CPU backend agrees with it "
            "when both run the same schedule.")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add pass")
        add_btn.setToolTip("Append a new pass after the last row, coarse-to-fine order.")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("Remove last pass")
        remove_btn.setToolTip("Remove the finest (last) pass. At least one pass must remain.")
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
        corr_box = QGroupBox("CORRELATION")
        corr_grid = QGridLayout(corr_box)
        corr_grid.setContentsMargins(6, 6, 6, 6)
        corr_grid.setSpacing(4)
        corr_grid.setColumnStretch(1, 1)
        # decimals=6 (not the shared style_spin default of 2) -- a 2-decimal
        # cap made any dt below 0.01 (e.g. real microsecond-scale PIV
        # timings) impossible to enter at all, not just imprecise.
        self.dt_spin = style_spin(QDoubleSpinBox(), decimals=6)
        self.dt_spin.setRange(1e-6, 1e6)
        self.dt_spin.setValue(1.0)
        self.dt_spin.setToolTip(
            "Time between frame A and frame B, in the correlation's own "
            "time unit (not necessarily seconds) -- results come out in "
            "px/dt. Not the same as the 'Frame Δt (s)' calibration field "
            "below, which additionally converts px/frame to physical "
            "units.")
        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["gaussian", "centroid", "parabolic"])
        self.subpixel_combo.setToolTip(
            "Peak-fitting method used to refine each correlation peak to "
            "sub-pixel accuracy. 'gaussian' is the standard choice for "
            "typical PIV particle images.")
        corr_grid.addWidget(QLabel("Δt:"), 0, 0)  # Δt
        corr_grid.addWidget(self.dt_spin, 0, 1)
        corr_grid.addWidget(QLabel("Subpixel method:"), 1, 0)
        corr_grid.addWidget(self.subpixel_combo, 1, 1)

        self.correlation_method_combo = QComboBox()
        self.correlation_method_combo.addItems(["circular", "linear"])
        self.correlation_method_combo.setToolTip("CPU backend only -- ignored on GPU.")
        corr_grid.addWidget(QLabel("Correlation method:"), 2, 0)
        corr_grid.addWidget(self.correlation_method_combo, 2, 1)

        self.deformation_method_combo = QComboBox()
        self.deformation_method_combo.addItems(["symmetric", "second image"])
        self.deformation_method_combo.setToolTip("CPU backend only -- ignored on GPU.")
        corr_grid.addWidget(QLabel("Deformation method:"), 3, 0)
        corr_grid.addWidget(self.deformation_method_combo, 3, 1)

        self.interpolation_order_spin = style_spin(QSpinBox())
        self.interpolation_order_spin.setRange(0, 5)
        self.interpolation_order_spin.setValue(3)
        self.interpolation_order_spin.setToolTip("CPU backend only -- ignored on GPU.")
        corr_grid.addWidget(QLabel("Interpolation order:"), 4, 0)
        corr_grid.addWidget(self.interpolation_order_spin, 4, 1)

        self.batch_size_check = QCheckBox("Set GPU batch size:")
        self.batch_size_check.setToolTip("GPU backend only. Unchecked = piv_gpu's own default (process all windows in one batch).")
        self.batch_size_check.toggled.connect(lambda c: self.batch_size_spin.setEnabled(c))
        self.batch_size_spin = style_spin(QSpinBox())
        self.batch_size_spin.setRange(1, 1_000_000)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setToolTip("Number of interrogation windows processed per GPU batch.")
        self.batch_size_spin.setEnabled(False)
        corr_grid.addWidget(self.batch_size_check, 5, 0)
        corr_grid.addWidget(self.batch_size_spin, 5, 1)

        self.tiling_check = QCheckBox("GPU tiling (large frames)")
        self.tiling_check.setToolTip("Split large frames into a grid of tiles to bound peak GPU memory. GPU backend only.")
        self.n_tiles_y_spin = style_spin(QSpinBox()); self.n_tiles_y_spin.setRange(1, 64); self.n_tiles_y_spin.setValue(1)
        self.n_tiles_y_spin.setToolTip("Number of tiles to split each frame into along the y (row) axis. GPU backend only.")
        self.n_tiles_x_spin = style_spin(QSpinBox()); self.n_tiles_x_spin.setRange(1, 64); self.n_tiles_x_spin.setValue(1)
        self.n_tiles_x_spin.setToolTip("Number of tiles to split each frame into along the x (column) axis. GPU backend only.")
        corr_grid.addWidget(self.tiling_check, 6, 0, 1, 2)
        corr_grid.addWidget(QLabel("n_tiles_y:"), 7, 0)
        corr_grid.addWidget(self.n_tiles_y_spin, 7, 1)
        corr_grid.addWidget(QLabel("n_tiles_x:"), 8, 0)
        corr_grid.addWidget(self.n_tiles_x_spin, 8, 1)

        self.tile_margin_check = QCheckBox("Override tile margin (px):")
        self.tile_margin_check.setToolTip(
            "Unchecked = auto (3x the finest pass's window size -- see "
            "engines.gpu_engine.default_tile_margin for why 1x isn't enough).")
        self.tile_margin_check.toggled.connect(lambda c: self.tile_margin_spin.setEnabled(c))
        self.tile_margin_spin = style_spin(QSpinBox())
        self.tile_margin_spin.setRange(1, 100_000)
        self.tile_margin_spin.setValue(96)
        self.tile_margin_spin.setToolTip("Manual overlap (px) between adjacent tiles, overriding the auto default.")
        self.tile_margin_spin.setEnabled(False)
        corr_grid.addWidget(self.tile_margin_check, 9, 0)
        corr_grid.addWidget(self.tile_margin_spin, 9, 1)
        layout.addWidget(corr_box)

        # ---- physical units (calibration) ----
        cal_box = QGroupBox("PHYSICAL UNITS")
        cal_box.setToolTip(
            "Unset = results stay in px/frame (px/frame for stereo's U/V/W "
            "too, unless world_scale_px_per_mm on the Calibration tab "
            "already converts in-plane units).")
        cal_grid = QGridLayout(cal_box)
        cal_grid.setContentsMargins(6, 6, 6, 6)
        cal_grid.setSpacing(4)

        self.pixel_pitch_check = QCheckBox("Pixel pitch (mm/px):")
        self.pixel_pitch_check.setToolTip(
            "Physical size of one pixel in mm -- multiplies px/frame "
            "displacements into mm/frame. Leave unchecked to keep results "
            "in px/frame.")
        self.pixel_pitch_check.toggled.connect(lambda c: self.pixel_pitch_spin.setEnabled(c))
        self.pixel_pitch_spin = style_spin(QDoubleSpinBox(), decimals=6)
        self.pixel_pitch_spin.setRange(1e-9, 1e6)
        self.pixel_pitch_spin.setToolTip("Physical size of one pixel, in mm.")
        self.pixel_pitch_spin.setEnabled(False)
        cal_grid.addWidget(self.pixel_pitch_check, 0, 0)
        cal_grid.addWidget(self.pixel_pitch_spin, 0, 1)

        self.frame_dt_check = QCheckBox("Frame Δt (s):")
        self.frame_dt_check.setToolTip(
            "Real time between frame A and frame B, in seconds -- divides "
            "mm/frame (or px/frame) into a true velocity per second. "
            "Combine with Pixel pitch above for full px -> physical-unit "
            "velocity conversion.")
        self.frame_dt_check.toggled.connect(lambda c: self.frame_dt_spin.setEnabled(c))
        self.frame_dt_spin = style_spin(QDoubleSpinBox(), decimals=6)
        self.frame_dt_spin.setRange(1e-9, 1e6)
        self.frame_dt_spin.setToolTip("Real time between frame A and frame B, in seconds.")
        self.frame_dt_spin.setEnabled(False)
        cal_grid.addWidget(self.frame_dt_check, 1, 0)
        cal_grid.addWidget(self.frame_dt_spin, 1, 1)
        layout.addWidget(cal_box)

        # ---- per-pass internal stability fill (runs inside the multi-
        # pass loop -- does NOT reject/validate vectors, see tooltip) ----
        val_box = QGroupBox("PER-PASS STABILITY (INTERNAL)")
        val_box.setToolTip(
            "Runs BETWEEN multi-pass iterations, feeding the next (finer) "
            "pass. This does NOT reject or validate vectors -- it only "
            "fills numerically-unstable (NaN) cells so a bad correlation "
            "in one pass can't poison the next pass's deformation grid. "
            "All actual vector validation happens once, in 'Remove "
            "invalid vectors (post-processing)' below, after the final "
            "field is produced.")
        val_grid = QGridLayout(val_box)
        val_grid.setContentsMargins(6, 6, 6, 6)
        val_grid.setSpacing(4)
        val_grid.setColumnStretch(1, 1)

        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItems(["localmean", "disk", "distance"])
        self.filter_method_combo.setToolTip(
            "CPU backend only -- GPU always uses its own 'median' "
            "replacement between passes (a different vocabulary; see "
            "config.legacy.to_gpu_settings' docstring).")
        val_grid.addWidget(QLabel("Fill filter:"), 0, 0)
        val_grid.addWidget(self.filter_method_combo, 0, 1)

        self.max_filter_iter_spin = style_spin(QSpinBox())
        self.max_filter_iter_spin.setRange(0, 100)
        self.max_filter_iter_spin.setValue(4)
        self.max_filter_iter_spin.setToolTip(
            "How many times to re-run the NaN-fill between passes. "
            "Applies on both backends (GPU's num_replacing_iters).")
        val_grid.addWidget(QLabel("Max fill iterations:"), 1, 0)
        val_grid.addWidget(self.max_filter_iter_spin, 1, 1)

        self.filter_kernel_size_spin = style_spin(QSpinBox())
        self.filter_kernel_size_spin.setRange(1, 20)
        self.filter_kernel_size_spin.setValue(2)
        self.filter_kernel_size_spin.setToolTip(
            "Neighborhood radius (in vectors) used to fill a NaN cell "
            "between passes. Applies on both backends (GPU's "
            "replacing_size).")
        val_grid.addWidget(QLabel("Fill kernel size:"), 2, 0)
        val_grid.addWidget(self.filter_kernel_size_spin, 2, 1)

        self.smoothn_check = QCheckBox("Per-pass smoothn")
        self.smoothn_check.setToolTip(
            "Smooth the field between passes (before it's used to deform "
            "the next, finer pass's windows) using the smoothn algorithm. "
            "Applies on both backends.")
        self.smoothn_check.toggled.connect(lambda c: self.smoothn_p_spin.setEnabled(c))
        self.smoothn_p_spin = style_spin(QDoubleSpinBox(), decimals=4)
        self.smoothn_p_spin.setRange(0.0, 100.0)
        self.smoothn_p_spin.setValue(0.05)
        self.smoothn_p_spin.setToolTip("smoothn's own smoothing strength parameter -- higher = smoother.")
        self.smoothn_p_spin.setEnabled(False)
        val_grid.addWidget(self.smoothn_check, 3, 0)
        val_grid.addWidget(self.smoothn_p_spin, 3, 1)
        layout.addWidget(val_box)

        # ---- remove invalid vectors (the SOLE validation step) ----
        post_box = QGroupBox("REMOVE INVALID VECTORS (POST-PROCESSING)")
        post_box.setToolTip(
            "The only place vectors are ever marked invalid -- runs once, "
            "after the final field is produced. Nothing during "
            "calculation itself rejects a vector (see 'Per-pass stability "
            "(internal)' above).")
        post_grid = QGridLayout(post_box)
        post_grid.setContentsMargins(6, 6, 6, 6)
        post_grid.setSpacing(4)

        self.std_filter_check = QCheckBox("Remove if |value - mean| exceeds:")
        self.std_filter_check.setChecked(True)
        self.std_filter_check.setToolTip(
            "Reject vectors more than n·σ (n times the standard "
            "deviation) from the field mean.")
        self.std_filter_check.toggled.connect(self._on_std_filter_toggled)
        self.n_std_spin = style_spin(QDoubleSpinBox())
        self.n_std_spin.setRange(0.1, 100.0)
        self.n_std_spin.setValue(3.0)
        self.n_std_spin.setToolTip("Number of standard deviations from the field mean beyond which a vector is rejected.")
        self.n_std_spin.setEnabled(True)
        post_grid.addWidget(self.std_filter_check, 0, 0, 1, 2)
        post_grid.addWidget(self.n_std_spin, 0, 2)
        post_grid.addWidget(QLabel("·σ"), 0, 3)  # ·σ

        self.residual_enabled_check = QCheckBox("Universal outlier detection — remove if residual exceeds:")
        self.residual_enabled_check.setChecked(True)
        self.residual_enabled_check.setToolTip(
            "Reject vectors whose distance from their local window median "
            "displacement (px/frame) exceeds this value.")
        self.residual_enabled_check.toggled.connect(self._on_residual_filter_toggled)
        self.residual_max_spin = style_spin(QDoubleSpinBox())
        self.residual_max_spin.setRange(0.0, 1e6)
        self.residual_max_spin.setValue(3.0)
        self.residual_max_spin.setToolTip("Max allowed distance (px/frame) from the local window median before a vector is rejected.")
        self.residual_max_spin.setEnabled(True)
        post_grid.addWidget(self.residual_enabled_check, 1, 0, 1, 2)
        post_grid.addWidget(self.residual_max_spin, 1, 2)

        self.window_size_spin = style_spin(QSpinBox())
        self.window_size_spin.setRange(3, 21)
        self.window_size_spin.setSingleStep(2)
        self.window_size_spin.setValue(3)
        self.window_size_spin.setToolTip("Size (in vectors, odd number) of the local neighborhood window used to compute each vector's local median for the residual check above.")
        self.window_size_spin.setEnabled(True)
        post_grid.addWidget(QLabel("  N (window size):"), 2, 1)
        post_grid.addWidget(self.window_size_spin, 2, 2)

        self.replace_invalid_check = QCheckBox("Interpolate removed vectors")
        self.replace_invalid_check.setToolTip(
            "Fill in gaps left by rejected vectors via interpolation from "
            "their neighbors, so the final field has no missing values. "
            "Runs once, after all rejection filters above.")
        post_grid.addWidget(self.replace_invalid_check, 3, 0, 1, 3)

        self.smooth_check = QCheckBox("Gaussian smooth")
        self.smooth_check.setToolTip(
            "Apply a Gaussian blur to the final field, once, after "
            "everything else -- purely cosmetic/noise-reduction, not part "
            "of the per-pass engine loop (see the 'Per-pass smoothn' "
            "option above for that).")
        self.smooth_sigma_spin = style_spin(QDoubleSpinBox())
        self.smooth_sigma_spin.setRange(0.1, 100.0)
        self.smooth_sigma_spin.setValue(1.0)
        self.smooth_sigma_spin.setToolTip("Standard deviation (in vectors) of the Gaussian smoothing kernel -- higher = smoother.")
        post_grid.addWidget(self.smooth_check, 4, 0)
        post_grid.addWidget(QLabel("σ:"), 4, 1)  # σ
        post_grid.addWidget(self.smooth_sigma_spin, 4, 2)

        layout.addWidget(post_box)

        # ---- performance (Tier 3: cross-pair parallelism) ----
        perf_box = QGroupBox("PERFORMANCE")
        perf_box.setToolTip(
            "Planar CPU batch runs process independent frame pairs across "
            "multiple worker processes (each pair's result is unaffected "
            "by how many workers ran the batch -- see "
            "processing.parallel_planar's module docstring). Everything "
            "else that makes the CPU pipeline faster is automatic and "
            "unconditional; this is the one knob left to the user, mainly "
            "useful to leave cores free on a shared machine.")
        perf_grid = QGridLayout(perf_box)
        perf_grid.setContentsMargins(6, 6, 6, 6)
        perf_grid.setSpacing(4)

        self.n_workers_check = QCheckBox("Limit worker processes to:")
        self.n_workers_check.setToolTip(
            "Unchecked = auto-detect from CPU count and available RAM "
            "(perf.autotune.recommended_workers()). Check to cap the "
            "number of pairs processed concurrently.")
        self.n_workers_check.toggled.connect(lambda c: self.n_workers_spin.setEnabled(c))
        self.n_workers_spin = style_spin(QSpinBox())
        self.n_workers_spin.setRange(1, 1024)
        self.n_workers_spin.setValue(max(1, recommended_workers()))
        self.n_workers_spin.setToolTip("Maximum number of frame pairs processed concurrently (planar CPU batches only).")
        self.n_workers_spin.setEnabled(False)
        perf_grid.addWidget(self.n_workers_check, 0, 0)
        perf_grid.addWidget(self.n_workers_spin, 0, 1)
        layout.addWidget(perf_box)

        layout.addStretch(1)

    def set_backend(self, backend):
        """Grey out whichever group of fields doesn't apply to the
        selected backend -- called by main_window whenever project_panel's
        CPU/GPU radio changes. Each field's own tooltip already explains
        why (see config.legacy.to_cpu_settings/to_gpu_settings for which
        canonical fields each backend actually reads); this makes that
        visible at a glance instead of only on hover.

        A checkbox-gated spin (e.g. batch_size_spin) must be enabled only
        when BOTH its backend applies AND its own checkbox is checked --
        the checkbox's existing toggled handler doesn't know about
        backend, so it's recombined here rather than replaced."""
        is_gpu = backend == "gpu"
        for w in (self.batch_size_check, self.tiling_check, self.n_tiles_y_spin,
                  self.n_tiles_x_spin, self.tile_margin_check):
            w.setEnabled(is_gpu)
        self.batch_size_spin.setEnabled(is_gpu and self.batch_size_check.isChecked())
        self.tile_margin_spin.setEnabled(is_gpu and self.tile_margin_check.isChecked())

        for w in (self.correlation_method_combo, self.deformation_method_combo,
                  self.interpolation_order_spin, self.filter_method_combo):
            w.setEnabled(not is_gpu)

    def _on_std_filter_toggled(self, checked):
        self.n_std_spin.setEnabled(checked)

    def _on_residual_filter_toggled(self, checked):
        self.residual_max_spin.setEnabled(checked)
        self.window_size_spin.setEnabled(checked)

    def get_correlation_settings(self) -> CorrelationSettings:
        return CorrelationSettings(
            passes=self.passes_table.get_passes(),
            dt=self.dt_spin.value(),
            correlation_method=self.correlation_method_combo.currentText(),
            subpixel_method=self.subpixel_combo.currentText(),
            deformation_method=self.deformation_method_combo.currentText(),
            interpolation_order=self.interpolation_order_spin.value(),
            batch_size=self.batch_size_spin.value() if self.batch_size_check.isChecked() else None,
            use_tiling=self.tiling_check.isChecked(),
            n_tiles_y=self.n_tiles_y_spin.value(),
            n_tiles_x=self.n_tiles_x_spin.value(),
            tile_margin_px=self.tile_margin_spin.value() if self.tile_margin_check.isChecked() else None,
        )

    def get_validation_settings(self) -> ValidationSettings:
        return ValidationSettings(
            filter_method=self.filter_method_combo.currentText(),
            max_filter_iteration=self.max_filter_iter_spin.value(),
            filter_kernel_size=self.filter_kernel_size_spin.value(),
            smoothn=self.smoothn_check.isChecked(),
            smoothn_p=self.smoothn_p_spin.value(),
        )

    def get_calibration_settings(self) -> CalibrationSettings:
        return CalibrationSettings(
            pixel_pitch_mm=self.pixel_pitch_spin.value() if self.pixel_pitch_check.isChecked() else None,
            frame_dt_s=self.frame_dt_spin.value() if self.frame_dt_check.isChecked() else None,
        )

    def get_performance_settings(self) -> PerformanceSettings:
        return PerformanceSettings(
            n_workers=self.n_workers_spin.value() if self.n_workers_check.isChecked() else None,
        )

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
