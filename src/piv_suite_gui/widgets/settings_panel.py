"""Settings panel: raw-frame pre-processing, multi-pass window/overlap
schedule, per-pass internal stability fill, the "remove invalid vectors"
post-processing step, and (behind Advanced -- see advanced_widget()) the
correlation algorithm pickers, GPU tiling, and worker-process limit.

Physical-unit calibration (pixel pitch / frame Δt) lives on ProjectPanel
now, right below where the project is selected -- not here. It used to be
a checkbox-gated pair of fields in a "PHYSICAL UNITS" box on this panel,
but every real workflow either auto-extracts both from the selected .set
or needs them filled in before anything downstream is meaningful, so
gating them behind an on/off toggle only added a click nobody needed.

Pre-processing (the min/max filter) used to live on ProjectPanel, folded
into the SOURCE card right after the input path -- moved here, right
above the window schedule, because it is a processing-pipeline choice
like everything else on this panel, not a question about where the data
comes from.

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
    CorrelationSettings, PassSettings, PerformanceSettings, PostProcessSettings,
    PreprocessSettings, RangeFilterSettings, ValidationSettings,
)
from piv_suite.perf.autotune import recommended_workers

from ._util import fit_table_to_rows, style_spin

#: The only correlation method this app supports -- see where it is displayed.
CORRELATION_METHOD = "circular"


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

        # ---- pre-processing (applied to raw frames, before correlation --
        # for stereo, before dewarping too) -- right above the window
        # schedule, the first thing that actually happens to the data. ----
        prep_box = QGroupBox("PRE-PROCESSING")
        prep_layout = QHBoxLayout(prep_box)
        prep_layout.setContentsMargins(6, 6, 6, 6)
        prep_layout.setSpacing(6)
        self.min_max_check = QCheckBox("Min/max filter (L px):")
        self.min_max_check.setToolTip(
            "LaVision-style sliding min/max background removal + local "
            "contrast normalization, applied to each raw frame before "
            "correlation. For stereo, applied per-camera BEFORE "
            "dewarping. On by default: on real DaVis recordings this is "
            "the single largest contributor to matching DaVis's own "
            "vectors -- see config.schema.PreprocessSettings.")
        # Checked to match PreprocessSettings' own default (see its
        # docstring for the real-data measurements behind that default).
        self.min_max_check.setChecked(True)
        self.min_max_length_spin = style_spin(QSpinBox())
        self.min_max_length_spin.setRange(1, 10000)
        self.min_max_length_spin.setValue(5)
        self.min_max_length_spin.setToolTip(
            "L, in pixels -- the sliding window size for the min/max "
            "filter's background-removal and local-contrast steps. "
            "Barely sensitive: 4 and 5 agree with DaVis equally well.")
        self.min_max_check.toggled.connect(self.min_max_length_spin.setEnabled)
        prep_layout.addWidget(self.min_max_check)
        prep_layout.addWidget(self.min_max_length_spin)
        prep_layout.addStretch(1)
        layout.addWidget(prep_box)

        self.passes_table = _PassesTable()
        layout.addWidget(self.passes_table)

        # ---- algorithm (Advanced) -- method pickers and GPU tiling are
        # expert territory: correct defaults already, wrong answers
        # available, and nothing a normal run needs to touch. ----
        algo_box = QGroupBox("ALGORITHM")
        adv_grid = QGridLayout(algo_box)
        adv_grid.setContentsMargins(6, 6, 6, 6)
        adv_grid.setSpacing(4)
        adv_grid.setColumnStretch(1, 1)

        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["gaussian", "centroid", "parabolic"])
        self.subpixel_combo.setToolTip(
            "Peak-fitting method used to refine each correlation peak to "
            "sub-pixel accuracy. 'gaussian' is the standard choice for "
            "typical PIV particle images.")
        adv_grid.addWidget(QLabel("Subpixel method:"), 0, 0)
        adv_grid.addWidget(self.subpixel_combo, 0, 1)

        # A LABEL, not a one-item combo. Zero-padded ("linear") correlation was
        # removed because it needs a normalization this app never applies and
        # measured 4.665 px RMS at 14 px against circular's 0.059 -- which left
        # a dropdown that could never change. An enabled control that cannot do
        # anything is its own kind of dead control; stating the value is
        # honest, and keeps it discoverable.
        self.correlation_method_value = QLabel(CORRELATION_METHOD)
        self.correlation_method_value.setToolTip(
            "Circular (unpadded) FFT cross-correlation. Not a choice: openpiv's "
            "zero-padded alternative needs a normalization this app does not "
            "apply, and measures far worse at every displacement.")
        adv_grid.addWidget(QLabel("Correlation method:"), 1, 0)
        adv_grid.addWidget(self.correlation_method_value, 1, 1)

        self.deformation_method_combo = QComboBox()
        self.deformation_method_combo.addItems(["symmetric", "second image"])
        self.deformation_method_combo.setToolTip("CPU backend only -- ignored on GPU.")
        adv_grid.addWidget(QLabel("Deformation method:"), 2, 0)
        adv_grid.addWidget(self.deformation_method_combo, 2, 1)

        self.interpolation_order_spin = style_spin(QSpinBox())
        self.interpolation_order_spin.setRange(0, 5)
        self.interpolation_order_spin.setValue(3)
        self.interpolation_order_spin.setToolTip(
            "CPU backend only -- ignored on GPU. 3 measured best on real "
            "data: order 5 reduces sub-pixel bias only below the noise "
            "floor of agreement with DaVis (22.18 vs 22.21 mm/s mean|diff|) "
            "while costing 14-21% more runtime; order 1 (bilinear) is a "
            "real regression (0.0412 px worst-case bias vs 0.0145 at "
            "order 3). See CorrelationSettings.interpolation_order's own "
            "comment for the full sweep.")
        adv_grid.addWidget(QLabel("Interpolation order:"), 3, 0)
        adv_grid.addWidget(self.interpolation_order_spin, 3, 1)

        self.batch_size_check = QCheckBox("Set GPU batch size:")
        self.batch_size_check.setToolTip("GPU backend only. Unchecked = piv_gpu's own default (process all windows in one batch).")
        self.batch_size_check.toggled.connect(lambda c: self.batch_size_spin.setEnabled(c))
        self.batch_size_spin = style_spin(QSpinBox())
        self.batch_size_spin.setRange(1, 1_000_000)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setToolTip("Number of interrogation windows processed per GPU batch.")
        self.batch_size_spin.setEnabled(False)
        adv_grid.addWidget(self.batch_size_check, 4, 0)
        adv_grid.addWidget(self.batch_size_spin, 4, 1)

        self.tiling_check = QCheckBox("GPU tiling (large frames)")
        self.tiling_check.setToolTip("Split large frames into a grid of tiles to bound peak GPU memory. GPU backend only.")
        self.n_tiles_y_spin = style_spin(QSpinBox()); self.n_tiles_y_spin.setRange(1, 64); self.n_tiles_y_spin.setValue(1)
        self.n_tiles_y_spin.setToolTip("Number of tiles to split each frame into along the y (row) axis. GPU backend only.")
        self.n_tiles_x_spin = style_spin(QSpinBox()); self.n_tiles_x_spin.setRange(1, 64); self.n_tiles_x_spin.setValue(1)
        self.n_tiles_x_spin.setToolTip("Number of tiles to split each frame into along the x (column) axis. GPU backend only.")
        adv_grid.addWidget(self.tiling_check, 5, 0, 1, 2)
        adv_grid.addWidget(QLabel("n_tiles_y:"), 6, 0)
        adv_grid.addWidget(self.n_tiles_y_spin, 6, 1)
        adv_grid.addWidget(QLabel("n_tiles_x:"), 7, 0)
        adv_grid.addWidget(self.n_tiles_x_spin, 7, 1)

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
        adv_grid.addWidget(self.tile_margin_check, 8, 0)
        adv_grid.addWidget(self.tile_margin_spin, 8, 1)

        # ---- per-pass internal stability fill (Advanced) -- runs inside
        # the multi-pass loop -- does NOT reject/validate vectors, see
        # tooltip ----
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
            "Applies on both backends. ON by default: confirmed via real "
            "DaVis-dataset comparison to raise U/V correlation from "
            "~0.95 to ~0.97 and density from ~95% to ~99% (see "
            "ValidationSettings.smoothn's docstring).")
        self.smoothn_check.setChecked(True)
        self.smoothn_check.toggled.connect(lambda c: self.smoothn_p_spin.setEnabled(c))
        self.smoothn_p_spin = style_spin(QDoubleSpinBox(), decimals=4)
        self.smoothn_p_spin.setRange(0.0, 100.0)
        # 0.75, NOT 15.0 -- see ValidationSettings.smoothn_p's own comment
        # for the ground-truth re-measurement (against known displacement,
        # not DaVis's own post-denoised output) that replaced 15.0 with
        # this. 15.0 measured 1.5x-4.9x WORSE RMS error on every field
        # with real spatial structure.
        self.smoothn_p_spin.setValue(0.75)
        self.smoothn_p_spin.setToolTip("smoothn's own smoothing strength parameter -- higher = smoother.")
        self.smoothn_p_spin.setEnabled(True)
        val_grid.addWidget(self.smoothn_check, 3, 0)
        val_grid.addWidget(self.smoothn_p_spin, 3, 1)

        # ---- remove invalid vectors (the SOLE validation step) ----
        post_box = QGroupBox("REMOVE INVALID VECTORS (POST-PROCESSING)")
        post_box.setToolTip(
            "The only place vectors are ever marked invalid -- runs once, "
            "after the final field is produced. Nothing during "
            "calculation itself rejects a vector (see 'Per-pass stability "
            "(internal)' under Advanced).")
        post_grid = QGridLayout(post_box)
        post_grid.setContentsMargins(6, 6, 6, 6)
        post_grid.setSpacing(4)

        self.std_filter_check = QCheckBox("Remove if |value - mean| exceeds:")
        self.std_filter_check.setChecked(True)
        self.std_filter_check.setToolTip(
            "Reject vectors more than n·σ (n times the standard deviation) "
            "from the FIELD-WIDE mean. For planar/per-camera data this "
            "rarely fires (a bad 2D correlation still returns some "
            "plausible small displacement) -- but for stereo, this now "
            "runs on the COMBINED/triangulated vector, where a small "
            "per-camera disagreement can amplify into a genuinely extreme "
            "value; confirmed on real data to catch garbage a local "
            "universal-outlier-detection check alone misses (see "
            "config.schema.PostProcessSettings' own docstring).")
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
            "Reject vectors whose deviation from their local window "
            "median, divided by that neighbourhood's median absolute "
            "deviation, exceeds this value (Westerweel & Scarano "
            "universal outlier detection). The threshold is a "
            "DIMENSIONLESS RATIO, not a pixel distance -- LaVision "
            "DaVis's equivalent 'removal factor' default is 2.")
        self.residual_enabled_check.toggled.connect(self._on_residual_filter_toggled)
        self.residual_max_spin = style_spin(QDoubleSpinBox())
        self.residual_max_spin.setRange(0.0, 1e6)
        self.residual_max_spin.setValue(2.0)
        self.residual_max_spin.setToolTip(
            "Normalized-residual threshold (a ratio, not pixels): how many "
            "times the local median absolute deviation a vector may differ "
            "from its neighbours' median before being rejected. 2 matches "
            "DaVis's own removal factor. NOTE: this used to be an absolute "
            "px/frame distance -- a value carried over from an older "
            "project file means something different now.")
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
        self.replace_invalid_check.setChecked(True)
        self.replace_invalid_check.setToolTip(
            "Fill in gaps left by rejected vectors via interpolation from "
            "their neighbors, so the final field has no missing values. "
            "Runs once, after all rejection filters above. ON by default "
            "to keep vector density comparable to DaVis's (~98% on the "
            "reference dataset). A filled vector is INTERPOLATED, not "
            "measured -- switch this off if your analysis needs strictly "
            "measured vectors (the saved `valid` mask marks which is "
            "which either way).")
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

        # ---- performance (Advanced -- Tier 3: cross-pair parallelism) ----
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

        # ONE advanced container, not three separate disclosures -- see
        # main_window._build_ui, which embeds this (alongside
        # CalibrationPanel.advanced_widget()) into the single Advanced
        # section at the bottom of the left rail. Built here, not added
        # to `layout` directly: this panel's own visible surface is just
        # the window schedule and the post-processing box above.
        self._advanced_container = QWidget()
        adv_layout = QVBoxLayout(self._advanced_container)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(6)
        adv_layout.addWidget(algo_box)
        adv_layout.addWidget(val_box)
        adv_layout.addWidget(perf_box)

        layout.addStretch(1)

    def advanced_widget(self) -> QWidget:
        """The algorithm/per-pass-internals/performance controls, as one
        widget for main_window to embed in the single consolidated
        Advanced disclosure -- see this panel's own _build_ui comment."""
        return self._advanced_container

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

        for w in (self.deformation_method_combo,
                  self.interpolation_order_spin, self.filter_method_combo):
            w.setEnabled(not is_gpu)

    def _on_std_filter_toggled(self, checked):
        self.n_std_spin.setEnabled(checked)

    def _on_residual_filter_toggled(self, checked):
        self.residual_max_spin.setEnabled(checked)
        self.window_size_spin.setEnabled(checked)

    def get_preprocess_settings(self) -> PreprocessSettings:
        return PreprocessSettings(
            min_max_filter_enabled=self.min_max_check.isChecked(),
            min_max_filter_length=self.min_max_length_spin.value(),
        )

    def get_correlation_settings(self) -> CorrelationSettings:
        return CorrelationSettings(
            passes=self.passes_table.get_passes(),
            # Not a GUI control. openpiv's own PIVSettings.dt convention
            # divides the raw pixel displacement by this before anything
            # else sees it, producing px/dt instead of px/frame -- but
            # this app's physical-unit conversion runs entirely through a
            # SEPARATE, dedicated path instead (ProjectPanel's Pixel
            # pitch/Frame Δt -> processing.postprocess.apply_calibration),
            # so nothing in this codebase, any test, or any real .set
            # workflow ever sets dt to anything but 1.0 -- confirmed by
            # searching every real call site. A control with no scenario
            # that changes it is not a control, just a place to
            # accidentally type the wrong number.
            dt=1.0,
            correlation_method=CORRELATION_METHOD,
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
