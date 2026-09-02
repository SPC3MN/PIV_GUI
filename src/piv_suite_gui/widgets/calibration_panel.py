"""Stereo calibration panel: per-camera DaVis polynomial coefficients,
world/dewarp geometry, and the two cameras' viewing angles used by
reconstruct_stereo. A real DaVis .set project carries its own calibration and decodes exactly
(io.davis_set.read_stereo_calibration_from_set), so the manual coefficient
form here is an ESCAPE HATCH for data that has none -- not the normal path.
It lives behind an "Advanced" disclosure for that reason.

cam0/cam1 forms are tabs rather than side-by-side, to keep the panel's
width fixed instead of doubling it. Labels use the same symbols as
calibration.camera_mapping.CameraMapping's own docstring (x0/y0 the
mapping's origin, x_span/y_span the normalization span -- displayed as
Δx/Δy since that's what a "span" is -- and the dx(s,t)/dy(s,t) polynomial
whose terms are s, s^2, s^3, t, t^2, t^3, st, s^2t, t^2s, matching
CameraMapping._poly exactly) and reconstruct_stereo's own alpha/beta
angle parameters.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHeaderView, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from piv_suite.config.schema import CameraMappingSettings, StereoSettings

from ._util import CollapsibleSection, fit_table_to_rows, style_spin

COEF_KEYS = ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")
# Display-only superscript notation for the polynomial table's row headers
# -- CameraMapping._poly's actual term for each key, e.g. s2 -> s^2. The
# underlying COEF_KEYS strings are unchanged (still what CameraMapping's
# dx_coefs/dy_coefs dicts are keyed by).
COEF_DISPLAY = {
    "1": "1", "s": "s", "s2": "s²", "s3": "s³",
    "t": "t", "t2": "t²", "t3": "t³",
    "st": "st", "s2t": "s²t", "t2s": "t²s",
}
SPIN_WIDTH = 90


class _CameraMappingForm(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._plane2 = None   # a second config.schema.CameraMappingSettings, or None
        self._z_mm = None     # this form's OWN (primary-plane) calibrated Z, or None if manual
        # This camera's real raw sensor size (DaVis Calibration.xml's
        # OriginalImageSize), same "extra state alongside the visible
        # widgets" treatment as _z_mm/_plane2 above -- there's no spin box
        # for this (nothing to hand-edit; it's a fixed physical constant
        # of the camera, only ever auto-extracted). REAL BUG this fixes:
        # get_settings() used to silently drop these on every call by
        # never reading them back at all, so CameraMapping.raw_domain_
        # valid's FOV-cropping mask was a permanent no-op the instant this
        # form's settings were read back for actual Preview/Run processing
        # -- calibration_panel.get_settings() is what preview_panel.py and
        # run_panel.py both call to build the settings a real run/preview
        # actually uses, so the auto-extracted values from set_settings()
        # never survived past the initial display. Confirmed via a real
        # GUI test: a fully-uncropped rectangular preview with a HIGHER
        # valid-vector count than raw_domain_valid's own ceiling allows,
        # which is only possible if it was returning "no masking" the
        # whole time.
        self._raw_width = None
        self._raw_height = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # DaVis auto-load summary -- hidden until set_settings() loads a
        # z_mm-tagged mapping. x0/x_span/y0/y_span/coef_table below always
        # show the PRIMARY plane's values (auto-loaded or manually typed);
        # a second plane, if any, isn't hand-editable (40 numbers), just
        # summarized here and carried alongside for interpolation at use time.
        davis_row = QHBoxLayout()
        self.davis_plane_label = QLabel()
        self.davis_plane_label.setWordWrap(True)
        self.davis_plane_label.setVisible(False)
        clear_davis_btn = QPushButton("Clear DaVis auto-load")
        clear_davis_btn.setToolTip(
            "Detach from the auto-loaded DaVis calibration (keeps the current "
            "numbers below, editable, but drops the second Z-plane and its "
            "auto-loaded tag).")
        clear_davis_btn.setVisible(False)
        clear_davis_btn.clicked.connect(self._clear_davis_autoload)
        self._clear_davis_btn = clear_davis_btn
        davis_row.addWidget(self.davis_plane_label, 1)
        davis_row.addWidget(clear_davis_btn)
        layout.addLayout(davis_row)

        top = QGridLayout()
        top.setSpacing(4)
        top.setColumnStretch(1, 1)
        self.name_edit = QLineEdit(title)
        self.name_edit.setToolTip("Label for this camera mapping -- cosmetic only, doesn't affect the calculation.")
        self.x0_spin = self._make_spin()
        self.x0_spin.setToolTip("Origin (in normalized sensor coordinates) that the dx(s,t)/dy(s,t) polynomial below is measured from, along x.")
        self.x_span_spin = self._make_spin(default=1.0)
        self.x_span_spin.setToolTip("Normalization span along x -- sensor coordinates are divided by this before being fed to the polynomial, so s stays in a well-conditioned range.")
        self.y0_spin = self._make_spin()
        self.y0_spin.setToolTip("Origin (in normalized sensor coordinates) that the dx(s,t)/dy(s,t) polynomial below is measured from, along y.")
        self.y_span_spin = self._make_spin(default=1.0)
        self.y_span_spin.setToolTip("Normalization span along y -- sensor coordinates are divided by this before being fed to the polynomial, so t stays in a well-conditioned range.")
        for i, (label, w) in enumerate([
            ("Name:", self.name_edit), ("x₀:", self.x0_spin), ("Δx:", self.x_span_spin),
            ("y₀:", self.y0_spin), ("Δy:", self.y_span_spin),
        ]):
            top.addWidget(QLabel(label), i, 0)
            top.addWidget(w, i, 1)
        layout.addLayout(top)

        self.coef_table = QTableWidget(len(COEF_KEYS), 2)
        self.coef_table.setHorizontalHeaderLabels(["dx(s,t)", "dy(s,t)"])
        self.coef_table.setToolTip(
            "Polynomial coefficients mapping normalized sensor coordinates "
            "(s, t) to world-space displacement (dx, dy), read off DaVis's "
            "own calibration report. Each row is one polynomial term "
            "(s, s², s³, t, t², t³, st, s²t, t²s); the constant '1' row "
            "is the offset term.")
        self.coef_table.setVerticalHeaderLabels([COEF_DISPLAY[k] for k in COEF_KEYS])
        self.coef_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row in range(len(COEF_KEYS)):
            self.coef_table.setItem(row, 0, QTableWidgetItem("0.00"))
            self.coef_table.setItem(row, 1, QTableWidgetItem("0.00"))
        fit_table_to_rows(self.coef_table)
        layout.addWidget(self.coef_table)

    @staticmethod
    def _make_spin(default=0.0):
        s = QDoubleSpinBox()
        s.setRange(-1e7, 1e7)
        s.setValue(default)
        return style_spin(s, width=SPIN_WIDTH)

    def _clear_davis_autoload(self):
        self._plane2 = None
        self._z_mm = None
        self._update_davis_label()

    def _update_davis_label(self):
        loaded = self._z_mm is not None or self._plane2 is not None
        self.davis_plane_label.setVisible(loaded)
        self._clear_davis_btn.setVisible(loaded)
        if not loaded:
            return
        if self._plane2 is not None:
            self.davis_plane_label.setText(
                f"Auto-loaded from DaVis -- 2 Z-planes: z={self._z_mm:.2f} mm / "
                f"z={self._plane2.z_mm:.2f} mm (Sheet Z below picks where between them)")
        else:
            self.davis_plane_label.setText(f"Auto-loaded from DaVis -- z={self._z_mm:.2f} mm")

    def get_settings(self) -> CameraMappingSettings:
        dx_coefs = {k: float(self.coef_table.item(i, 0).text()) for i, k in enumerate(COEF_KEYS)}
        dy_coefs = {k: float(self.coef_table.item(i, 1).text()) for i, k in enumerate(COEF_KEYS)}
        return CameraMappingSettings(
            x0=self.x0_spin.value(), x_span=self.x_span_spin.value(),
            y0=self.y0_spin.value(), y_span=self.y_span_spin.value(),
            dx_coefs=dx_coefs, dy_coefs=dy_coefs, name=self.name_edit.text(),
            z_mm=self._z_mm, raw_width=self._raw_width, raw_height=self._raw_height,
        )

    @property
    def plane2(self):
        return self._plane2

    def set_settings(self, settings: CameraMappingSettings, plane2=None):
        self.name_edit.setText(settings.name)
        self.x0_spin.setValue(settings.x0)
        self.x_span_spin.setValue(settings.x_span)
        self.y0_spin.setValue(settings.y0)
        self.y_span_spin.setValue(settings.y_span)
        for i, k in enumerate(COEF_KEYS):
            self.coef_table.setItem(i, 0, QTableWidgetItem(f"{settings.dx_coefs.get(k, 0.0):.2f}"))
            self.coef_table.setItem(i, 1, QTableWidgetItem(f"{settings.dy_coefs.get(k, 0.0):.2f}"))
        self._z_mm = settings.z_mm
        self._plane2 = plane2
        self._raw_width = settings.raw_width
        self._raw_height = settings.raw_height
        self._update_davis_label()


class CalibrationPanel(QWidget):
    # Emitted when the user clicks "Load stereo calibration from .set...";
    # main_window listens and re-runs extraction against the CURRENTLY
    # selected input path (this panel has no reference to it itself).
    load_from_set_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # DaVis's PinholeOpenCV calibration, when the loaded project has that
        # model instead of the polynomial one. Held opaquely and passed
        # straight back out by get_settings: it is 16 fitted camera
        # parameters per camera, not something a user hand-edits, so it has no
        # editable form here -- but it MUST survive a set_settings/
        # get_settings round trip, because every processing entry point builds
        # its config from this panel. Dropping it silently produced a config
        # with an identity polynomial mapping and no pinhole, which then
        # failed deep inside view_angles telling the user to enter angles
        # manually -- directly contradicting the status line that had just
        # said the calibration was extracted successfully.
        self._cam0_pinhole = None
        self._cam1_pinhole = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        cam_box = QGroupBox("CAMERA CALIBRATION")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setContentsMargins(4, 4, 4, 4)

        load_from_set_btn = QPushButton("Load stereo calibration from .set...")
        load_from_set_btn.setToolTip(
            "Re-extract calibration from the currently selected .set project's "
            "own calibration-target mark data (same as happens automatically "
            "when the input path is selected).")
        load_from_set_btn.clicked.connect(self.load_from_set_requested.emit)
        cam_layout.addWidget(load_from_set_btn)

        # Which of DaVis's two calibration models this project actually has.
        # The coefficient forms below are meaningful only for the polynomial
        # one; for a PinholeOpenCV project they stay at their defaults and are
        # not used, so say so rather than showing the user an editable table
        # of zeros that looks like it matters.
        self.model_label = QLabel("Model: polynomial — decoded from the project, editable under Advanced")
        self.model_label.setWordWrap(True)
        cam_layout.addWidget(self.model_label)

        # A calibration that cannot be read is a BLOCKING condition -- the
        # project cannot be processed at all -- so it gets a persistent,
        # readable home here, next to the controls that would fix it. It used
        # to go only to the status bar, which truncated an 832-character
        # explanation to whatever fitted and then cleared it after 8 seconds.
        self.problem_label = QLabel()
        self.problem_label.setObjectName("problemLabel")
        self.problem_label.setWordWrap(True)
        self.problem_label.setVisible(False)
        cam_layout.addWidget(self.problem_label)

        # Everything below is the ESCAPE HATCH, not the normal path. A real
        # DaVis .set decodes exactly on its own (see this module's docstring),
        # so the 40-odd coefficient fields only matter for data that has no
        # DaVis calibration -- and presenting them by default made a rarely
        # used fallback the most prominent thing on the panel.
        # A PEER of the camera card, not a child of it. The drawer holds the
        # world grid and the viewing-angle overrides as well as the coefficient
        # forms, and neither of those is a camera mapping -- nesting them made
        # the CAMERA CALIBRATION card's border run down the whole panel around
        # sections that are not camera calibration.
        self.advanced = CollapsibleSection("Advanced — calibration")

        cam_tabs = QTabWidget()
        self.cam0_form = _CameraMappingForm("cam0")
        self.cam1_form = _CameraMappingForm("cam1")
        cam_tabs.addTab(self.cam0_form, "cam0")
        cam_tabs.addTab(self.cam1_form, "cam1")
        layout.addWidget(cam_box)
        layout.addWidget(self.advanced)
        self.advanced.add_widget(cam_tabs)

        geom_box = QGroupBox("WORLD GRID / DEWARP")
        geom_grid = QGridLayout(geom_box)
        geom_grid.setContentsMargins(6, 6, 6, 6)
        geom_grid.setSpacing(4)
        self.world_h_spin = style_spin(QSpinBox(), width=SPIN_WIDTH); self.world_h_spin.setRange(1, 100000); self.world_h_spin.setValue(1000)
        self.world_h_spin.setToolTip("Height (pixels) of the shared world grid both cameras' frames are dewarped onto before correlation.")
        self.world_w_spin = style_spin(QSpinBox(), width=SPIN_WIDTH); self.world_w_spin.setRange(1, 100000); self.world_w_spin.setValue(1000)
        self.world_w_spin.setToolTip("Width (pixels) of the shared world grid both cameras' frames are dewarped onto before correlation.")
        self.world_scale_spin = style_spin(QDoubleSpinBox(), width=SPIN_WIDTH); self.world_scale_spin.setRange(1e-6, 1e6); self.world_scale_spin.setValue(1.0)
        self.world_scale_spin.setToolTip("World-grid scale in pixels per mm -- used to convert the reconstructed in-plane displacement into physical units before applying Frame Δt.")
        self.dewarp_order_spin = style_spin(QSpinBox(), width=SPIN_WIDTH); self.dewarp_order_spin.setRange(0, 5); self.dewarp_order_spin.setValue(1)
        self.dewarp_order_spin.setToolTip("Interpolation order used when resampling each camera's raw frame onto the world grid (0 = nearest, 1 = linear, higher = smoother but slower).")
        self.sheet_z_mm_check = QCheckBox("Laser sheet Z (mm):")
        self.sheet_z_mm_check.setToolTip(
            "Real Z position of the laser sheet for THIS recording -- required "
            "when a camera has two DaVis-calibrated Z-planes (see the camera "
            "tabs above), used to interpolate between them. Not derivable from "
            "any calibration file -- it's an acquisition-time choice, always "
            "manual.")
        self.sheet_z_mm_check.toggled.connect(lambda c: self.sheet_z_mm_spin.setEnabled(c))
        self.sheet_z_mm_spin = style_spin(QDoubleSpinBox(), width=SPIN_WIDTH)
        self.sheet_z_mm_spin.setRange(-1e6, 1e6)
        # -0.5 matches the Swirl project's own validated recording (real
        # calibrated range -2.0mm to +1.0mm, sheet at the midpoint) -- like
        # alpha1/alpha2's placeholder above, this is a per-recording
        # acquisition-time value with NO calibration-file-derivable default
        # (build_camera_mapping raises rather than guessing), so treat this
        # as "what was last validated," not a generally-correct number.
        self.sheet_z_mm_spin.setValue(-0.5)
        self.sheet_z_mm_spin.setEnabled(False)
        geom_grid.addWidget(QLabel("World shape (H, W):"), 0, 0)
        geom_grid.addWidget(self.world_h_spin, 0, 1)
        geom_grid.addWidget(self.world_w_spin, 0, 2)
        geom_grid.addWidget(QLabel("World scale (px/mm):"), 1, 0)
        geom_grid.addWidget(self.world_scale_spin, 1, 1)
        geom_grid.addWidget(QLabel("Dewarp order:"), 2, 0)
        geom_grid.addWidget(self.dewarp_order_spin, 2, 1)
        geom_grid.addWidget(self.sheet_z_mm_check, 3, 0)
        geom_grid.addWidget(self.sheet_z_mm_spin, 3, 1)
        self.advanced.add_widget(geom_box)

        angle_box = QGroupBox("STEREO VIEWING ANGLES (DEG)")
        angle_box.setToolTip(
            "See the original Stereo_PIV_GPU/Stereo_PIV_CPU README: these "
            "angles are placeholders that should be verified per Z-plane "
            "before trusting the reconstructed W (or U/V)."
        )
        angle_grid = QGridLayout(angle_box)
        angle_grid.setContentsMargins(6, 6, 6, 6)
        angle_grid.setSpacing(4)
        # alpha1/alpha2 (the primary triangulation angle) are REQUIRED, no
        # usable auto-derive -- confirmed measurably wrong against a real
        # rig (see StereoSettings.alpha1_deg's own comment for the full
        # story: a 2-plane linear-parallax model that can't handle a real
        # calibration's higher-order/finite-standoff terms). Gated behind
        # this checkbox (unchecked = not yet entered = None, mirroring
        # sheet_z_mm_check's own pattern just above) rather than shown as
        # some numeric placeholder that looks like a real answer.
        self.alpha_override_check = QCheckBox("Override derived angles:")
        self.alpha_override_check.setToolTip(
            "Leave UNCHECKED (the default). The stereo triangulation angles are derived "
            "PER PIXEL from this project's own calibration, exactly, at every correlation "
            "point -- calibration.camera_mapping.stereo_view_angles. That is strictly better "
            "than any single number you can type here: the real viewing angle varies several "
            "degrees across a stereo field of view (8.4°/8.8° on the reference rig), "
            "and collapsing it to one value per camera puts ~13% of W into U at the field "
            "edges and ~7% into V at top/bottom.\n\n"
            "Check this ONLY for a calibration that genuinely cannot supply the geometry -- "
            "a single calibrated Z-plane, or a manual/marks-fit mapping. Then enter a real "
            "measured value, e.g. DaVis's own Calibration report's \"Min/Max angle 1-2\", "
            "split symmetrically.")
        self.alpha_override_check.toggled.connect(
            lambda c: [w.setEnabled(c) for w in (self.alpha1_spin, self.alpha2_spin,
                                                 self.beta1_spin, self.beta2_spin)])
        self.alpha1_spin = self._angle_spin(44.765)
        self.alpha1_spin.setToolTip("Camera 0's in-plane viewing angle (deg) relative to the world Z-axis. See the checkbox's tooltip above.")
        self.alpha1_spin.setEnabled(False)
        self.alpha2_spin = self._angle_spin(-44.765)
        self.alpha2_spin.setToolTip("Camera 1's in-plane viewing angle (deg) relative to the world Z-axis. See the checkbox's tooltip above.")
        self.alpha2_spin.setEnabled(False)
        self.beta1_spin = self._angle_spin(0.0)
        self.beta1_spin.setToolTip("Camera 0's out-of-plane viewing angle (deg). Derived per pixel from the calibration unless the override above is checked.")
        self.beta1_spin.setEnabled(False)
        self.beta2_spin = self._angle_spin(0.0)
        self.beta2_spin.setToolTip("Camera 1's out-of-plane viewing angle (deg). See β₁'s tooltip.")
        self.beta2_spin.setEnabled(False)
        angle_grid.addWidget(self.alpha_override_check, 0, 0)
        for i, (label, w) in enumerate([
            ("α₁:", self.alpha1_spin), ("α₂:", self.alpha2_spin),
        ], start=1):
            angle_grid.addWidget(QLabel(label), i, 0)
            angle_grid.addWidget(w, i, 1)
        for i, (label, w) in enumerate([
            ("β₁:", self.beta1_spin), ("β₂:", self.beta2_spin),
        ], start=3):
            angle_grid.addWidget(QLabel(label), i, 0)
            angle_grid.addWidget(w, i, 1)
        self.advanced.add_widget(angle_box)
        layout.addStretch(1)

    @staticmethod
    def _angle_spin(default):
        s = QDoubleSpinBox()
        s.setRange(-180.0, 180.0)
        s.setValue(default)
        return style_spin(s, width=SPIN_WIDTH)

    def set_problem(self, text):
        """Show a blocking calibration problem, or clear it with None/"".

        Opens the Advanced drawer alongside it. Every one of these messages
        ends by telling the user to "enter calibration manually on the
        Calibration panel" -- and manual entry is exactly what the drawer
        hides, so leaving it shut points the user at a control they cannot
        see. A remedy the reader cannot reach is not a remedy."""
        self.problem_label.setText(text or "")
        self.problem_label.setVisible(bool(text))
        self.model_label.setVisible(not text)
        if text:
            self.advanced.set_expanded(True)

    def get_settings(self) -> StereoSettings:
        return StereoSettings(
            cam0_pinhole=self._cam0_pinhole,
            cam1_pinhole=self._cam1_pinhole,
            cam0_mapping=self.cam0_form.get_settings(),
            cam0_mapping_plane2=self.cam0_form.plane2,
            cam1_mapping=self.cam1_form.get_settings(),
            cam1_mapping_plane2=self.cam1_form.plane2,
            world_shape=(self.world_h_spin.value(), self.world_w_spin.value()),
            world_scale_px_per_mm=self.world_scale_spin.value(),
            dewarp_order=self.dewarp_order_spin.value(),
            # Unchecked -> all four None -> derived per pixel downstream.
            alpha1_deg=self.alpha1_spin.value() if self.alpha_override_check.isChecked() else None,
            alpha2_deg=self.alpha2_spin.value() if self.alpha_override_check.isChecked() else None,
            beta1_deg=self.beta1_spin.value() if self.alpha_override_check.isChecked() else None,
            beta2_deg=self.beta2_spin.value() if self.alpha_override_check.isChecked() else None,
            sheet_z_mm=self.sheet_z_mm_spin.value() if self.sheet_z_mm_check.isChecked() else None,
        )

    def set_settings(self, settings: StereoSettings):
        """Push extracted/auto-loaded stereo calibration into the form,
        authoritatively overwriting whatever was there before -- mirrors
        settings_panel.set_calibration_settings' pattern for the planar
        case."""
        self._cam0_pinhole = settings.cam0_pinhole
        self._cam1_pinhole = settings.cam1_pinhole
        self.model_label.setText(
            "Model: DaVis PinholeOpenCV — exact, angles derived per pixel"
            if settings.cam0_pinhole is not None
            else "Model: polynomial — decoded from the project, editable under Advanced")
        self.cam0_form.set_settings(settings.cam0_mapping, settings.cam0_mapping_plane2)
        self.cam1_form.set_settings(settings.cam1_mapping, settings.cam1_mapping_plane2)
        if settings.world_shape and settings.world_shape != (0, 0):
            self.world_h_spin.setValue(settings.world_shape[0])
            self.world_w_spin.setValue(settings.world_shape[1])
        self.world_scale_spin.setValue(settings.world_scale_px_per_mm)
        # All four angles are an OPTIONAL OVERRIDE now. A freshly-extracted
        # calibration leaves them None (derived per pixel downstream), so the
        # override box goes unchecked; a project saved with real measured
        # values re-loads them and stays checked.
        has_override = settings.alpha1_deg is not None and settings.alpha2_deg is not None
        self.alpha_override_check.setChecked(has_override)
        if has_override:
            self.alpha1_spin.setValue(settings.alpha1_deg)
            self.alpha2_spin.setValue(settings.alpha2_deg)
            if settings.beta1_deg is not None:
                self.beta1_spin.setValue(settings.beta1_deg)
            if settings.beta2_deg is not None:
                self.beta2_spin.setValue(settings.beta2_deg)
        self.sheet_z_mm_check.setChecked(settings.sheet_z_mm is not None)
        if settings.sheet_z_mm is not None:
            self.sheet_z_mm_spin.setValue(settings.sheet_z_mm)
