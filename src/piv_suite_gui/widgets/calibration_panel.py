"""Stereo calibration panel: per-camera DaVis polynomial coefficients,
world/dewarp geometry, and the two cameras' viewing angles used by
reconstruct_stereo. Manual coefficient entry (structured form fields
instead of hand-edited JSON) is the near-term workflow -- the "Load from
DaVis report..." button is wired to calibration.report_parser's stub
interface but stays disabled until that parser is implemented (see
calibration/report_parser.py).

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
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from piv_suite.calibration.report_parser import parse_davis_calibration_report
from piv_suite.config.schema import CameraMappingSettings, StereoSettings

from ._util import fit_table_to_rows, style_spin

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

        load_btn = QPushButton("Load from DaVis report...")
        load_btn.setEnabled(False)
        load_btn.setToolTip(
            "Not implemented yet -- calibration.report_parser is a stub. "
            "Enter coefficients manually below, read off DaVis's own "
            "calibration report panel."
        )
        load_btn.clicked.connect(self._load_from_report)
        layout.addWidget(load_btn)

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

    def _load_from_report(self):
        try:
            parse_davis_calibration_report("")
        except NotImplementedError as e:
            QMessageBox.information(self, "Not implemented", str(e))

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
            z_mm=self._z_mm,
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
        self._update_davis_label()


class CalibrationPanel(QWidget):
    # Emitted when the user clicks "Load stereo calibration from .set...";
    # main_window listens and re-runs extraction against the CURRENTLY
    # selected input path (this panel has no reference to it itself).
    load_from_set_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
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

        cam_tabs = QTabWidget()
        self.cam0_form = _CameraMappingForm("cam0")
        self.cam1_form = _CameraMappingForm("cam1")
        cam_tabs.addTab(self.cam0_form, "cam0")
        cam_tabs.addTab(self.cam1_form, "cam1")
        cam_layout.addWidget(cam_tabs)
        layout.addWidget(cam_box)

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
        layout.addWidget(geom_box)

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
        # some numeric placeholder that looks like a real answer -- get_
        # settings()/set_settings() below refuse to silently invent a
        # value, and every processing entry point refuses to run while
        # unchecked.
        self.alpha_measured_check = QCheckBox("Angles measured (required):")
        self.alpha_measured_check.setToolTip(
            "Real stereo triangulation angle for camera 0/1 (deg), used by reconstruct_stereo "
            "to solve for U/V/W. NOT auto-derived from calibration data -- io.davis_set."
            "_estimate_stereo_angles' 2-plane linear-parallax model was found to be measurably "
            "wrong for a real (finite-standoff) camera rig, and there's no general fix derivable "
            "from a calibration file alone. Enter a real measured value instead -- e.g. DaVis's "
            "own Calibration report's \"Min/Max angle 1-2\", split symmetrically (α₁=+half, "
            "α₂=-half), or your own rig measurement. Processing refuses to run until this is "
            "checked.")
        self.alpha_measured_check.toggled.connect(
            lambda c: (self.alpha1_spin.setEnabled(c), self.alpha2_spin.setEnabled(c)))
        self.alpha1_spin = self._angle_spin(44.765)
        self.alpha1_spin.setToolTip("Camera 0's in-plane viewing angle (deg) relative to the world Z-axis. See the checkbox's tooltip above.")
        self.alpha1_spin.setEnabled(False)
        self.alpha2_spin = self._angle_spin(-44.765)
        self.alpha2_spin.setToolTip("Camera 1's in-plane viewing angle (deg) relative to the world Z-axis. See the checkbox's tooltip above.")
        self.alpha2_spin.setEnabled(False)
        self.beta1_spin = self._angle_spin(0.0)
        self.beta1_spin.setToolTip("Camera 0's out-of-plane viewing angle (deg), used by reconstruct_stereo to solve for U/V/W. Auto-derived from the calibration's own two Z-planes (io.davis_set._estimate_stereo_angles) -- edit freely if you know the real angle.")
        self.beta2_spin = self._angle_spin(0.0)
        self.beta2_spin.setToolTip("Camera 1's out-of-plane viewing angle (deg), used by reconstruct_stereo to solve for U/V/W. Auto-derived -- see β₁'s tooltip.")
        angle_grid.addWidget(self.alpha_measured_check, 0, 0)
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
        layout.addWidget(angle_box)
        layout.addStretch(1)

    @staticmethod
    def _angle_spin(default):
        s = QDoubleSpinBox()
        s.setRange(-180.0, 180.0)
        s.setValue(default)
        return style_spin(s, width=SPIN_WIDTH)

    def get_settings(self) -> StereoSettings:
        return StereoSettings(
            cam0_mapping=self.cam0_form.get_settings(),
            cam0_mapping_plane2=self.cam0_form.plane2,
            cam1_mapping=self.cam1_form.get_settings(),
            cam1_mapping_plane2=self.cam1_form.plane2,
            world_shape=(self.world_h_spin.value(), self.world_w_spin.value()),
            world_scale_px_per_mm=self.world_scale_spin.value(),
            dewarp_order=self.dewarp_order_spin.value(),
            alpha1_deg=self.alpha1_spin.value() if self.alpha_measured_check.isChecked() else None,
            alpha2_deg=self.alpha2_spin.value() if self.alpha_measured_check.isChecked() else None,
            beta1_deg=self.beta1_spin.value(), beta2_deg=self.beta2_spin.value(),
            sheet_z_mm=self.sheet_z_mm_spin.value() if self.sheet_z_mm_check.isChecked() else None,
        )

    def set_settings(self, settings: StereoSettings):
        """Push extracted/auto-loaded stereo calibration into the form,
        authoritatively overwriting whatever was there before -- mirrors
        settings_panel.set_calibration_settings' pattern for the planar
        case."""
        self.cam0_form.set_settings(settings.cam0_mapping, settings.cam0_mapping_plane2)
        self.cam1_form.set_settings(settings.cam1_mapping, settings.cam1_mapping_plane2)
        if settings.world_shape and settings.world_shape != (0, 0):
            self.world_h_spin.setValue(settings.world_shape[0])
            self.world_w_spin.setValue(settings.world_shape[1])
        self.world_scale_spin.setValue(settings.world_scale_px_per_mm)
        # beta1/beta2: davis_set.read_stereo_calibration_from_set derives
        # these from the real calibration mapping itself (see its own
        # _estimate_stereo_angles docstring) -- still just a starting
        # point, same "always overwrite, user can edit afterward"
        # convention as every other auto-extracted field here, not locked/
        # read-only.
        self.beta1_spin.setValue(settings.beta1_deg)
        self.beta2_spin.setValue(settings.beta2_deg)
        # alpha1/alpha2: REQUIRED, no auto-derive any more (see the
        # checkbox's own tooltip) -- checked/populated only if this
        # settings object already carries a real measured value (e.g.
        # loaded from a previously-saved project); otherwise left
        # unchecked/disabled, same "not yet entered" state a brand new
        # project starts in.
        has_alpha = settings.alpha1_deg is not None and settings.alpha2_deg is not None
        self.alpha_measured_check.setChecked(has_alpha)
        if has_alpha:
            self.alpha1_spin.setValue(settings.alpha1_deg)
            self.alpha2_spin.setValue(settings.alpha2_deg)
        self.sheet_z_mm_check.setChecked(settings.sheet_z_mm is not None)
        if settings.sheet_z_mm is not None:
            self.sheet_z_mm_spin.setValue(settings.sheet_z_mm)
