"""Stereo calibration panel: per-camera DaVis polynomial coefficients,
world/dewarp geometry, and the two cameras' viewing angles used by
reconstruct_stereo. Manual coefficient entry (structured form fields
instead of hand-edited JSON) is the near-term workflow -- the "Load from
DaVis report..." button is wired to calibration.report_parser's stub
interface but stays disabled until that parser is implemented (see
calibration/report_parser.py).

cam0/cam1 forms are tabs rather than side-by-side, to keep the panel's
width fixed instead of doubling it.
"""

from PySide6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from piv_suite.calibration.report_parser import parse_davis_calibration_report
from piv_suite.config.schema import CameraMappingSettings, StereoSettings

from ._util import fit_table_to_rows

COEF_KEYS = ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")
SPIN_WIDTH = 90
DECIMALS = 2


class _CameraMappingForm(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QGridLayout()
        top.setSpacing(4)
        top.setColumnStretch(1, 1)
        self.name_edit = QLineEdit(title)
        self.x0_spin = self._make_spin()
        self.x_span_spin = self._make_spin(default=1.0)
        self.y0_spin = self._make_spin()
        self.y_span_spin = self._make_spin(default=1.0)
        for i, (label, w) in enumerate([
            ("Name:", self.name_edit), ("x0:", self.x0_spin), ("x_span:", self.x_span_spin),
            ("y0:", self.y0_spin), ("y_span:", self.y_span_spin),
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
        self.coef_table.setHorizontalHeaderLabels(["dx_coefs", "dy_coefs"])
        self.coef_table.setVerticalHeaderLabels(list(COEF_KEYS))
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
        s.setDecimals(DECIMALS)
        s.setValue(default)
        s.setMaximumWidth(SPIN_WIDTH)
        return s

    def _load_from_report(self):
        try:
            parse_davis_calibration_report("")
        except NotImplementedError as e:
            QMessageBox.information(self, "Not implemented", str(e))

    def get_settings(self) -> CameraMappingSettings:
        dx_coefs = {k: float(self.coef_table.item(i, 0).text()) for i, k in enumerate(COEF_KEYS)}
        dy_coefs = {k: float(self.coef_table.item(i, 1).text()) for i, k in enumerate(COEF_KEYS)}
        return CameraMappingSettings(
            x0=self.x0_spin.value(), x_span=self.x_span_spin.value(),
            y0=self.y0_spin.value(), y_span=self.y_span_spin.value(),
            dx_coefs=dx_coefs, dy_coefs=dy_coefs, name=self.name_edit.text(),
        )

    def set_settings(self, settings: CameraMappingSettings):
        self.name_edit.setText(settings.name)
        self.x0_spin.setValue(settings.x0)
        self.x_span_spin.setValue(settings.x_span)
        self.y0_spin.setValue(settings.y0)
        self.y_span_spin.setValue(settings.y_span)
        for i, k in enumerate(COEF_KEYS):
            self.coef_table.setItem(i, 0, QTableWidgetItem(f"{settings.dx_coefs.get(k, 0.0):.2f}"))
            self.coef_table.setItem(i, 1, QTableWidgetItem(f"{settings.dy_coefs.get(k, 0.0):.2f}"))


class CalibrationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        cam_box = QGroupBox("Camera calibration")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        cam_tabs = QTabWidget()
        self.cam0_form = _CameraMappingForm("cam0")
        self.cam1_form = _CameraMappingForm("cam1")
        cam_tabs.addTab(self.cam0_form, "cam0")
        cam_tabs.addTab(self.cam1_form, "cam1")
        cam_layout.addWidget(cam_tabs)
        layout.addWidget(cam_box)

        geom_box = QGroupBox("World grid / dewarp")
        geom_grid = QGridLayout(geom_box)
        geom_grid.setContentsMargins(6, 6, 6, 6)
        geom_grid.setSpacing(4)
        self.world_h_spin = QSpinBox(); self.world_h_spin.setRange(1, 100000); self.world_h_spin.setValue(1000); self.world_h_spin.setMaximumWidth(SPIN_WIDTH)
        self.world_w_spin = QSpinBox(); self.world_w_spin.setRange(1, 100000); self.world_w_spin.setValue(1000); self.world_w_spin.setMaximumWidth(SPIN_WIDTH)
        self.world_scale_spin = QDoubleSpinBox(); self.world_scale_spin.setRange(1e-6, 1e6); self.world_scale_spin.setValue(1.0); self.world_scale_spin.setMaximumWidth(SPIN_WIDTH); self.world_scale_spin.setDecimals(DECIMALS)
        self.dewarp_order_spin = QSpinBox(); self.dewarp_order_spin.setRange(0, 5); self.dewarp_order_spin.setValue(1); self.dewarp_order_spin.setMaximumWidth(SPIN_WIDTH)
        geom_grid.addWidget(QLabel("World shape (H, W):"), 0, 0)
        geom_grid.addWidget(self.world_h_spin, 0, 1)
        geom_grid.addWidget(self.world_w_spin, 0, 2)
        geom_grid.addWidget(QLabel("World scale (px/mm):"), 1, 0)
        geom_grid.addWidget(self.world_scale_spin, 1, 1)
        geom_grid.addWidget(QLabel("Dewarp order:"), 2, 0)
        geom_grid.addWidget(self.dewarp_order_spin, 2, 1)
        layout.addWidget(geom_box)

        angle_box = QGroupBox("Stereo viewing angles (deg)")
        angle_box.setToolTip(
            "See the original Stereo_PIV_GPU/Stereo_PIV_CPU README: these "
            "angles are placeholders that should be verified per Z-plane "
            "before trusting the reconstructed W (or U/V)."
        )
        angle_grid = QGridLayout(angle_box)
        angle_grid.setContentsMargins(6, 6, 6, 6)
        angle_grid.setSpacing(4)
        self.alpha1_spin = self._angle_spin(-45.0)
        self.alpha2_spin = self._angle_spin(45.0)
        self.beta1_spin = self._angle_spin(0.0)
        self.beta2_spin = self._angle_spin(0.0)
        for i, (label, w) in enumerate([
            ("alpha1:", self.alpha1_spin), ("alpha2:", self.alpha2_spin),
            ("beta1:", self.beta1_spin), ("beta2:", self.beta2_spin),
        ]):
            angle_grid.addWidget(QLabel(label), i, 0)
            angle_grid.addWidget(w, i, 1)
        layout.addWidget(angle_box)
        layout.addStretch(1)

    @staticmethod
    def _angle_spin(default):
        s = QDoubleSpinBox()
        s.setRange(-180.0, 180.0)
        s.setValue(default)
        s.setMaximumWidth(SPIN_WIDTH)
        s.setDecimals(DECIMALS)
        return s

    def get_settings(self) -> StereoSettings:
        return StereoSettings(
            cam0_mapping=self.cam0_form.get_settings(),
            cam1_mapping=self.cam1_form.get_settings(),
            world_shape=(self.world_h_spin.value(), self.world_w_spin.value()),
            world_scale_px_per_mm=self.world_scale_spin.value(),
            dewarp_order=self.dewarp_order_spin.value(),
            alpha1_deg=self.alpha1_spin.value(), alpha2_deg=self.alpha2_spin.value(),
            beta1_deg=self.beta1_spin.value(), beta2_deg=self.beta2_spin.value(),
        )
