"""Stereo calibration panel: per-camera DaVis polynomial coefficients,
world/dewarp geometry, and the two cameras' viewing angles used by
reconstruct_stereo. Manual coefficient entry (structured form fields
instead of hand-edited JSON) is the near-term workflow -- the "Load from
DaVis report..." button is wired to calibration.report_parser's stub
interface but stays disabled until that parser is implemented (see
calibration/report_parser.py).
"""

from PySide6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from piv_suite.calibration.report_parser import parse_davis_calibration_report
from piv_suite.config.schema import CameraMappingSettings, StereoSettings

COEF_KEYS = ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")


class _CameraMappingForm(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        top = QGridLayout()
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

        load_btn = QPushButton("Load from DaVis calibration report...")
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
        for row in range(len(COEF_KEYS)):
            self.coef_table.setItem(row, 0, QTableWidgetItem("0.0"))
            self.coef_table.setItem(row, 1, QTableWidgetItem("0.0"))
        layout.addWidget(self.coef_table)

    @staticmethod
    def _make_spin(default=0.0):
        s = QDoubleSpinBox()
        s.setRange(-1e7, 1e7)
        s.setDecimals(4)
        s.setValue(default)
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
            self.coef_table.setItem(i, 0, QTableWidgetItem(str(settings.dx_coefs.get(k, 0.0))))
            self.coef_table.setItem(i, 1, QTableWidgetItem(str(settings.dy_coefs.get(k, 0.0))))


class CalibrationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        cams = QHBoxLayout()
        self.cam0_form = _CameraMappingForm("cam0")
        self.cam1_form = _CameraMappingForm("cam1")
        cams.addWidget(self.cam0_form)
        cams.addWidget(self.cam1_form)
        layout.addLayout(cams)

        geom_box = QGroupBox("World grid / dewarp")
        geom_grid = QGridLayout(geom_box)
        self.world_h_spin = QSpinBox(); self.world_h_spin.setRange(1, 100000); self.world_h_spin.setValue(1000)
        self.world_w_spin = QSpinBox(); self.world_w_spin.setRange(1, 100000); self.world_w_spin.setValue(1000)
        self.world_scale_spin = QDoubleSpinBox(); self.world_scale_spin.setRange(1e-6, 1e6); self.world_scale_spin.setValue(1.0)
        self.dewarp_order_spin = QSpinBox(); self.dewarp_order_spin.setRange(0, 5); self.dewarp_order_spin.setValue(1)
        geom_grid.addWidget(QLabel("World shape (H, W):"), 0, 0)
        geom_grid.addWidget(self.world_h_spin, 0, 1)
        geom_grid.addWidget(self.world_w_spin, 0, 2)
        geom_grid.addWidget(QLabel("World scale (px/mm):"), 1, 0)
        geom_grid.addWidget(self.world_scale_spin, 1, 1)
        geom_grid.addWidget(QLabel("Dewarp interpolation order:"), 2, 0)
        geom_grid.addWidget(self.dewarp_order_spin, 2, 1)
        layout.addWidget(geom_box)

        angle_box = QGroupBox("Stereo viewing angles (deg) -- see original README: verify per Z-plane")
        angle_grid = QGridLayout(angle_box)
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
