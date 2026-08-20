"""Branded application header: product mark and wordmark on the left, a
live backend-availability badge and the primary "Run batch" action on the
right.

Purely a presentation shell -- the Run button here forwards to
run_panel's own button rather than duplicating its logic, so there's
exactly one run code path and one source of truth for whether running is
currently allowed (see main_window's wiring).

The mark is painted with QPainter instead of shipping an icon file: the
frozen PyInstaller build bundles no loose GUI assets today, and keeping
it that way means one less thing that can silently fail to be included
(the same class of bug as the missing graphlib/imageio metadata that
broke the packaged app before -- see installer/README.md).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from piv_suite.engines.registry import gpu_summary

from ..theme import INK, INK_SOFT, OK

MARK_BOX = 26.0  # logical coordinate box the mark is drawn in, then scaled


class _ProductMark(QWidget):
    """A ring with two opposed arcs and arrowheads -- a rotating flow
    field, the thing this whole application measures."""

    def __init__(self, size=28, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale = self.width() / MARK_BOX
        painter.scale(scale, scale)

        color = QColor(INK)
        painter.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(1.2, 1.2, MARK_BOX - 2.4, MARK_BOX - 2.4)

        # Two 90-degree arcs on the inner radius, opposite each other, so
        # the mark reads as rotation rather than as a plain target/circle.
        # Qt angles are 1/16th degree, 0 at 3 o'clock, counter-clockwise.
        painter.setPen(QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap))
        inner = (6.0, 6.0, 14.0, 14.0)  # x, y, w, h -- radius 7 about (13, 13)
        painter.drawArc(*inner, 180 * 16, -90 * 16)
        painter.drawArc(*inner, 0 * 16, -90 * 16)

        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        for tip_y, base_y, flip in ((5.4, 9.4, 1.0), (20.6, 16.6, -1.0)):
            head = QPainterPath()
            head.moveTo(13.0, tip_y)
            head.lineTo(13.0 + 2.1 * flip, base_y)
            head.lineTo(13.0 - 2.1 * flip, base_y)
            head.closeSubpath()
            painter.drawPath(head)
        painter.end()


class HeaderBar(QWidget):
    run_requested = Signal()

    def __init__(self, version="", parent=None):
        super().__init__(parent)
        self.setObjectName("appHeader")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._build_ui(version)

    def _build_ui(self, version):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        layout.addWidget(_ProductMark())

        wordmark = QVBoxLayout()
        wordmark.setSpacing(0)
        name = QLabel("PIV SUITE")
        name.setObjectName("brandName")
        tagline = QLabel("PARTICLE IMAGE VELOCIMETRY")
        tagline.setObjectName("brandTagline")
        # Letter-spacing has no QSS equivalent, so it's set on the font --
        # safe here because this label has no children to inherit it.
        spaced = tagline.font()
        spaced.setLetterSpacing(QFont.AbsoluteSpacing, 1.3)
        tagline.setFont(spaced)
        wordmark.addWidget(name)
        wordmark.addWidget(tagline)
        layout.addLayout(wordmark)

        layout.addStretch(1)

        self.backend_badge = QLabel()
        self.backend_badge.setObjectName("statusBadge")
        layout.addWidget(self.backend_badge)

        self.run_btn = QPushButton("▶  Run batch")
        self.run_btn.setObjectName("headerRunButton")
        self.run_btn.setProperty("accent", True)
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Preview a pair successfully before running a batch.")
        self.run_btn.clicked.connect(self.run_requested)
        layout.addWidget(self.run_btn)

        self._version = version
        self.refresh_backend_badge()

    def refresh_backend_badge(self):
        """Show which backend this machine can actually use. Called once at
        construction; safe to call again if the environment changes."""
        summary = gpu_summary()
        if summary:
            dot, text, tip = OK, f"GPU · {summary}", "GPU backend available on this machine."
        else:
            dot, text, tip = INK_SOFT, "CPU only", (
                "GPU backend unavailable -- cupy/openpiv-python-gpu not importable, "
                "or no CUDA device detected.")
        self.backend_badge.setText(
            f'<span style="color:{dot};">●</span>'
            f'<span style="color:{INK_SOFT};"> {text}</span>')
        self.backend_badge.setToolTip(tip)

    def set_run_enabled(self, enabled: bool):
        self.run_btn.setEnabled(enabled)

    def status_text(self):
        """Left-hand status-bar line: the same backend fact as the badge,
        spelled out for the bottom bar."""
        summary = gpu_summary()
        return f"GPU ready · {summary}" if summary else "CPU backend"

    def version_text(self):
        return f"PIV Suite v{self._version}" if self._version else "PIV Suite"
