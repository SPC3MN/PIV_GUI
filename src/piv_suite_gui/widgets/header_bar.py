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

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from piv_suite.engines.registry import gpu_summary

from ..theme import INK_SOFT, LOGO_INK, OK

MARK_BOX = 100.0  # logical coordinate box the mark is drawn in, then scaled

# Each entry is one flow line of the lab mark: (bar_start_x, bar_end_x,
# y, curl_radius, curls_up). The bar runs left-to-right at height `y` and
# terminates in a ~270-degree curl -- the wind/vortex figure from the
# Flow and Turbulence Engineering Laboratory logo. Bars start left of 0
# so they run flush to the rounded square's edge, which clips them.
_FLOW_LINES = (
    (-4.0, 38.0, 32.0, 11.0, True),
    (-4.0, 68.0, 51.0, 11.0, True),
    (-4.0, 54.0, 70.0, 11.0, False),
)
_STROKE = 10.0
# The knocked-out lines are the logo's own white, not a theme surface --
# they sit on LOGO_INK, which never changes with the palette.
_LINE_ON_LOGO = "#FFFFFF"


class _ProductMark(QWidget):
    """The Flow and Turbulence Engineering Laboratory mark: three curling
    flow lines knocked out of a solid rounded square.

    Drawn with QPainter rather than loaded from an image or SVG file so
    the frozen build stays free of loose GUI assets -- see this module's
    docstring for why that matters here.
    """

    def __init__(self, size=34, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale = self.width() / MARK_BOX
        painter.scale(scale, scale)

        square = QPainterPath()
        square.addRoundedRect(QRectF(0, 0, MARK_BOX, MARK_BOX), 11, 11)
        painter.setClipPath(square)  # so bars can overrun the left edge
        painter.fillPath(square, QColor(LOGO_INK))

        painter.setPen(QPen(QColor(_LINE_ON_LOGO), _STROKE, Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        for x0, cx, y, radius, up in _FLOW_LINES:
            painter.drawPath(_flow_line(x0, cx, y, radius, up))
        painter.end()


def _flow_line(x0, cx, y, radius, curls_up):
    """A horizontal bar ending in a ~270-degree curl.

    Qt arc angles: 0 deg at 3 o'clock, positive counter-clockwise on
    screen. An upward curl centres the circle above the bar's end, meets
    it at the circle's 6 o'clock point (-90 deg) and sweeps +270 deg, all
    the way round to 9 o'clock -- which is what leaves the curl's tail
    tucked back over itself the way the logo's do.
    """
    path = QPainterPath()
    path.moveTo(x0, y)
    path.lineTo(cx, y)
    if curls_up:
        path.arcTo(QRectF(cx - radius, y - 2 * radius, 2 * radius, 2 * radius), -90, 270)
    else:
        path.arcTo(QRectF(cx - radius, y, 2 * radius, 2 * radius), 90, -270)
    return path


class HeaderBar(QWidget):
    run_requested = Signal()

    def __init__(self, version="", parent=None):
        super().__init__(parent)
        self.setObjectName("appHeader")
        # A plain QWidget SUBCLASS ignores background-color/border from a
        # stylesheet unless it opts in here -- without this the header
        # silently renders in the window's ground color instead of its own
        # (confirmed by sampling the pixel, which came back as BACKDROP).
        self.setAttribute(Qt.WA_StyledBackground, True)
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
        tagline = QLabel("FLOW AND TURBULENCE ENGINEERING LABORATORY")
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
