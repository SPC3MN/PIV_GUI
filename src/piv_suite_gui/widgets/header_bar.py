"""Branded application header: product mark and wordmark on the left, a
live backend-availability badge and the primary "Run batch" action on the
right.

Purely a presentation shell -- the Run button here forwards to
run_panel's own button rather than duplicating its logic, so there's
exactly one run code path and one source of truth for whether running is
currently allowed (see main_window's wiring).

The mark is the real Flow and Turbulence Engineering Laboratory logo
(supplied as an SVG), embedded here as a Python string constant and
rendered with QtSvg rather than loaded from a loose file on disk -- the
frozen PyInstaller build bundles no loose GUI assets today, and keeping
it that way means one less thing that can silently fail to be included
(the same class of bug as the missing graphlib/imageio metadata that
broke the packaged app before -- see installer/README.md).
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from piv_suite.engines.registry import gpu_summary

from ..theme import INK_SOFT, OK

# The lab's mark, isolated from the supplied logosvg.svg (its `id="rect1"`
# path is the icon; the rest of that file is unrendered Inkscape flow-text
# scaffolding for the wordmark, which this header sets as real QLabel text
# instead). viewBox matches the path's own bounding box
# (QSvgRenderer.boundsOnElement confirmed ~60x50) so it fills the render
# target with no extra margin baked in.
_LOGO_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 50">
<path fill="#000000" fill-rule="evenodd" d="M 2.50011,0 A 2.4999999,2.4999999 0 0 0 0,2.50011 v 9.80818 h 23.94273 c 0.002,-10e-6 0.003,-5.1e-4 0.005,-5.1e-4 h 6.05234 c 1.38059,-2.3e-4 2.49965,-1.11952 2.49959,-2.5001096 -2.3e-4,-1.38039 -1.1192,-2.4993604 -2.49959,-2.4995904 -1.12456,-5e-5 -2.07579,0.7427204 -2.39004,1.7642304 -0.12079,0.39266 -0.44444,0.73536 -0.85679,0.73536 h -3.25355 c -0.55228,0 -1.007,-0.44977 -0.93431,-0.99684 C 23.00977,5.46625 25.65862,2.81817 29.00323,2.37401 c 0.27354,-0.0363 0.63505,-0.0548 0.99684,-0.0548 0.36178,1e-5 0.72381,0.0184 0.99735,0.0548 3.34439,0.44437 5.99245,3.09243 6.43682,6.4368204 0.0727,0.54708 0.0726,1.4471196 0,1.9941896 -0.44416,3.34462 -3.09224,5.99295 -6.43682,6.43734 -0.27354,0.0363 -0.63555,0.0543 -0.99735,0.0537 v 0.0119 H 0 v 5.19141 h 43.9477 6.05235 c 1.38059,-2.3e-4 2.49965,-1.11951 2.49959,-2.5001 -2.3e-4,-1.3804 -1.1192,-2.49937 -2.49959,-2.49959 -1.12456,-5e-5 -2.07579,0.74271 -2.39004,1.76423 -0.12079,0.39266 -0.44445,0.73536 -0.85679,0.73536 h -3.25355 c -0.55229,0 -1.007,-0.44977 -0.93431,-0.99684 0.44438,-3.34459 3.09323,-5.99318 6.43785,-6.43734 0.27354,-0.0363 0.63505,-0.0543 0.99684,-0.0543 0.36178,2e-5 0.72381,0.0184 0.99735,0.0548 3.34439,0.44436 5.99245,3.09242 6.43682,6.43682 0.0727,0.54707 0.0727,1.44711 0,1.99419 -0.44416,3.34462 -3.09224,5.99295 -6.43682,6.43733 -0.27354,0.0363 -0.63555,0.0543 -0.99735,0.0538 v 0.0119 H 0 v 5.19142 h 40.00014 v 0.0119 c 0.36177,-5.3e-4 0.7233,0.0174 0.99684,0.0538 3.34458,0.44438 5.99318,3.09271 6.43733,6.43733 0.0726,0.54708 0.0727,1.44712 0,1.99419 -0.44436,3.3444 -3.09294,5.99298 -6.43733,6.43734 -0.54707,0.0727 -1.44711,0.0727 -1.9942,0 -3.34461,-0.44416 -5.99294,-3.09275 -6.43733,-6.43734 -0.0727,-0.54707 0.38203,-0.99684 0.93431,-0.99684 h 3.25355 c 0.41235,0 0.736,0.3427 0.85679,0.73536 0.31425,1.02152 1.26548,1.76428 2.39004,1.76423 1.38039,-2.2e-4 2.49936,-1.11919 2.49959,-2.49959 6e-5,-1.38059 -1.119,-2.49987 -2.49959,-2.5001 H 33.9478 0 v 9.8087 a 2.4999999,2.4999999 0 0 0 2.50011,2.50011 h 54.99974 a 2.4999999,2.4999999 0 0 0 2.5001,-2.50011 V 2.5002 A 2.4999999,2.4999999 0 0 0 57.49985,9e-5 Z" />
</svg>"""
_LOGO_ASPECT = 60.0 / 50.0  # width / height, from the viewBox above


class _ProductMark(QWidget):
    """The Flow and Turbulence Engineering Laboratory mark, rendered from
    the embedded SVG (see _LOGO_SVG above) rather than loaded from a file
    on disk -- see this module's docstring for why that matters here.

    The source path's cutout curls are transparent (evenodd fill against
    nothing), not white -- rendered directly they'd show whatever sits
    behind the header instead of reading as a clean mark. A cached
    QImage with an opaque white background baked in first sidesteps that,
    and means the SVG is only ever rasterized once per widget rather
    than on every paint.
    """

    def __init__(self, size=34, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        render_h = 240
        render_w = round(render_h * _LOGO_ASPECT)
        self._image = QImage(render_w, render_h, QImage.Format_ARGB32)
        self._image.fill(Qt.white)
        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        QSvgRenderer(_LOGO_SVG).render(painter, QRectF(0, 0, render_w, render_h))
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)
        # Fit the (slightly wider-than-tall) mark into the square widget
        # without distorting it, centered.
        w, h = self.width(), self.height()
        fit_w, fit_h = w, w / _LOGO_ASPECT
        if fit_h > h:
            fit_h, fit_w = h, h * _LOGO_ASPECT
        target = QRectF((w - fit_w) / 2, (h - fit_h) / 2, fit_w, fit_h)
        painter.drawImage(target, self._image)
        painter.end()


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
        name = QLabel("PIV TESTING")
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
        return f"PIV Testing v{self._version}" if self._version else "PIV Testing"
