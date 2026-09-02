"""Small shared GUI helpers."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDoubleSpinBox, QLabel, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)


def style_spin(spin, width=80, decimals=None):
    """Cap a spin box's width, hide its up/down counter buttons (value is
    still editable by typing or scrolling), and set a uniform decimal
    display for QDoubleSpinBox."""
    spin.setMaximumWidth(width)
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    if isinstance(spin, QDoubleSpinBox):
        spin.setDecimals(2 if decimals is None else decimals)
    return spin


def fit_table_to_rows(table):
    """Size a QTableWidget's height to show every row without its own
    vertical scroll bar -- the containing panel's own QScrollArea handles
    any overall overflow instead. Call again after adding/removing rows."""
    header_h = table.horizontalHeader().height()
    rows_h = sum(table.rowHeight(r) for r in range(table.rowCount()))
    frame = 2 * table.frameWidth()
    total = header_h + rows_h + frame + 2
    table.setMinimumHeight(total)
    table.setMaximumHeight(total)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


class CollapsibleSection(QWidget):
    """A disclosure: a flat toggle button over a body that is hidden by
    default.

    Exists so the expert surface stops competing with the everyday one. The
    panel used to present every setting at equal weight -- thirteen titled
    group boxes in one scrolling column, including a 10x2 polynomial
    coefficient table and a group literally labelled "(INTERNAL)" -- so
    nothing read as primary. Settings that are real but rarely touched live
    in one of these instead: still one click away, no longer the first thing
    a new user has to scroll past.

    Collapsed state is the DEFAULT rather than remembered, deliberately:
    reopening the app to whatever drawer happened to be open last time is
    how a tidy panel drifts back into a wall of controls.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle = QToolButton()
        self.toggle.setObjectName("disclosure")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        self.body = QWidget()
        self.body.setVisible(False)
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(0, 4, 0, 0)
        self._body_layout.setSpacing(4)
        layout.addWidget(self.body)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.body.setVisible(checked)

    def add_widget(self, widget):
        self._body_layout.addWidget(widget)
        return widget

    def set_expanded(self, expanded):
        self.toggle.setChecked(bool(expanded))

    @property
    def is_expanded(self):
        return self.toggle.isChecked()


def section_label(text):
    """A small uppercase label for a control group that no longer has its own
    titled group box.

    Sections that were folded together (Mode and Backend into Source, say)
    still need to say what they are; this gives them a header's weight without
    a card's chrome, which is the whole point of folding them."""
    label = QLabel(text.upper())
    label.setObjectName("inlineSectionLabel")
    return label


class ElidingLabel(QLabel):
    """A QLabel that shortens with an ellipsis instead of being cut off.

    A plain QLabel clipped by its layout truncates mid-word with no ellipsis,
    so the reader cannot tell text is missing -- the preview's status line lost
    "...valid (range/residual rej" at the window's own minimum size. The full
    text is always available as the tooltip.
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text):
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._apply_elide()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self):
        metrics = self.fontMetrics()
        super().setText(metrics.elidedText(self._full_text, Qt.ElideRight,
                                           max(self.width() - 2, 24)))
