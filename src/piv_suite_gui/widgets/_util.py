"""Small shared GUI helpers."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox


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
