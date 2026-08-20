"""Application-wide visual theme: charcoal/alabaster monochrome instrument
look, applied globally via QApplication.setStyleSheet() rather than styling
each widget individually. Neutral by design -- no accent hue -- since the
brief was specifically charcoal (#5D5E60) + alabaster (#EBE9E9), not the
mauve/snow palette used elsewhere in this project's design exploration.

Tokens (kept as module constants rather than buried in the QSS string so
a future palette swap only touches this one block):
    BACKDROP  -- window/page background, the darkest surface
    PANEL     -- QGroupBox ("card") background, one step up from BACKDROP
    PAPER     -- tab bar / table header background, between BACKDROP and PANEL
    INK       -- primary text AND the sole accent (buttons, checked states,
                 focus rings, progress fill) -- both derive from the same
                 alabaster since the palette has no separate accent hue
    INK_SOFT  -- secondary text (field labels, unselected tabs)
    INK_FAINT -- disabled text, placeholder-level text
    LINE      -- borders/hairlines
    INK_ON_ACCENT -- text color used ON TOP of an INK-colored fill (e.g. a
                 primary button) -- needs to be dark, not the accent itself
"""

BACKDROP = "#242527"
PANEL = "#5D5E60"
PAPER = "#3A3B3D"
INK = "#EBE9E9"
INK_SOFT = "#B4B2B2"
INK_FAINT = "#8B8989"
LINE = "#4C4D4F"
INK_ON_ACCENT = "#1C1D1E"

_MONO_FONTS = '"Consolas", "Cascadia Mono", "Courier New", monospace'
_UI_FONTS = '"Segoe UI", "Helvetica Neue", Arial, sans-serif'

STYLESHEET = f"""
QWidget {{
    background-color: {BACKDROP};
    color: {INK};
    font-family: {_UI_FONTS};
    font-size: 9.5pt;
}}

QToolTip {{
    background-color: {PAPER};
    color: {INK};
    border: 1px solid {LINE};
    padding: 5px 8px;
    border-radius: 4px;
}}

QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget, QTabWidget::pane {{
    background-color: {BACKDROP};
    border: none;
}}

QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 8px;
    margin-top: 16px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 3px;
    padding: 0 4px;
    color: {INK};
    font-weight: 600;
}}

QLabel {{
    background: transparent;
    color: {INK};
}}

QPushButton {{
    background-color: {PAPER};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: #46474A; }}
QPushButton:pressed {{ background-color: #303132; }}
QPushButton:disabled {{ color: {INK_FAINT}; background-color: #2B2C2E; border-color: #38393B; }}

QPushButton[accent="true"] {{
    background-color: {INK};
    color: {INK_ON_ACCENT};
    border: 1px solid {INK};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background-color: #D8D6D6; }}
QPushButton[accent="true"]:pressed {{ background-color: #C7C5C5; }}
QPushButton[accent="true"]:disabled {{ background-color: #4A4B4D; color: {INK_FAINT}; border-color: #4A4B4D; }}

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BACKDROP};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {INK};
    selection-color: {INK_ON_ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {INK};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {INK_FAINT};
    background-color: #2B2C2E;
    border-color: #38393B;
}}
QSpinBox, QDoubleSpinBox {{
    font-family: {_MONO_FONTS};
}}

QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {PAPER};
    color: {INK};
    selection-background-color: {PANEL};
    border: 1px solid {LINE};
    outline: none;
}}

QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
    color: {INK};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {INK_FAINT}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {INK_FAINT};
    background: {BACKDROP};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {INK};
    border-color: {INK};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {LINE};
}}

QTabBar::tab {{
    background: transparent;
    color: {INK_SOFT};
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {INK}; }}
QTabBar::tab:hover {{ color: {INK}; }}

QProgressBar {{
    border: 1px solid {LINE};
    border-radius: 5px;
    background: {PAPER};
    text-align: center;
    color: {INK};
}}
QProgressBar::chunk {{ background-color: {INK}; border-radius: 4px; }}

QHeaderView::section {{
    background-color: {PAPER};
    color: {INK_SOFT};
    padding: 4px 6px;
    border: none;
    border-bottom: 1px solid {LINE};
    font-weight: 600;
}}
QTableWidget, QTableView {{
    background-color: {BACKDROP};
    gridline-color: {LINE};
    border: 1px solid {LINE};
    border-radius: 5px;
    font-family: {_MONO_FONTS};
    selection-background-color: {PANEL};
    selection-color: {INK};
}}

QScrollBar:vertical {{ background: {BACKDROP}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {PANEL}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {BACKDROP}; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {PANEL}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def apply_theme(app):
    """Apply the charcoal/alabaster stylesheet to the whole application,
    plus a matching QPalette (belt-and-suspenders -- native dialogs like
    QFileDialog/QMessageBox lean on the palette more than the stylesheet)
    and, on Windows, the OS-level dark title bar so the window chrome
    itself doesn't stay a bright default white."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QStyleFactory

    # The native Windows style ("windowsvista") draws checkbox/radio
    # indicators itself and mostly ignores QSS ::indicator rules -- Fusion
    # is the cross-platform style that actually respects them, confirmed
    # against a real screenshot: checked radios/checkboxes rendered as
    # plain empty rings under the native style regardless of the
    # :checked background-color rule above.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet(STYLESHEET)

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(BACKDROP))
    palette.setColor(QPalette.WindowText, QColor(INK))
    palette.setColor(QPalette.Base, QColor(BACKDROP))
    palette.setColor(QPalette.AlternateBase, QColor(PANEL))
    palette.setColor(QPalette.ToolTipBase, QColor(PAPER))
    palette.setColor(QPalette.ToolTipText, QColor(INK))
    palette.setColor(QPalette.Text, QColor(INK))
    palette.setColor(QPalette.Button, QColor(PAPER))
    palette.setColor(QPalette.ButtonText, QColor(INK))
    palette.setColor(QPalette.BrightText, QColor(INK))
    palette.setColor(QPalette.Highlight, QColor(INK))
    palette.setColor(QPalette.HighlightedText, QColor(INK_ON_ACCENT))
    palette.setColor(QPalette.PlaceholderText, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(INK_FAINT))
    app.setPalette(palette)


def enable_dark_titlebar(widget):
    """Best-effort: ask Windows' DWM to draw this top-level window's own
    title bar in dark mode (DWMWA_USE_IMMERSIVE_DARK_MODE), so the native
    chrome around the app doesn't stay bright white against a dark app
    body. Windows 10 2004+/11 only; a no-op (silently) everywhere else --
    matches this codebase's existing pattern of best-effort OS integration
    that must never block the app from launching (see gpu_engine.py's
    diagnostic logging)."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        value = ctypes.c_int(1)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 on recent Windows, 19 on early 1809/1903 builds
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
            if result == 0:
                break
    except Exception:
        pass  # cosmetic only -- never let this break the app launching
