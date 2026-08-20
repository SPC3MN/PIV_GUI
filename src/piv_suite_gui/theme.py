"""Application-wide visual theme: a light, hairline-ruled "spec sheet"
look built on charcoal and alabaster, applied globally via
QApplication.setStyleSheet() rather than styling each widget individually.

Neutral by design -- no accent hue. Charcoal (#5D5E60) and its darker
text derivative carry every emphasis job (checked controls, focus rings,
primary buttons, the active tab), so the only non-neutral color in the
whole app is OK, and it means "this is working" rather than branding.

Surfaces run light-on-light with 1px rules doing the separating, rather
than the elevation/shadow approach a dark theme needs: cards are plain
white on an alabaster ground, corners are nearly square, and nothing
casts a shadow.

Tokens (kept as module constants rather than buried in the QSS string so
a future palette swap only touches this one block):
    BACKDROP  -- window ground the cards sit on
    PANEL     -- QGroupBox ("card") background
    PAPER     -- header / tab bar / table header background
    INK       -- primary text
    ACCENT    -- emphasis fill: checked controls, primary buttons, active tab
    INK_SOFT  -- secondary text (field labels, unselected tabs)
    INK_FAINT -- disabled text, status-bar text
    LINE      -- borders and hairlines
    INK_ON_ACCENT -- text drawn ON TOP of an ACCENT fill
    LOGO_INK  -- the lab mark's solid square (see widgets/header_bar.py);
                 kept separate from INK because it is artwork, not text,
                 and must stay the logo's own black if INK is ever tuned
    OK        -- the one non-neutral color, reserved for "this is working"
                 status. Darker than a dark theme's green would be, so it
                 still reads against white.
"""

BACKDROP = "#EBE9E9"
PANEL = "#FFFFFF"
PAPER = "#F6F5F5"
INK = "#2E2F30"
ACCENT = "#5D5E60"
INK_SOFT = "#6C6D6F"
INK_FAINT = "#9A9B9C"
LINE = "#D9D7D7"
INK_ON_ACCENT = "#FFFFFF"
LOGO_INK = "#111111"
OK = "#2E7D51"

_MONO_FONTS = '"Consolas", "Cascadia Mono", "Courier New", monospace'
_UI_FONTS = '"Segoe UI", "Helvetica Neue", Arial, sans-serif'

STYLESHEET = f"""
/* No background here on purpose: a blanket QWidget background paints
   plain container widgets (e.g. project_panel's loose-options box) as
   solid rectangles INSIDE the white cards that contain them. Grounds are
   set on the named surfaces below instead, and everything else inherits
   from its parent. */
QWidget {{
    color: {INK};
    font-family: {_UI_FONTS};
    font-size: 9.5pt;
}}

QMainWindow {{ background-color: {BACKDROP}; }}
QWidget#centralSurface {{ background-color: {BACKDROP}; }}
QScrollArea, QScrollArea > QWidget > QWidget, QTabWidget::pane {{
    background: transparent;
    border: none;
}}

QToolTip {{
    background-color: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    padding: 5px 8px;
}}

/* ---- branded header + status bar (see widgets/header_bar.py) ---- */
QWidget#appHeader {{
    background-color: {PANEL};
    border-bottom: 1px solid {LINE};
}}
QLabel#brandName {{
    font-size: 13pt;
    font-weight: 700;
    color: {INK};
}}
QLabel#brandTagline {{
    font-size: 7pt;
    font-weight: 600;
    color: {INK_SOFT};
}}
QLabel#statusBadge {{
    background-color: {PAPER};
    border: 1px solid {LINE};
    border-radius: 3px;
    padding: 4px 11px;
    font-family: {_MONO_FONTS};
    font-size: 8.5pt;
}}
QPushButton#headerRunButton {{ padding: 6px 16px; }}

QStatusBar {{
    background-color: {PANEL};
    border-top: 1px solid {LINE};
    color: {INK_FAINT};
}}
QStatusBar QLabel {{
    color: {INK_FAINT};
    font-family: {_MONO_FONTS};
    font-size: 8pt;
}}
QStatusBar::item {{ border: none; }}

QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 4px;
    /* margin-top must clear the title's full line height, or the box's
       top border draws straight through the text -- confirmed on a real
       screenshot, where the taller titles were struck through and the
       shorter ones happened to clear it. */
    margin-top: 21px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 3px;
    top: 1px;
    padding: 0 3px 0 0;
    color: {ACCENT};
    font-weight: 700;
}}

QLabel {{ background: transparent; color: {INK}; }}

QPushButton {{
    background-color: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 3px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: {PAPER}; border-color: {INK_FAINT}; }}
QPushButton:pressed {{ background-color: #E7E5E5; }}
QPushButton:disabled {{ color: {INK_FAINT}; background-color: {PAPER}; border-color: {LINE}; }}

QPushButton[accent="true"] {{
    background-color: {ACCENT};
    color: {INK_ON_ACCENT};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background-color: #6E6F71; border-color: #6E6F71; }}
QPushButton[accent="true"]:pressed {{ background-color: #4B4C4E; }}
/* Not a faded ACCENT fill: white-on-light-grey is nearly unreadable, so
   a disabled primary button falls back to the ordinary disabled look. */
QPushButton[accent="true"]:disabled {{ background-color: {PAPER}; color: {INK_FAINT}; border-color: {LINE}; }}

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {PANEL};
    color: {INK};
    border: 1px solid {LINE};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: {INK_ON_ACCENT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {INK_FAINT};
    background-color: {PAPER};
    border-color: {LINE};
}}
QSpinBox, QDoubleSpinBox {{ font-family: {_MONO_FONTS}; }}

QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {INK};
    selection-background-color: {ACCENT};
    selection-color: {INK_ON_ACCENT};
    border: 1px solid {LINE};
    outline: none;
}}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; color: {INK}; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {INK_FAINT}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {INK_FAINT};
    background: {PANEL};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator {{ border-radius: 2px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {LINE};
    background: {PAPER};
}}

QTabBar::tab {{
    background: transparent;
    color: {INK_SOFT};
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {INK}; }}

QProgressBar {{
    border: 1px solid {LINE};
    border-radius: 3px;
    background: {PAPER};
    text-align: center;
    color: {INK};
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 2px; }}

QHeaderView::section {{
    background-color: {PAPER};
    color: {INK_SOFT};
    padding: 4px 6px;
    border: none;
    border-bottom: 1px solid {LINE};
    font-weight: 600;
}}
QTableWidget, QTableView {{
    background-color: {PANEL};
    gridline-color: {LINE};
    border: 1px solid {LINE};
    border-radius: 3px;
    font-family: {_MONO_FONTS};
    selection-background-color: {ACCENT};
    selection-color: {INK_ON_ACCENT};
}}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #C9C7C7; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {INK_FAINT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #C9C7C7; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {INK_FAINT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def apply_theme(app):
    """Apply the charcoal-on-alabaster stylesheet to the whole
    application, plus a matching QPalette (belt-and-suspenders -- native
    dialogs like QFileDialog/QMessageBox lean on the palette more than the
    stylesheet)."""
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
    palette.setColor(QPalette.Base, QColor(PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(PAPER))
    palette.setColor(QPalette.ToolTipBase, QColor(PANEL))
    palette.setColor(QPalette.ToolTipText, QColor(INK))
    palette.setColor(QPalette.Text, QColor(INK))
    palette.setColor(QPalette.Button, QColor(PANEL))
    palette.setColor(QPalette.ButtonText, QColor(INK))
    palette.setColor(QPalette.BrightText, QColor(INK))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(INK_ON_ACCENT))
    palette.setColor(QPalette.PlaceholderText, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(INK_FAINT))
    app.setPalette(palette)


def apply_titlebar_theme(widget):
    """Best-effort: force this top-level window's native Windows title bar
    to LIGHT (DWMWA_USE_IMMERSIVE_DARK_MODE = 0), so a user running
    Windows in dark mode doesn't get a black title bar sitting on top of
    a light app.

    Windows 10 2004+/11 only; a silent no-op everywhere else -- matches
    this codebase's existing pattern of best-effort OS integration that
    must never block the app from launching (see gpu_engine.py's
    diagnostic logging)."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        value = ctypes.c_int(0)
        for attribute in (20, 19):  # 20 on recent Windows, 19 on early 1809/1903 builds
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
            if result == 0:
                break
    except Exception:
        pass  # cosmetic only -- never let this break the app launching
