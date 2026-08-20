"""GUI entry point: `python -m piv_suite_gui.app`."""

import sys

from PySide6.QtWidgets import QApplication

from piv_suite_gui.main_window import MainWindow
from piv_suite_gui.theme import apply_theme, apply_titlebar_theme


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    apply_titlebar_theme(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
