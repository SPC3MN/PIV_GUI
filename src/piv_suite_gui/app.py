"""GUI entry point: `python -m piv_suite_gui.app`."""

import sys

from PySide6.QtWidgets import QApplication

from piv_suite_gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
