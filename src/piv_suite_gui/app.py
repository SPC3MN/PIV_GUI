"""GUI entry point: `python -m piv_suite_gui.app`."""

import multiprocessing
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
    # Required for Tier 3's ProcessPoolExecutor (processing/parallel_planar.py)
    # to work in the FROZEN (PyInstaller) installer build: on Windows,
    # multiprocessing always spawns by re-executing sys.executable, which
    # for a frozen app IS PIV_Suite.exe itself, not python.exe -- without
    # this call (a no-op when NOT frozen, so harmless for the normal `pip
    # install` case), each worker process would re-run this same
    # `if __name__ == "__main__"` block from the top instead of running as
    # a worker, relaunching the whole GUI per worker rather than
    # correlating a frame pair. Must be the very first thing here, per
    # the multiprocessing docs.
    multiprocessing.freeze_support()
    main()
