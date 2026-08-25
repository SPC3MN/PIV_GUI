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

    # Second safety net alongside MainWindow.closeEvent -- aboutToQuit
    # fires right before the event loop actually exits for ANY reason,
    # not only "the window closed normally" (e.g. a future
    # QApplication.quit()/exit() call, which stops the event loop
    # directly WITHOUT closing any window or firing its closeEvent at
    # all). Cheap and idempotent (see MainWindow._free_backend_resources'
    # docstring), so wiring it costs nothing on the normal path where
    # closeEvent already ran first. Deliberately does NOT also repeat
    # closeEvent's running-batch stop_and_wait() here: aboutToQuit
    # handlers still run ON the GUI event loop, and stop_and_wait()
    # blocks that same loop waiting on a QThread -- closeEvent already
    # owns that responsibility for the one real path this app has today
    # (the window closing), so duplicating it here would only add a
    # second place for that logic to drift, not real additional coverage.
    app.aboutToQuit.connect(window._free_backend_resources)

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
