"""Run panel: progress bar, per-pair status table, log console, and
Run/Cancel buttons. Owns the QThread lifecycle for PipelineWorker --
construct worker -> moveToThread -> connect signals -> thread.start();
on cancel, calls the worker's force_stop() (the strongest cancellation
currently available -- see pipeline_worker.PipelineWorker.force_stop's
docstring) and waits for a clean exit.
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QPlainTextEdit, QProgressBar, QPushButton,
    QTableView, QVBoxLayout, QWidget,
)

from piv_suite_gui.models.job_model import JobModel
from piv_suite_gui.workers.pipeline_worker import PipelineWorker


class RunPanel(QWidget):
    # Mirrors every change to the Run button's enabled state so a second
    # surface for the same action (main_window's header button) can follow
    # it without duplicating the gating rules -- which are: enabled only
    # after a successful preview, and disabled again while a batch runs.
    run_enabled_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run batch")
        self.run_btn.setProperty("accent", True)
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Preview the first pair successfully before running a batch.")
        self.run_btn.clicked.connect(self._start_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_run)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.job_model = JobModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.job_model)
        # Numeric columns size to their contents; Pair and Status share the
        # slack, with Status CAPPED. A failed pair's status is the full
        # exception text -- 200-700 characters on this project's own real
        # errors -- and sizing that column to its contents pushed the other
        # five off-screen entirely, which is worse than the clipped headers
        # this replaced. The full text is in the cell's tooltip.
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setMaximumSectionSize(260)
        header.setStretchLastSection(False)
        self.table_view.setTextElideMode(Qt.ElideRight)
        self.table_view.setWordWrap(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.verticalHeader().setVisible(False)
        # The table is the thing being watched; the log is for when something
        # goes wrong. 3:1 rather than the even split they had, which gave half
        # the panel to a console that is empty on a healthy run.
        layout.addWidget(self.table_view, stretch=3)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(5000)
        self.log_console.setPlaceholderText("Run output appears here.")
        layout.addWidget(self.log_console, stretch=1)

    def set_run_enabled(self, enabled: bool):
        self.run_btn.setEnabled(enabled)
        self.run_enabled_changed.emit(enabled)

    def is_running(self) -> bool:
        """True while a batch's QThread is alive -- main_window.closeEvent
        uses this (rather than reaching into self._thread/_worker
        directly) to decide whether closing the window needs to stop a
        running batch first."""
        return self._thread is not None

    def _start_run(self):
        main_window = self.window()
        project = main_window.project_panel.get_project_settings()
        preprocess = main_window.project_panel.get_preprocess_settings()
        correlation = main_window.settings_panel.get_correlation_settings()
        validation = main_window.settings_panel.get_validation_settings()
        post = main_window.settings_panel.get_postprocess_settings()
        calibration = main_window.settings_panel.get_calibration_settings()
        performance = main_window.settings_panel.get_performance_settings()

        from piv_suite.config.schema import ProjectConfig
        config = ProjectConfig(project=project, preprocess=preprocess,
                                correlation=correlation, validation=validation,
                                postprocess=post, calibration=calibration,
                                performance=performance)
        if project.mode == "stereo":
            config.stereo = main_window.calibration_panel.get_settings()
        elif project.dual_camera:
            config.dual_planar = main_window.project_panel.get_dual_planar_settings()
        config.output.save_npz = True
        config.output.save_summary_csv = True

        self.job_model.reset()
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.set_run_enabled(False)
        self.cancel_btn.setEnabled(True)

        self._thread = QThread()
        self._worker = PipelineWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.pair_started.connect(self.job_model.start_pair)
        self._worker.pair_finished.connect(self.job_model.finish_pair)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self.log_console.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        # Direct (not queued) connection: guarantees the QThread's own
        # event loop is told to quit the MOMENT the worker actually
        # finishes, regardless of whether the GUI event loop happens to
        # be pumping right then. _on_finished's own thread.quit()/wait()
        # (queued, only runs once the GUI loop next pumps) is enough for
        # the normal Cancel-button path, but stop_and_wait() (used by
        # main_window.closeEvent) blocks the GUI thread on QThread.wait()
        # itself -- with no event loop left to pump, that would otherwise
        # deadlock until its own timeout even for a worker that already
        # stopped. quit() is documented thread-safe, so calling it
        # straight from the worker thread here is safe, and harmless
        # alongside _on_finished's own call (idempotent).
        self._worker.finished.connect(self._thread.quit, Qt.DirectConnection)

        self._thread.start()

    def _on_error(self, pair_id, message):
        self.job_model.error_pair(pair_id, message)
        self.log_console.appendPlainText(f"[error] {pair_id}: {message}")

    def _on_progress(self, current, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
        else:
            self.progress_bar.setMaximum(0)  # busy indicator when total is unknown
        self.progress_bar.setValue(current)

    def _on_finished(self, cancelled):
        self.log_console.appendPlainText("Cancelled." if cancelled else "Done.")
        self.set_run_enabled(True)
        self.cancel_btn.setEnabled(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _cancel_run(self):
        if self._worker is not None:
            self._worker.force_stop()
            # Honest about the actual bound now: force_stop() (see its
            # docstring) checks between multi-pass iterations/tiles, not
            # only between whole pairs -- so this is usually much faster
            # than "after the current pair", but the current PASS (or,
            # tiled-GPU, the current TILE) still has to finish first,
            # since a single blocking correlation call can't be safely
            # pre-empted mid-call.
            self.log_console.appendPlainText("Cancelling (finishes the current pass/tile first)...")

    def stop_and_wait(self, timeout_ms=5000) -> bool:
        """Used by main_window.closeEvent: force_stop() the worker (if one
        is running) and block for up to timeout_ms waiting for its
        QThread to actually terminate, so the window doesn't close while
        a worker thread (or, once the sibling process-pool cancellation
        change lands, its ProcessPoolExecutor's worker processes) is
        still alive underneath it. Returns True if nothing was running or
        the thread stopped in time; False if it's still running after the
        timeout -- the caller should log that as a real problem, not
        silently proceed as if cleanup succeeded.

        Safe to call unconditionally (no-ops if nothing is running), but
        callers should still gate on is_running() themselves if they want
        to log/skip the "cancelling a running batch" messaging only when
        there actually is one."""
        if self._thread is None:
            return True
        if self._worker is not None:
            self._worker.force_stop()
        stopped = self._thread.wait(timeout_ms)
        if stopped:
            self._thread = None
            self._worker = None
        return stopped
