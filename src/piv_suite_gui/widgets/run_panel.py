"""Run panel: progress bar, per-pair status table, log console, and
Run/Cancel buttons. Owns the QThread lifecycle for PipelineWorker --
construct worker -> moveToThread -> connect signals -> thread.start();
on cancel, sets the worker's cancellation flag and waits for a clean exit.
"""

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QPlainTextEdit, QProgressBar, QPushButton,
    QTableView, QVBoxLayout, QWidget,
)

from piv_suite_gui.models.job_model import JobModel
from piv_suite_gui.workers.pipeline_worker import PipelineWorker


class RunPanel(QWidget):
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
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_view)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(5000)
        layout.addWidget(self.log_console)

    def set_run_enabled(self, enabled: bool):
        self.run_btn.setEnabled(enabled)

    def _start_run(self):
        main_window = self.window()
        project = main_window.project_panel.get_project_settings()
        correlation = main_window.settings_panel.get_correlation_settings()
        validation = main_window.settings_panel.get_validation_settings()
        post = main_window.settings_panel.get_postprocess_settings()
        calibration = main_window.settings_panel.get_calibration_settings()

        from piv_suite.config.schema import ProjectConfig
        config = ProjectConfig(project=project, correlation=correlation,
                                validation=validation, postprocess=post,
                                calibration=calibration)
        if project.mode == "stereo":
            config.stereo = main_window.calibration_panel.get_settings()
        config.output.save_npz = True
        config.output.save_summary_csv = True

        self.job_model.reset()
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.run_btn.setEnabled(False)
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
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _cancel_run(self):
        if self._worker is not None:
            self._worker.cancel()
            self.log_console.appendPlainText("Cancelling after the current pair...")
