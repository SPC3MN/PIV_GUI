"""Preview panel: runs the pipeline on a small range of consecutive pairs
(one pair by default, matching the previous single-pair-only behavior) and
renders their AVERAGED velocity-magnitude field inline, replacing
piv_common.preview_first_snapshot()'s blocking terminal y/N prompt.
Settings can be tweaked and re-previewed as many times as needed before
committing to a full batch run (gates run_panel's Run button via the
`previewed` signal -- see main_window.py). Supports planar (single
camera), dual-camera-planar (SideBySide2D, stitched), and stereo (two
cameras, dewarped and combined via reconstruct_stereo) preview.

The plot itself (piv_suite.plotting.preview.make_preview_figure) always
shows a single filled contour of |V| = sqrt(u**2+v**2[+w**2]), auto-scaled
to that field's own min/max, on a fixed "turbo" colormap -- there is no
longer a per-component/manual-range/colormap UI, see that module's
docstring for why. The one remaining toggle is an optional (u, v) vector
overlay, which only becomes available once a preview result exists to
draw it on top of (see show_vectors_check below).
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QObject, Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import numpy as np

from piv_suite.calibration.camera_mapping import (build_stereo_cameras, stereo_angles_for,
                                                  stereo_fov_valid)
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.davis_set import (
    get_dual_planar_from_set, get_pair_from_set, get_stereo_from_set,
    list_pair_ids_from_set, resolve_set_paths,
)
from piv_suite.io.loose_files import (
    get_pair_from_loose_files, get_stereo_from_loose_files,
    list_pair_ids_from_loose_files, list_pair_ids_stereo_from_loose_files,
)
from piv_suite.plotting.preview import make_preview_figure
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration
from piv_suite.processing.preprocess import apply_preprocess_pair

from ._util import ElidingLabel, style_spin


def _average_results(results):
    """Average several already-computed single-pair preview results
    (same kind, same x/y grid -- the grid is deterministic from settings
    alone, not from any pair's own data, so this is a plain element-wise
    average, never a resample) into ONE combined field for
    _compute_range's multi-pair case: per-cell nanmean, each pair's own
    `valid` mask turning ITS OWN invalid cells to NaN before averaging --
    the same convention pipeline.combine_dual_planar_pair's own overlap-
    region averaging already uses (see its docstring), applied here
    across PAIRS instead of across CAMERAS. `valid` in the returned dict
    is True only where every component averaged to a real (non-NaN)
    value."""
    first = results[0]

    def stacked(key):
        return np.stack([np.where(r["valid"], r[key], np.nan) for r in results])

    with np.errstate(invalid="ignore"):
        u = np.nanmean(stacked("u"), axis=0)
        v = np.nanmean(stacked("v"), axis=0)
        w = np.nanmean(stacked("w"), axis=0) if first["kind"] == "stereo" else None
    valid = ~np.isnan(u) & ~np.isnan(v)
    if w is not None:
        valid &= ~np.isnan(w)

    return dict(
        kind=first["kind"],
        pair_id=f"{first['pair_id']}..{results[-1]['pair_id']} (avg of {len(results)})",
        x=first["x"], y=first["y"], u=u, v=v, w=w, valid=valid,
        elapsed=sum(r["elapsed"] for r in results),
        n_valid=int(valid.sum()), n_total=int(valid.size),
        n_range=sum(r["n_range"] for r in results),
        n_std=sum(r["n_std"] for r in results),
        units=first["units"],
    )


def _build_engine(backend, frame_shape, correlation, validation):
    factory = get_engine_factory(backend)
    if backend == "cpu":
        settings = {"cpu_settings": to_cpu_settings(correlation, validation)}
    else:
        min_search_size, piv_settings = to_gpu_settings(correlation, validation)
        settings = {"min_search_size": min_search_size, "piv_settings": piv_settings}
    return factory(frame_shape, settings)


class _PreviewWorker(QObject):
    """Runs one preview's COMPUTE off the GUI thread.

    Correlating a single full-resolution pair is not fast: a real
    3008x4096 pair through the default 4-pass schedule takes ~45s on a
    24-core workstation. This used to run inline on the GUI thread (with
    a comment asserting "a single pair is fast enough that a background
    thread wasn't worth the complexity"), which meant the window stopped
    answering Windows' paint/ping messages for that whole time and the OS
    painted it "(Not Responding)" -- reported from real use, and
    reproducible on any full-resolution dataset.

    Only the computation moves here. Figure construction stays on the GUI
    thread (matplotlib's Qt canvas must be built there), so this emits
    plain arrays and the panel renders them in its own slot."""

    finished = Signal(object)   # dict payload -> PreviewPanel._render
    failed = Signal(str)
    # (done, total) pairs -- only meaningful for a multi-pair range preview
    # (_compute_range calls this once per pair); a single-pair preview
    # still uses the plain indeterminate bar, see PreviewPanel._do_preview.
    progress = Signal(int, int)

    def __init__(self, compute_fn):
        super().__init__()
        self._compute_fn = compute_fn

    def run(self):
        try:
            self.finished.emit(self._compute_fn())
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc))


class PreviewPanel(QWidget):
    previewed = Signal(bool)  # emits True on a successful preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self._preview_thread = None
        self._preview_worker = None
        # The last successfully-rendered result payload (see _render) --
        # cached so toggling Vectors can re-render the SAME already-
        # computed field with/without the overlay instead of re-running
        # the whole (potentially ~45s-per-pair) PIV computation just to
        # flip a display option.
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        pair_row = QHBoxLayout()
        pair_row.setSpacing(6)
        pair_row.addWidget(QLabel("Pair:"))
        self.pair_combo = QComboBox()
        self.pair_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.pair_combo.setToolTip("Which pair (by index/id) to preview -- click Refresh pairs after changing the input path/glob/suffixes.")
        pair_row.addWidget(self.pair_combo, 1)
        pair_row.addWidget(QLabel("Pairs:"))
        self.range_count_spin = style_spin(QSpinBox())
        self.range_count_spin.setRange(1, 9999)
        self.range_count_spin.setValue(1)
        self.range_count_spin.setToolTip(
            "How many consecutive pairs, starting at the selected pair above, "
            "to average into ONE velocity field before plotting -- 1 (the "
            "default) previews just the selected pair, unchanged from "
            "before. >1 runs each pair through the same compute path and "
            "averages U/V(/W) across them (per-cell nanmean, each pair's "
            "own valid mask respected), smoothing out per-pair noise. "
            "Clamped to however many pairs are actually available from the "
            "selected one onward.")
        pair_row.addWidget(self.range_count_spin)
        self.refresh_pairs_btn = QPushButton("Refresh pairs")
        self.refresh_pairs_btn.setToolTip("Re-scan the current input settings and repopulate the Pair list above.")
        self.refresh_pairs_btn.clicked.connect(self._refresh_pairs)
        pair_row.addWidget(self.refresh_pairs_btn)

        # Preview lives IN the toolbar rather than as its own full-width row.
        # It used to span the panel, which cost a whole band of vertical space
        # for one button -- space the plot needs far more than the button does.
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setProperty("accent", True)
        self.preview_btn.clicked.connect(self._do_preview)
        pair_row.addWidget(self.preview_btn)
        layout.addLayout(pair_row)

        # Indeterminate ("busy") mode by default -- there's no meaningful
        # percentage for a single preview pair, just a running/not-running
        # state. Switched to a determinate 0..count range for a multi-pair
        # range preview instead (see _do_preview/_on_preview_progress),
        # since there IS real per-pair progress to report there. Hidden
        # except while a preview is in progress.
        # Status and the one remaining plot option share a single thin band.
        # "Plot options" used to be a whole titled group box around one
        # checkbox -- an entire card of chrome for one control.
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.status_label = ElidingLabel("No preview yet.")
        # The stretch goes to the STATUS LABEL, not to a bare spacer. An
        # eliding label declares an Ignored width policy (that is how it agrees
        # to shrink), so a spacer taking the slack instead leaves it at zero
        # width and it elides away to nothing.
        status_row.addWidget(self.status_label, 1)

        # Indeterminate ("busy") by default -- there's no meaningful
        # percentage for a single preview pair, just a running/not-running
        # state. Switched to a determinate 0..count range for a multi-pair
        # range preview instead (see _do_preview/_on_preview_progress),
        # since there IS real per-pair progress to report there. Hidden
        # except while a preview is in progress.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(160)
        status_row.addWidget(self.progress_bar)

        # Starts disabled: it only makes sense once a preview result exists
        # to draw arrows on top of (see _on_preview_finished/_render, which
        # enable it, and _on_vectors_toggled, which re-renders the cached
        # result rather than recomputing).
        self.show_vectors_check = QCheckBox("Vectors")
        self.show_vectors_check.setChecked(False)
        self.show_vectors_check.setEnabled(False)
        self.show_vectors_check.setToolTip(
            "Quiver arrow overlay of the in-plane (u, v) direction, drawn "
            "on top of the magnitude contour. Unavailable until a preview "
            "has actually rendered -- toggling it afterward re-renders the "
            "same already-computed result, it doesn't re-run the PIV "
            "computation.")
        self.show_vectors_check.toggled.connect(self._on_vectors_toggled)
        status_row.addWidget(self.show_vectors_check)
        layout.addLayout(status_row)

        # THE PLOT GETS THE REMAINING SPACE. This used to be an unstretched
        # layout followed by addStretch(1), so the spare height went to the
        # stretch and the canvas was pinned to its minimum size hint --
        # leaving most of the window empty grey while the plot sat small in
        # the corner.
        self.plot_area = QFrame()
        self.plot_area.setObjectName("plotArea")
        self.plot_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_container = QVBoxLayout(self.plot_area)
        self.canvas_container.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("Select a pair and press Preview to render a velocity field.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("plotPlaceholder")
        self.canvas_container.addWidget(self.placeholder)
        layout.addWidget(self.plot_area, stretch=1)

    def _on_vectors_toggled(self, _checked):
        if self._last_result is not None:
            self._render(self._last_result)

    def _list_pair_ids(self, project):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            # a .set's pair ids are index-based off the same underlying
            # dataset for both planar and stereo -- one listing works for
            # either mode.
            return list_pair_ids_from_set(set_paths[0], project.multiset_index)
        if project.mode == "stereo":
            return list_pair_ids_stereo_from_loose_files(
                project.input_path, project.loose_glob, project.suffix_cam0, project.suffix_cam1)
        return list_pair_ids_from_loose_files(
            project.input_path, project.loose_glob, project.suffix_a, project.suffix_b)

    def _refresh_pairs(self):
        main_window = self.window()
        project = main_window.project_panel.get_project_settings()
        try:
            pair_ids = self._list_pair_ids(project)
        except Exception as e:
            self.status_label.setText(f"Couldn't list pairs: {e}")
            return
        self.pair_combo.clear()
        self.pair_combo.addItems(pair_ids)
        if pair_ids:
            self.pair_combo.setCurrentIndex(0)
            self.status_label.setText(f"{len(pair_ids)} pair(s) found.")
        else:
            self.status_label.setText("No pairs found for the current input settings.")

    def _first_pair_planar(self, project, index):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return get_pair_from_set(set_paths[0], index, project.multiset_index)
        return get_pair_from_loose_files(
            project.input_path, index, project.loose_glob, project.suffix_a, project.suffix_b)

    def _first_pair_stereo(self, project, index):
        if project.input_mode == "set":
            set_paths, _ = resolve_set_paths(project.input_path)
            return get_stereo_from_set(set_paths[0], index, project.multiset_index, project.stereo_frame_order)
        return get_stereo_from_loose_files(
            project.input_path, index, project.loose_glob,
            project.suffix_cam0, project.suffix_cam1, project.stereo_frame_order)

    def _first_pair_dual_planar(self, project, index):
        # .set-mode-only -- see cli.main/pipeline_worker's identical
        # restriction, a SideBySide2D combined 4-frame buffer has no
        # loose-file equivalent in this app.
        set_paths, _ = resolve_set_paths(project.input_path)
        return get_dual_planar_from_set(set_paths[0], index, project.multiset_index)

    def _set_canvas(self, fig):
        # setParent(None) only detaches the OLD canvas WIDGET from Qt's
        # layout -- it does nothing to the matplotlib Figure it wraps.
        # pyplot keeps every Figure that was never explicitly plt.close()d
        # alive forever in its own global registry (Gcf), regardless of Qt
        # parentage/garbage collection, so every preview run (now also
        # every range-preview average, and every Vectors-toggle re-render,
        # see _render/_on_vectors_toggled) that built a new figure here
        # leaked the previous one -- a real, unbounded memory leak (a full-
        # resolution preview's contourf/quiver artists carry real array
        # data, not just a few bytes), confirmed by watching
        # matplotlib.pyplot.get_fignums() grow across repeated previews
        # before this fix and stay flat after it.
        if self.canvas is not None:
            old_fig = self.canvas.figure
            self.canvas.setParent(None)
            plt.close(old_fig)
        if self.placeholder is not None:
            self.placeholder.setParent(None)
            self.placeholder = None
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # FigureCanvasQTAgg resizes the figure to match this widget, so a
        # degenerate widget size makes a degenerate figure -- matplotlib's
        # constrained layout then gives up with "axes sizes collapsed to zero"
        # and the plot renders without its layout. Reachable whenever the
        # canvas is laid out before it has a real size.
        self.canvas.setMinimumSize(220, 165)
        self.canvas_container.addWidget(self.canvas)

    def _do_preview(self):
        main_window = self.window()
        if self.pair_combo.count() == 0:
            self._refresh_pairs()
        if self.pair_combo.count() == 0:
            self.status_label.setText("Preview failed: no pairs found for the current input settings.")
            self.previewed.emit(False)
            return
        index = self.pair_combo.currentIndex()
        # Clamped to however many pairs are actually available from the
        # selected one onward -- never read past the end of the list just
        # because range_count_spin was left at a stale, too-large value.
        count = min(self.range_count_spin.value(), self.pair_combo.count() - index)

        # Settings are read HERE, on the GUI thread -- they touch widgets,
        # which is not safe from a worker thread. Only the correlation
        # itself is handed off (see _PreviewWorker for why it has to be).
        try:
            project = main_window.project_panel.get_project_settings()
            preprocess = main_window.settings_panel.get_preprocess_settings()
            correlation = main_window.settings_panel.get_correlation_settings()
            validation = main_window.settings_panel.get_validation_settings()
            post = main_window.settings_panel.get_postprocess_settings()
            calibration = main_window.project_panel.get_calibration_settings()
            stereo_settings = main_window.calibration_panel.get_settings() if project.mode == "stereo" else None
            dual_planar_settings = (main_window.project_panel.get_dual_planar_settings()
                                     if project.dual_camera else None)
        except Exception as e:
            self.status_label.setText(f"Preview failed: {e}")
            self.previewed.emit(False)
            return

        # Disabling the button also blocks double-clicks from queuing a
        # second preview while one is still running.
        self.preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        if count > 1:
            # Real "N/M pairs done" progress -- there's genuine multi-step
            # work to report for a range preview (unlike a single pair,
            # which keeps the plain indeterminate bar below).
            self.progress_bar.setRange(0, count)
            self.progress_bar.setValue(0)
        self.status_label.setText("Running preview...")

        self._preview_thread = QThread()
        self._preview_worker = _PreviewWorker(None)
        progress_cb = self._preview_worker.progress.emit

        def compute():
            return self._compute_range(project, preprocess, correlation, validation, post, calibration,
                                        stereo_settings, dual_planar_settings, index, count, progress_cb)

        self._preview_worker._compute_fn = compute
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.progress.connect(self._on_preview_progress)
        self._preview_thread.start()

    def _on_preview_progress(self, done, total):
        # total==1 (the single-pair case) keeps the plain indeterminate
        # bar untouched -- see _do_preview, which only switches the bar
        # to a determinate range when count > 1.
        if total > 1:
            self.progress_bar.setValue(done)
            self.status_label.setText(f"Running preview... ({done}/{total} pairs done)")

    def _teardown_preview_thread(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # back to indeterminate, ready for next time
        self.preview_btn.setEnabled(True)
        thread = getattr(self, "_preview_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait()
            self._preview_thread = None
            self._preview_worker = None

    def _on_preview_failed(self, message):
        self._teardown_preview_thread()
        self._clear_canvas("Preview failed. Fix the problem above and try again.")
        self.status_label.setText(f"Preview failed: {message}")
        self.previewed.emit(False)

    def _clear_canvas(self, message):
        """Drop any rendered field and show `message` in its place.

        A failed preview used to leave the PREVIOUS pair's field on screen,
        with only the status line saying otherwise -- so the plot showed one
        pair while the controls described another, and toggling Vectors
        re-rendered the stale result as if it were current."""
        if self.canvas is not None:
            old_fig = self.canvas.figure
            self.canvas.setParent(None)
            plt.close(old_fig)
            self.canvas = None
        self._last_result = None
        self.show_vectors_check.setEnabled(False)
        if self.placeholder is None:
            self.placeholder = QLabel()
            self.placeholder.setAlignment(Qt.AlignCenter)
            self.placeholder.setObjectName("plotPlaceholder")
            self.canvas_container.addWidget(self.placeholder)
        self.placeholder.setText(message)
        self.placeholder.setVisible(True)

    def _on_preview_finished(self, result):
        self._teardown_preview_thread()
        try:
            self._render(result)
        except Exception as e:
            self.status_label.setText(f"Preview failed while rendering: {e}")
            self.previewed.emit(False)
            raise
        self.previewed.emit(True)

    def _render(self, r):
        """Build the figure and status line from a worker payload (or the
        cached _last_result, when only the Vectors toggle changed -- see
        _on_vectors_toggled). Runs on the GUI thread (matplotlib's Qt
        canvas requires it). Enables the Vectors checkbox -- it starts
        disabled (see the Vectors checkbox in the status row) since there's nothing to
        overlay vectors onto before a preview has actually rendered."""
        self.status_label.setText(
            f"Pair '{r['pair_id']}': {r['elapsed']:.3f}s, {r['n_valid']}/{r['n_total']} valid "
            f"(range/residual rejected {r['n_range']}, std-dev rejected {r['n_std']})"
        )
        self._last_result = r
        self.show_vectors_check.setEnabled(True)
        fig = make_preview_figure(
            "stereo" if r["kind"] == "stereo" else "planar",
            r["x"], r["y"], r["u"], r["v"], r["valid"], w=r.get("w"), units=r["units"],
            title=f"Preview -- {r['pair_id']}", show_vectors=self.show_vectors_check.isChecked())
        self._set_canvas(fig)

    def _compute_planar(self, project, preprocess, correlation, validation, post, calibration, index):
        pair_id, frame_a, frame_b = self._first_pair_planar(project, index)
        frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, preprocess)
        engine, x, y = _build_engine(project.backend, frame_a.shape, correlation, validation)

        u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post.for_pipeline())
        u, v = apply_calibration(u, v, calibration.pixel_pitch_mm, calibration.frame_dt_s)
        # apply_calibration is a no-op (stays px/frame) unless BOTH
        # pixel_pitch_mm and frame_dt_s are set -- see its own docstring.
        units = ("m/s" if calibration.pixel_pitch_mm is not None and calibration.frame_dt_s is not None
                 else "px/frame")

        return dict(kind="planar", pair_id=pair_id, x=x, y=y, u=u, v=v, valid=valid,
                    elapsed=elapsed, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=rejects["range_residual"], n_std=rejects["std_dev"], units=units)

    def _compute_dual_planar(self, project, preprocess, correlation, validation, post, calibration,
                              dual_planar_settings, index):
        pair_id, fa0, fb0, fa1, fb1 = self._first_pair_dual_planar(project, index)
        fa0, fb0 = apply_preprocess_pair(fa0, fb0, preprocess)
        fa1, fb1 = apply_preprocess_pair(fa1, fb1, preprocess)

        # RAW (row-down, unflipped) engine.coords -- NOT the display-
        # flipped x, y _build_engine's factory returns -- see
        # pipeline.combine_dual_planar_pair's docstring for why.
        engine0, _x0f, _y0f = _build_engine(project.backend, fa0.shape, correlation, validation)
        u0, v0, valid0, elapsed0, r0 = pipeline.process_frames(engine0, fa0, fb0, post.for_pipeline())
        x0, y0 = engine0.coords

        engine1, _x1f, _y1f = _build_engine(project.backend, fa1.shape, correlation, validation)
        u1, v1, valid1, elapsed1, r1 = pipeline.process_frames(engine1, fa1, fb1, post.for_pipeline())
        x1, y1 = engine1.coords

        X, Y, U, V, valid = pipeline.combine_dual_planar_pair(
            (u0, v0, x0, y0, valid0), (u1, v1, x1, y1, valid1),
            dual_planar_settings, calibration.frame_dt_s)
        # combine_dual_planar_pair always converts position/velocity to mm
        # (mandatory just to place two cameras on one shared canvas) --
        # frame_dt_s=None only means it stops short of the final m/s
        # divide, leaving mm/frame, NEVER raw px/frame -- see its own
        # docstring.
        units = "m/s" if calibration.frame_dt_s is not None else "mm/frame"

        return dict(kind="planar", pair_id=pair_id, x=X, y=Y, u=U, v=V, valid=valid,
                    elapsed=elapsed0 + elapsed1, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=r0["range_residual"] + r1["range_residual"],
                    n_std=r0["std_dev"] + r1["std_dev"], units=units)

    def _compute_stereo(self, project, preprocess, correlation, validation, post, calibration,
                         stereo_settings, index):
        cam0, cam1 = build_stereo_cameras(stereo_settings)

        pair_id, fa0, fb0, fa1, fb1 = self._first_pair_stereo(project, index)
        fa0, fb0 = apply_preprocess_pair(fa0, fb0, preprocess)
        fa1, fb1 = apply_preprocess_pair(fa1, fb1, preprocess)
        dw_a0 = cam0.dewarp_image(fa0, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_b0 = cam0.dewarp_image(fb0, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_a1 = cam1.dewarp_image(fa1, stereo_settings.world_shape, stereo_settings.dewarp_order)
        dw_b1 = cam1.dewarp_image(fb1, stereo_settings.world_shape, stereo_settings.dewarp_order)

        engine0, x, y = _build_engine(project.backend, dw_a0.shape, correlation, validation)
        engine1, _, _ = _build_engine(project.backend, dw_a1.shape, correlation, validation)

        # cam0/cam1 each genuinely see this world-grid point (x, y) AND
        # have a trustworthy (non-extrapolated) calibration fit there --
        # see calibration.camera_mapping.stereo_fov_valid's own docstring.
        # `y` here is _build_engine's DISPLAY-flipped ("y increases
        # upward") coordinate (engines.cpu_engine.init_cpu_processor /
        # gpu_engine.init_gpu_processor: y = frame_shape[0]*scaling_par -
        # y_raw, scaling_par==1.0 pre-calibration) -- world_to_raw's own
        # grid (_ensure_grid's np.mgrid) is row-down/UNflipped, so this
        # needs y un-flipped back first or every point near the canvas's
        # top/bottom edge gets checked against the wrong row.
        y_row_down = stereo_settings.world_shape[0] - y
        fov_valid = stereo_fov_valid(cam0, cam1, x, y_row_down)
        # Per-pixel triangulation angles, derived from this project's own
        # calibration on the same grid stereo_fov_valid just used. A single
        # global angle per camera measurably corrupts U and V (see
        # config.schema.StereoSettings.alpha1_deg); StereoSettings' scalar
        # fields survive only as an explicit override.
        angles = stereo_angles_for(stereo_settings, cam0, cam1, x, y_row_down)
        # process_stereo_pair validates the COMBINED/triangulated field
        # once (not each camera's raw 2D field independently, then
        # intersected) -- see its own docstring for why: on real data,
        # per-camera-then-intersect compounded two individually reasonable
        # per-camera rejection rates into a combined density meaningfully
        # worse than DaVis's own real final density.
        U, V, W, valid, elapsed, r = pipeline.process_stereo_pair(
            engine0, engine1, dw_a0, dw_b0, dw_a1, dw_b1, angles,
            stereo_settings.world_scale_px_per_mm, calibration.frame_dt_s,
            fov_valid, post.for_pipeline(), x, y)
        # combine_stereo_pair always converts to mm (dividing by
        # world_scale_px_per_mm happens unconditionally); frame_dt_s=None
        # only means it stops short of the final /1000 divide to m/s,
        # leaving mm/frame, NEVER raw px/frame -- see its own docstring
        # (and 3fa6c8f, which fixed this function silently skipping the
        # /1000 step when frame_dt_s WAS given).
        units = "m/s" if calibration.frame_dt_s is not None else "mm/frame"

        return dict(kind="stereo", pair_id=pair_id, x=x, y=y, u=U, v=V, w=W, valid=valid,
                    elapsed=elapsed, n_valid=int(valid.sum()), n_total=int(valid.size),
                    n_range=r["range_residual"], n_std=r["std_dev"], n_group=r["small_groups"],
                    n_fov=int(fov_valid.sum()), units=units)

    def _compute_range(self, project, preprocess, correlation, validation, post, calibration,
                        stereo_settings, dual_planar_settings, start_index, count, progress_cb):
        """Run `count` consecutive pairs starting at start_index through
        whichever single-pair compute path matches this project's mode
        (_compute_stereo/_compute_dual_planar/_compute_planar, reused
        as-is -- not duplicated), then, for count > 1, average their
        results into ONE combined field via _average_results. count=1 is
        the previous, unchanged single-pair preview: returns that one
        pair's own result untouched, no averaging overhead for the common
        case. progress_cb(done, total) is called after each pair
        finishes so the GUI thread can show real per-pair progress for a
        multi-pair range (see PreviewPanel._on_preview_progress) --
        harmless for the single-pair case, which just ignores it."""
        results = []
        for offset in range(count):
            index = start_index + offset
            if project.mode == "stereo":
                r = self._compute_stereo(project, preprocess, correlation, validation, post,
                                          calibration, stereo_settings, index)
            elif project.dual_camera:
                r = self._compute_dual_planar(project, preprocess, correlation, validation, post,
                                               calibration, dual_planar_settings, index)
            else:
                r = self._compute_planar(project, preprocess, correlation, validation, post,
                                          calibration, index)
            results.append(r)
            progress_cb(offset + 1, count)
        return results[0] if count == 1 else _average_results(results)
