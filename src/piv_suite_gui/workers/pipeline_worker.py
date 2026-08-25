"""QObject batch-processing worker, moved to a QThread by run_panel.py --
runs the same underlying pieces as piv_suite.cli.main (io iterators,
engines.registry, processing.pipeline, plotting), but reports progress via
Qt signals instead of print()/sys.exit(), and checks a cancellation flag
between pairs instead of blocking on a terminal y/N prompt. Supports both
planar and stereo batch modes (cfg.project.mode), mirroring
cli.main.process_pairs_planar/process_pairs_stereo.
"""

import csv
import os
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, Signal

from piv_suite.calibration.camera_mapping import CameraMapping
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.engines.base import EngineCancelled
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.davis_set import (
    iter_pairs_from_set, iter_stereo_from_set, resolve_set_paths, set_label,
)
from piv_suite.io.loose_files import iter_pairs_from_loose_files, iter_stereo_from_loose_files
from piv_suite.perf.autotune import recommended_workers
from piv_suite.plotting.planar import plot_and_save_planar
from piv_suite.plotting.stereo import plot_and_save_stereo
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration
from piv_suite.processing.preprocess import apply_preprocess_pair


def _build_engine(backend, frame_shape, correlation, validation):
    factory = get_engine_factory(backend)
    if backend == "cpu":
        settings = {"cpu_settings": to_cpu_settings(correlation, validation)}
    else:
        min_search_size, piv_settings = to_gpu_settings(correlation, validation)
        settings = {"min_search_size": min_search_size, "piv_settings": piv_settings}
    return factory(frame_shape, settings)


def _make_tiled_init_fn(backend, correlation, validation):
    if backend != "gpu":
        raise ValueError("tiling is only supported on the gpu backend")
    from piv_suite.engines.gpu_engine import _init_gpu_processor_raw
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    return lambda shape: _init_gpu_processor_raw(shape, min_search_size, piv_settings)


def _tile_margin(backend, correlation, validation):
    from piv_suite.engines.gpu_engine import default_tile_margin
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    return correlation.tile_margin_px or default_tile_margin(min_search_size, piv_settings)


def _free_gpu(backend):
    if backend == "gpu":
        from piv_suite.engines.gpu_engine import free_gpu_pools
        free_gpu_pools()


class PipelineWorker(QObject):
    pair_started = Signal(str)
    pair_finished = Signal(str, dict)   # pair_id, {elapsed, n_valid, n_total, ...}
    progress = Signal(int, int)         # current, total (total may be 0 if unknown up front)
    error = Signal(str, str)            # pair_id, message (non-fatal -- batch continues)
    log = Signal(str)
    finished = Signal(bool)             # cancelled?

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def force_stop(self):
        """Strongest cancellation currently available -- a deliberate SEAM
        for a sibling change (process-pool cancellation in
        processing.parallel_planar / a new processing.parallel_stereo,
        wired through _process_set_planar_parallel / an eventual
        _process_set_stereo_parallel) to extend, not a finished hard-kill
        of everything this worker might be doing.

        Right now this just calls cancel(): for the SERIAL loop (both
        planar and stereo, and the tiled-GPU path), that IS the strongest
        safe option available -- self._cancel_event is polled between
        PAIRS (the outer loops below) same as before, but is now ALSO
        forwarded as `cancel_check` into pipeline.process_frames /
        process_frames_tiled, which poll it between multi-pass iterations
        (engines.cpu_engine.CPUPIVProcess) or between tiles
        (engines.gpu_engine.run_tiled) -- see EngineCancelled's docstring
        (engines/base.py). That bounds worst-case cancellation latency to
        one pass or one tile instead of one whole pair, without resorting
        to killing a thread mid-BLAS/FFT call (unsafe -- deliberately not
        attempted, see this module's docstring history / the task that
        added this method).

        For the PARALLEL executor path (n_workers > 1), setting the event
        only stops NEW submissions -- pairs already dispatched to a
        worker process finish naturally (see
        processing.parallel_planar.run_planar_batch_parallel's own
        docstring). A sibling change is expected to extend THIS method
        (not add a second, competing one) to also reach into a live
        ProcessPoolExecutor and terminate its in-flight worker processes
        once that lands. Callers (run_panel.py's Cancel button,
        main_window.py's closeEvent) call force_stop(), not cancel()
        directly, so they pick up that strengthening automatically the
        moment it exists here -- no caller-side change needed."""
        self.cancel()

    def run(self):
        cfg = self.config
        stereo = cfg.project.mode == "stereo"
        cancelled = False
        try:
            os.makedirs(cfg.project.output_dir, exist_ok=True)
            if cfg.project.input_mode == "set":
                set_paths, is_batch = resolve_set_paths(cfg.project.input_path)
            else:
                set_paths, is_batch = [cfg.project.input_path], False

            angles = cam0 = cam1 = None
            if stereo:
                cam0 = CameraMapping(
                    cfg.stereo.cam0_mapping.x0, cfg.stereo.cam0_mapping.x_span,
                    cfg.stereo.cam0_mapping.y0, cfg.stereo.cam0_mapping.y_span,
                    cfg.stereo.cam0_mapping.dx_coefs, cfg.stereo.cam0_mapping.dy_coefs,
                    cfg.stereo.cam0_mapping.name,
                )
                cam1 = CameraMapping(
                    cfg.stereo.cam1_mapping.x0, cfg.stereo.cam1_mapping.x_span,
                    cfg.stereo.cam1_mapping.y0, cfg.stereo.cam1_mapping.y_span,
                    cfg.stereo.cam1_mapping.dx_coefs, cfg.stereo.cam1_mapping.dy_coefs,
                    cfg.stereo.cam1_mapping.name,
                )
                angles = (np.deg2rad(cfg.stereo.alpha1_deg), np.deg2rad(cfg.stereo.alpha2_deg),
                          np.deg2rad(cfg.stereo.beta1_deg), np.deg2rad(cfg.stereo.beta2_deg))

            grand_total = 0
            for set_path in set_paths:
                if self._cancel_event.is_set():
                    cancelled = True
                    break

                output_dir = (os.path.join(cfg.project.output_dir, set_label(set_path))
                               if is_batch else cfg.project.output_dir)
                os.makedirs(output_dir, exist_ok=True)

                if stereo:
                    if cfg.project.input_mode == "set":
                        self.log.emit(f"[info] processing set '{set_path}'")
                        pair_source = iter_stereo_from_set(set_path, cfg.project.multiset_index,
                                                            cfg.project.stereo_frame_order)
                    else:
                        pair_source = iter_stereo_from_loose_files(
                            cfg.project.input_path, cfg.project.loose_glob,
                            cfg.project.suffix_cam0, cfg.project.suffix_cam1, cfg.project.stereo_frame_order)
                    summary_rows, batch_cancelled = self._process_set_stereo(
                        pair_source, cfg, output_dir, cam0, cam1, angles)
                else:
                    if cfg.project.input_mode == "set":
                        self.log.emit(f"[info] processing set '{set_path}'")
                        pair_source = iter_pairs_from_set(set_path, cfg.project.multiset_index)
                    else:
                        pair_source = iter_pairs_from_loose_files(
                            cfg.project.input_path, cfg.project.loose_glob,
                            cfg.project.suffix_a, cfg.project.suffix_b)
                    summary_rows, batch_cancelled = self._process_set_planar(pair_source, cfg, output_dir)

                grand_total += len(summary_rows)
                self._write_summary(summary_rows, output_dir, cfg, stereo=stereo)
                if batch_cancelled:
                    cancelled = True
                    break

            if grand_total == 0 and not cancelled:
                self.log.emit("[warn] no pairs were processed -- check input_mode/input_path")
        except Exception as e:
            self.log.emit(f"[error] batch run failed: {e}")
            raise
        finally:
            self.finished.emit(cancelled)

    def _process_set_planar(self, pair_source, cfg, output_dir):
        backend = cfg.project.backend
        correlation, validation = cfg.correlation, cfg.validation

        # Tier 3: process-level parallelism across independent pairs,
        # planar CPU only -- see processing.parallel_planar's module
        # docstring. n_workers<=1 always falls through to the unmodified
        # serial loop below (a hard requirement, not just an
        # optimization -- see that module's docstring for why).
        if backend == "cpu" and not correlation.use_tiling:
            n_workers = recommended_workers(cfg.performance.n_workers)
            auto_note = "auto" if cfg.performance.n_workers is None else "user override"
            self.log.emit(f"[info] planar CPU batch: {n_workers} worker process(es) ({auto_note})")
            if n_workers > 1:
                return self._process_set_planar_parallel(pair_source, cfg, output_dir, n_workers)

        post = cfg.postprocess.for_pipeline()
        engine = x = y = None
        summary_rows = []
        cancelled = False

        for pair_id, frame_a, frame_b in pair_source:
            if self._cancel_event.is_set():
                cancelled = True
                break

            self.pair_started.emit(pair_id)
            try:
                t_pre0 = time.time()
                frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, cfg.preprocess)
                t_pre = time.time() - t_pre0
                if correlation.use_tiling:
                    init_fn = _make_tiled_init_fn(backend, correlation, validation)
                    margin = _tile_margin(backend, correlation, validation)
                    t_post0 = time.time()
                    x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
                        frame_a, frame_b, post, init_fn, correlation.n_tiles_y, correlation.n_tiles_x,
                        margin, free_pools_fn=lambda: _free_gpu(backend),
                        cancel_check=self._cancel_event.is_set,
                    )
                    t_post = time.time() - t_post0 - elapsed
                else:
                    if engine is None:
                        engine, x, y = _build_engine(backend, frame_a.shape, correlation, validation)
                    t_post0 = time.time()
                    u, v, valid, elapsed, rejects = pipeline.process_frames(
                        engine, frame_a, frame_b, post, cancel_check=self._cancel_event.is_set)
                    t_post = time.time() - t_post0 - elapsed

                if cfg.output.verbose:
                    self.log.emit(f"[timing] {pair_id}: preprocess={t_pre:.3f}s "
                                   f"correlation={elapsed:.3f}s postprocess={t_post:.3f}s "
                                   f"total={t_pre + elapsed + t_post:.3f}s")

                u, v = apply_calibration(u, v, cfg.calibration.pixel_pitch_mm, cfg.calibration.frame_dt_s)
                n_valid, n_total = int(valid.sum()), int(valid.size)

                if cfg.output.save_npz:
                    np.savez(os.path.join(output_dir, f"{pair_id}_velocity.npz"),
                              x=x, y=y, u=u, v=v, valid=valid)
                if cfg.output.save_plot:
                    plot_and_save_planar(x, y, u, v, valid,
                                          os.path.join(output_dir, f"{pair_id}_quiver.png"),
                                          title=f"PIV velocity field -- {pair_id}",
                                          quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi)

                row = (pair_id, elapsed, n_valid, n_total, rejects["range_residual"], rejects["std_dev"])
                summary_rows.append(row)
                self.pair_finished.emit(pair_id, {
                    "elapsed": elapsed, "n_valid": n_valid, "n_total": n_total,
                    "n_rejected_range_residual": rejects["range_residual"],
                    "n_rejected_std_dev": rejects["std_dev"],
                })
                self.progress.emit(len(summary_rows), 0)
            except EngineCancelled:
                # Fired mid-pair (between passes/tiles, see cpu_engine.py/
                # gpu_engine.py) rather than at this loop's own top-of-
                # iteration check -- this pair's partial result is
                # discarded (no summary row, no pair_finished signal,
                # matching the outer per-pair check's behavior when it
                # fires BEFORE a pair even starts), and the batch stops
                # the same way an between-pairs cancel does.
                cancelled = True
                break
            except Exception as e:
                self.error.emit(pair_id, str(e))

        if backend == "gpu" and engine is not None:
            from piv_suite.engines.gpu_engine import free_gpu_pools
            del engine
            free_gpu_pools()

        return summary_rows, cancelled

    def _process_set_planar_parallel(self, pair_source, cfg, output_dir, n_workers):
        """The n_workers > 1 branch of _process_set_planar -- same Qt
        signal contract (pair_started/pair_finished/progress/error) as
        the serial loop, just driven by
        processing.parallel_planar.run_planar_batch_parallel's callbacks
        instead of an inline try/except per pair. Progress counts
        COMPLETED pairs (not submission order, since workers can finish
        out of order) -- matches the serial loop's `len(summary_rows)`
        meaning (how many are done so far), just not necessarily counting
        up in pair_id order along the way."""
        from piv_suite.processing.parallel_planar import run_planar_batch_parallel

        finished_count = [0]

        def _on_finished(pair_id, result):
            if cfg.output.verbose:
                self.log.emit(
                    f"[timing] {pair_id}: preprocess={result['t_pre']:.3f}s "
                    f"correlation={result['elapsed']:.3f}s postprocess={result['t_post']:.3f}s "
                    f"total={result['t_pre'] + result['elapsed'] + result['t_post']:.3f}s")
            self.pair_finished.emit(pair_id, {
                "elapsed": result["elapsed"], "n_valid": result["n_valid"], "n_total": result["n_total"],
                "n_rejected_range_residual": result["n_rejected_range_residual"],
                "n_rejected_std_dev": result["n_rejected_std_dev"],
            })
            finished_count[0] += 1
            self.progress.emit(finished_count[0], 0)

        def _on_error(pair_id, exc):
            self.error.emit(pair_id, str(exc))

        results, cancelled = run_planar_batch_parallel(
            pair_source, cfg, output_dir, n_workers,
            on_pair_started=self.pair_started.emit, on_pair_finished=_on_finished,
            on_pair_error=_on_error, cancel_check=self._cancel_event.is_set,
        )
        summary_rows = [
            (r["pair_id"], r["elapsed"], r["n_valid"], r["n_total"],
             r["n_rejected_range_residual"], r["n_rejected_std_dev"])
            for r in results
        ]
        return summary_rows, cancelled

    def _run_camera(self, frame_a, frame_b, cfg):
        backend = cfg.project.backend
        engine, x, y = _build_engine(backend, frame_a.shape, cfg.correlation, cfg.validation)
        u, v, valid, elapsed, rejects = pipeline.process_frames(
            engine, frame_a, frame_b, cfg.postprocess.for_pipeline(),
            cancel_check=self._cancel_event.is_set)
        del engine
        if backend == "gpu":
            from piv_suite.engines.gpu_engine import free_gpu_pools
            free_gpu_pools()
        return u, v, valid, elapsed, x, y, rejects

    def _process_set_stereo(self, pair_source, cfg, output_dir, cam0, cam1, angles):
        summary_rows = []
        cancelled = False

        for pair_id, fa0, fb0, fa1, fb1 in pair_source:
            if self._cancel_event.is_set():
                cancelled = True
                break

            self.pair_started.emit(pair_id)
            try:
                t_pre0 = time.time()
                fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
                fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)
                dw_a0 = cam0.dewarp_image(fa0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_b0 = cam0.dewarp_image(fb0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_a1 = cam1.dewarp_image(fa1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_b1 = cam1.dewarp_image(fb1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                t_pre = time.time() - t_pre0

                t_run0 = time.time()
                u1, v1, valid1, elapsed1, x, y, r1 = self._run_camera(dw_a0, dw_b0, cfg)
                u2, v2, valid2, elapsed2, _, _, r2 = self._run_camera(dw_a1, dw_b1, cfg)
                valid = valid1 & valid2
                elapsed = elapsed1 + elapsed2
                t_post = time.time() - t_run0 - elapsed

                if cfg.output.verbose:
                    self.log.emit(f"[timing] {pair_id}: preprocess/dewarp={t_pre:.3f}s "
                                   f"correlation={elapsed:.3f}s postprocess={t_post:.3f}s "
                                   f"total={t_pre + elapsed + t_post:.3f}s")

                U, V, W = pipeline.combine_stereo_pair(
                    u1, v1, u2, v2, angles, cfg.stereo.world_scale_px_per_mm, cfg.calibration.frame_dt_s)
                U = np.where(valid, U, np.nan)
                V = np.where(valid, V, np.nan)
                W = np.where(valid, W, np.nan)
                n_valid, n_total = int(valid.sum()), int(valid.size)

                if cfg.output.save_npz:
                    np.savez(os.path.join(output_dir, f"{pair_id}_stereo_velocity.npz"),
                              x=x, y=y, U=U, V=V, W=W, valid=valid)
                if cfg.output.save_plot:
                    plot_and_save_stereo(x, y, U, V, W, valid,
                                          os.path.join(output_dir, f"{pair_id}_stereo_quiver.png"),
                                          title=f"Stereo PIV -- {pair_id}",
                                          quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi)

                n_range = r1["range_residual"] + r2["range_residual"]
                n_std = r1["std_dev"] + r2["std_dev"]
                row = (pair_id, elapsed, n_valid, n_total, n_range, n_std)
                summary_rows.append(row)
                self.pair_finished.emit(pair_id, {
                    "elapsed": elapsed, "n_valid": n_valid, "n_total": n_total,
                    "n_rejected_range_residual": n_range, "n_rejected_std_dev": n_std,
                })
                self.progress.emit(len(summary_rows), 0)
            except EngineCancelled:
                # See the matching except clause in _process_set_planar --
                # same contract, fired mid-pair (between passes, from
                # either camera's _run_camera call) rather than at this
                # loop's own top-of-iteration check.
                cancelled = True
                break
            except Exception as e:
                self.error.emit(pair_id, str(e))

        return summary_rows, cancelled

    def _write_summary(self, summary_rows, output_dir, cfg, stereo=False):
        if not summary_rows or not cfg.output.save_summary_csv:
            return
        name = "stereo_processing_summary.csv" if stereo else "processing_summary.csv"
        csv_path = os.path.join(output_dir, name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total",
                              "n_rejected_range_residual", "n_rejected_std_dev"])
            writer.writerows(summary_rows)
        self.log.emit(f"Summary written to {csv_path}")
