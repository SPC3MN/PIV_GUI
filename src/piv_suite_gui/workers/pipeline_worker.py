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

import numpy as np
from PySide6.QtCore import QObject, Signal

from piv_suite.calibration.camera_mapping import CameraMapping
from piv_suite.config.legacy import to_cpu_settings, to_gpu_settings
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.davis_set import (
    iter_pairs_from_set, iter_stereo_from_set, resolve_set_paths, set_label,
)
from piv_suite.io.loose_files import iter_pairs_from_loose_files, iter_stereo_from_loose_files
from piv_suite.plotting.planar import plot_and_save_planar
from piv_suite.plotting.stereo import plot_and_save_stereo
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration


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
                if correlation.use_tiling:
                    init_fn = _make_tiled_init_fn(backend, correlation, validation)
                    margin = _tile_margin(backend, correlation, validation)
                    x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
                        frame_a, frame_b, post, init_fn, correlation.n_tiles_y, correlation.n_tiles_x,
                        margin, free_pools_fn=lambda: _free_gpu(backend),
                    )
                else:
                    if engine is None:
                        engine, x, y = _build_engine(backend, frame_a.shape, correlation, validation)
                    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post)

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
            except Exception as e:
                self.error.emit(pair_id, str(e))

        if backend == "gpu" and engine is not None:
            from piv_suite.engines.gpu_engine import free_gpu_pools
            del engine
            free_gpu_pools()

        return summary_rows, cancelled

    def _run_camera(self, frame_a, frame_b, cfg):
        backend = cfg.project.backend
        engine, x, y = _build_engine(backend, frame_a.shape, cfg.correlation, cfg.validation)
        u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, cfg.postprocess.for_pipeline())
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
                dw_a0 = cam0.dewarp_image(fa0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_b0 = cam0.dewarp_image(fb0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_a1 = cam1.dewarp_image(fa1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
                dw_b1 = cam1.dewarp_image(fb1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)

                u1, v1, valid1, elapsed1, x, y, r1 = self._run_camera(dw_a0, dw_b0, cfg)
                u2, v2, valid2, elapsed2, _, _, r2 = self._run_camera(dw_a1, dw_b1, cfg)
                valid = valid1 & valid2
                elapsed = elapsed1 + elapsed2

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
