"""Unified CLI -- one entry point (`piv-suite`) reproducing all four
original scripts' (Planar.py, Stereo-PIV.py, CPU_Planar_Processing.py,
CPU_Stereo_Processing.py) behavior, selected via --backend/--mode instead
of being four separate files. This is the Phase-1 parity target: run
against the same inputs as an original script and its `.npz`/CSV output
should match within floating-point tolerance (see tests/golden).

Usage:
    piv-suite my_project.pivproj --backend gpu --mode stereo
    piv-suite my_project.pivproj --backend cpu --mode planar --input-path ./data
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from ..calibration.camera_mapping import CameraMapping
from ..config.io import ProjectConfig, load_project, save_project
from ..config.legacy import to_cpu_settings, to_gpu_settings
from ..engines.registry import get_engine_factory
from ..io.davis_set import iter_pairs_from_set, iter_stereo_from_set, resolve_set_paths, set_label
from ..io.loose_files import iter_pairs_from_loose_files, iter_stereo_from_loose_files
from ..plotting.planar import plot_and_save_planar
from ..plotting.preview import preview_first_snapshot_cli
from ..plotting.stereo import plot_and_save_stereo
from ..processing import pipeline
from ..processing.postprocess import apply_calibration
from ..processing.preprocess import apply_preprocess_pair


def _engine_settings(backend, correlation, validation):
    if backend == "cpu":
        return {"cpu_settings": to_cpu_settings(correlation, validation)}
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    return {"min_search_size": min_search_size, "piv_settings": piv_settings}


def _build_engine(backend, frame_shape, correlation, validation):
    factory = get_engine_factory(backend)
    settings = _engine_settings(backend, correlation, validation)
    return factory(frame_shape, settings)


def _make_tiled_init_fn(backend, correlation, validation):
    if backend != "gpu":
        raise ValueError("tiling is only supported on the gpu backend")
    from ..engines.gpu_engine import _init_gpu_processor_raw
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    return lambda shape: _init_gpu_processor_raw(shape, min_search_size, piv_settings)


def _tile_margin(backend, correlation, validation):
    from ..engines.gpu_engine import default_tile_margin
    min_search_size, piv_settings = to_gpu_settings(correlation, validation)
    return correlation.tile_margin_px or default_tile_margin(min_search_size, piv_settings)


def _free_gpu(backend):
    if backend == "gpu":
        from ..engines.gpu_engine import free_gpu_pools
        free_gpu_pools()


def _gpu_report_fn(backend, verbose):
    if backend == "gpu" and verbose:
        from ..engines.gpu_engine import gpu_free_report
        return gpu_free_report
    return None


# ======================================================================
# Planar
# ======================================================================
def handle_pair_planar(pair_id, frame_a, frame_b, cfg, output_dir, engine, x, y):
    post = cfg.postprocess.for_pipeline()
    verbose = cfg.output.verbose
    if verbose:
        print(f"Processing {pair_id} ...", end=" ", flush=True)

    correlation, validation = cfg.correlation, cfg.validation
    backend = cfg.project.backend

    t_post0 = time.time()
    if correlation.use_tiling:
        init_fn = _make_tiled_init_fn(backend, correlation, validation)
        margin = _tile_margin(backend, correlation, validation)
        x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
            frame_a, frame_b, post, init_fn, correlation.n_tiles_y, correlation.n_tiles_x,
            margin, report_gpu_mem=True, free_pools_fn=lambda: _free_gpu(backend), verbose=verbose,
        )
    else:
        u, v, valid, elapsed, rejects = pipeline.process_frames(
            engine, frame_a, frame_b, post, report_gpu_mem=True,
            on_gpu_report=_gpu_report_fn(backend, verbose),
        )
    t_post = time.time() - t_post0 - elapsed

    u, v = apply_calibration(u, v, cfg.calibration.pixel_pitch_mm, cfg.calibration.frame_dt_s)
    n_valid, n_total = int(valid.sum()), int(valid.size)
    if verbose:
        print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors "
              f"(postprocess {t_post:.3f}s)")

    if cfg.output.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_velocity.npz"),
                 x=x, y=y, u=u, v=v, valid=valid)
    if cfg.output.save_plot:
        plot_and_save_planar(x, y, u, v, valid,
                              os.path.join(output_dir, f"{pair_id}_quiver.png"),
                              title=f"PIV velocity field -- {pair_id}",
                              quiver_scale=cfg.output.quiver_scale,
                              plot_dpi=cfg.output.plot_dpi, show_plots=cfg.output.show_plots)

    row = (pair_id, elapsed, n_valid, n_total, rejects["range_residual"], rejects["std_dev"])
    return row, u, v, valid, x, y


def process_pairs_planar(pair_source, cfg, output_dir, interactive_preview):
    backend = cfg.project.backend
    engine = x = y = None
    summary_rows = []

    for idx, (pair_id, frame_a, frame_b) in enumerate(pair_source):
        t_pre0 = time.time()
        frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, cfg.preprocess)
        if cfg.output.verbose and cfg.preprocess.min_max_filter_enabled:
            print(f"[timing] {pair_id}: preprocess {time.time() - t_pre0:.3f}s")
        if not cfg.correlation.use_tiling and engine is None:
            engine, x, y = _build_engine(backend, frame_a.shape, cfg.correlation, cfg.validation)

        row, u, v, valid, x, y = handle_pair_planar(pair_id, frame_a, frame_b, cfg, output_dir, engine, x, y)
        summary_rows.append(row)

        if idx == 0 and interactive_preview:
            preview_path = os.path.join(output_dir, f"{pair_id}_first_snapshot_preview.png")
            plot_and_save_planar(x, y, u, v, valid, preview_path,
                                  title=f"First snapshot preview -- {pair_id}",
                                  quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi)
            preview_first_snapshot_cli(preview_path)

    if backend == "gpu" and engine is not None:
        del engine
        _free_gpu(backend)
    return summary_rows


# ======================================================================
# Stereo
# ======================================================================
def _run_camera(frame_a, frame_b, cfg):
    backend = cfg.project.backend
    correlation, validation = cfg.correlation, cfg.validation
    post = cfg.postprocess.for_pipeline()
    verbose = cfg.output.verbose

    t_post0 = time.time()
    if correlation.use_tiling:
        init_fn = _make_tiled_init_fn(backend, correlation, validation)
        margin = _tile_margin(backend, correlation, validation)
        x, y, u, v, valid, elapsed, rejects = pipeline.process_frames_tiled(
            frame_a, frame_b, post, init_fn, correlation.n_tiles_y, correlation.n_tiles_x,
            margin, report_gpu_mem=True, free_pools_fn=lambda: _free_gpu(backend), verbose=verbose,
        )
        return u, v, valid, elapsed, x, y, rejects, time.time() - t_post0 - elapsed

    engine, x, y = _build_engine(backend, frame_a.shape, correlation, validation)
    u, v, valid, elapsed, rejects = pipeline.process_frames(
        engine, frame_a, frame_b, post, report_gpu_mem=True,
        on_gpu_report=_gpu_report_fn(backend, verbose),
    )
    t_post = time.time() - t_post0 - elapsed
    del engine
    if backend == "gpu":
        _free_gpu(backend)
    return u, v, valid, elapsed, x, y, rejects


def handle_pair_stereo(pair_id, dw_a0, dw_b0, dw_a1, dw_b1, cfg, angles, output_dir):
    verbose = cfg.output.verbose
    if verbose:
        print(f"Processing {pair_id} ...", end=" ", flush=True)

    u1, v1, valid1, elapsed1, x, y, r1, t_post1 = _run_camera(dw_a0, dw_b0, cfg)
    u2, v2, valid2, elapsed2, _, _, r2, t_post2 = _run_camera(dw_a1, dw_b1, cfg)
    valid = valid1 & valid2
    elapsed = elapsed1 + elapsed2

    U, V, W = pipeline.combine_stereo_pair(
        u1, v1, u2, v2, angles, cfg.stereo.world_scale_px_per_mm, cfg.calibration.frame_dt_s)

    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    W = np.where(valid, W, np.nan)

    n_valid, n_total = int(valid.sum()), int(valid.size)
    if verbose:
        print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors "
              f"(postprocess {t_post1 + t_post2:.3f}s)")

    if cfg.output.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_stereo_velocity.npz"),
                 x=x, y=y, U=U, V=V, W=W, valid=valid)
    if cfg.output.save_plot:
        plot_and_save_stereo(x, y, U, V, W, valid,
                              os.path.join(output_dir, f"{pair_id}_stereo_quiver.png"),
                              title=f"Stereo PIV -- {pair_id}",
                              quiver_scale=cfg.output.quiver_scale,
                              plot_dpi=cfg.output.plot_dpi, show_plots=cfg.output.show_plots)

    n_range = r1["range_residual"] + r2["range_residual"]
    n_std = r1["std_dev"] + r2["std_dev"]
    row = (pair_id, elapsed, n_valid, n_total, n_range, n_std)
    return row, x, y, U, V, W, valid


def process_pairs_stereo(pair_source, cfg, angles, output_dir, interactive_preview):
    summary_rows = []
    for idx, (pair_id, fa0, fb0, fa1, fb1) in enumerate(pair_source):
        fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
        fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)
        dw_a0 = cfg.stereo._cam0.dewarp_image(fa0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
        dw_b0 = cfg.stereo._cam0.dewarp_image(fb0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
        dw_a1 = cfg.stereo._cam1.dewarp_image(fa1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
        dw_b1 = cfg.stereo._cam1.dewarp_image(fb1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)

        row, x, y, U, V, W, valid = handle_pair_stereo(pair_id, dw_a0, dw_b0, dw_a1, dw_b1, cfg, angles, output_dir)
        summary_rows.append(row)

        if idx == 0 and interactive_preview:
            preview_path = os.path.join(output_dir, f"{pair_id}_first_snapshot_preview.png")
            plot_and_save_stereo(x, y, U, V, W, valid, preview_path,
                                  title=f"First snapshot preview -- {pair_id}",
                                  quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi)
            preview_first_snapshot_cli(preview_path)

    return summary_rows


# ======================================================================
# Shared summary writer
# ======================================================================
def write_summary(summary_rows, output_dir, cfg, stereo=False):
    if not summary_rows:
        return
    if cfg.output.save_summary_csv:
        name = "stereo_processing_summary.csv" if stereo else "processing_summary.csv"
        csv_path = os.path.join(output_dir, name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total",
                              "n_rejected_range_residual", "n_rejected_std_dev"])
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    total_time = sum(row[1] for row in summary_rows)
    print(f"Done: {len(summary_rows)} pair(s) in {total_time:.3f} s "
          f"({total_time / len(summary_rows):.3f} s/pair average)")


# ======================================================================
# Main
# ======================================================================
def _apply_cli_overrides(cfg: ProjectConfig, args):
    if args.backend is not None:
        cfg.project.backend = args.backend
    if args.mode is not None:
        cfg.project.mode = args.mode
    if args.input_path is not None:
        cfg.project.input_path = args.input_path
    if args.output_dir is not None:
        cfg.project.output_dir = args.output_dir


def build_arg_parser():
    p = argparse.ArgumentParser(prog="piv-suite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="piv_config.pivproj",
                   help="Path to a .pivproj JSON config file (created with defaults if missing).")
    p.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    p.add_argument("--mode", choices=["planar", "stereo"], default=None)
    p.add_argument("--input-path", default=None)
    p.add_argument("--output-dir", default=None)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = load_project(args.config)
    _apply_cli_overrides(cfg, args)
    save_project(args.config, cfg)  # persist CLI overrides back, matching original "edit the JSON" UX

    os.makedirs(cfg.project.output_dir, exist_ok=True)

    if cfg.project.input_mode == "set":
        set_paths, is_batch = resolve_set_paths(cfg.project.input_path)
    elif cfg.project.input_mode == "loose":
        set_paths, is_batch = [cfg.project.input_path], False
    else:
        sys.exit(f"Unknown input_mode: {cfg.project.input_mode!r} (use 'set' or 'loose')")

    if is_batch:
        print(f"[info] '{cfg.project.input_path}' contains {len(set_paths)} set(s) -- "
              "batch-processing each (no first-snapshot preview in this mode)")

    stereo = cfg.project.mode == "stereo"
    if stereo:
        cfg.stereo._cam0 = CameraMapping(
            cfg.stereo.cam0_mapping.x0, cfg.stereo.cam0_mapping.x_span,
            cfg.stereo.cam0_mapping.y0, cfg.stereo.cam0_mapping.y_span,
            cfg.stereo.cam0_mapping.dx_coefs, cfg.stereo.cam0_mapping.dy_coefs,
            cfg.stereo.cam0_mapping.name,
        )
        cfg.stereo._cam1 = CameraMapping(
            cfg.stereo.cam1_mapping.x0, cfg.stereo.cam1_mapping.x_span,
            cfg.stereo.cam1_mapping.y0, cfg.stereo.cam1_mapping.y_span,
            cfg.stereo.cam1_mapping.dx_coefs, cfg.stereo.cam1_mapping.dy_coefs,
            cfg.stereo.cam1_mapping.name,
        )
        angles = (np.deg2rad(cfg.stereo.alpha1_deg), np.deg2rad(cfg.stereo.alpha2_deg),
                  np.deg2rad(cfg.stereo.beta1_deg), np.deg2rad(cfg.stereo.beta2_deg))

    grand_summary = []
    for set_path in set_paths:
        output_dir = (os.path.join(cfg.project.output_dir, set_label(set_path))
                       if is_batch else cfg.project.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if stereo:
            if cfg.project.input_mode == "set":
                print(f"[info] processing set '{set_path}'")
                pair_source = iter_stereo_from_set(set_path, cfg.project.multiset_index,
                                                    cfg.project.stereo_frame_order)
            else:
                pair_source = iter_stereo_from_loose_files(
                    cfg.project.input_path, cfg.project.loose_glob,
                    cfg.project.suffix_cam0, cfg.project.suffix_cam1, cfg.project.stereo_frame_order)
            summary_rows = process_pairs_stereo(pair_source, cfg, angles, output_dir,
                                                 interactive_preview=not is_batch)
        else:
            if cfg.project.input_mode == "set":
                print(f"[info] processing set '{set_path}'")
                pair_source = iter_pairs_from_set(set_path, cfg.project.multiset_index)
            else:
                pair_source = iter_pairs_from_loose_files(
                    cfg.project.input_path, cfg.project.loose_glob,
                    cfg.project.suffix_a, cfg.project.suffix_b)
            summary_rows = process_pairs_planar(pair_source, cfg, output_dir,
                                                 interactive_preview=not is_batch)

        if not summary_rows:
            print(f"[warn] no pairs were processed for '{set_path}'")
            continue

        write_summary(summary_rows, output_dir, cfg, stereo=stereo)
        grand_summary.extend(summary_rows)

    if not grand_summary:
        sys.exit("No pairs were processed -- check input_mode/input_path")


if __name__ == "__main__":
    main()
