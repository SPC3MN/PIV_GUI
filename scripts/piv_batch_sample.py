r"""Drive this app's own CLI pipeline over a bounded slice of a DaVis `.set`
project, writing the same per-pair `.npz` files a normal batch run produces
-- so scripts/compare_dataset.py, scripts/make_comparison_plots.py, or
scripts/piv_field_quality.py can measure them against DaVis's own `.vc7`
output for a handful of pairs, without committing to a full multi-hour run.

Exists because cli.main has no "process only pairs N..M" switch -- a real
DaVis recording can be 1000+ pairs of several-megapixel frames, and a full
run is a multi-hour proposition per experiment. Everything else (config
construction, preprocessing, dewarp, correlation, validation) goes through
the exact same functions the CLI uses; only the pair source is bounded.

    python scripts/piv_batch_sample.py --mode planar \
        --set "C:\data\MyRecording.set" --out piv_sample_output --n 25

The default preprocessing/postprocessing values below (min/max filter
length 4, universal-outlier removal factor 2 / insertion factor 3 / 3x3
neighbourhood / minimum 3 neighbours) match a typical DaVis "Standard PIV"
job as reported in a real project's JobHistory.xml -- override them via the
flags below for a project whose own JobHistory reports different settings.
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piv_suite.calibration.camera_mapping import build_stereo_cameras
from piv_suite.cli.main import process_pairs_planar, process_pairs_stereo, write_summary
from piv_suite.config.schema import ProjectConfig
from piv_suite.io.davis_set import (iter_pairs_from_set, iter_stereo_from_set,
                                    read_calibration_from_set, read_stereo_calibration_from_set)


def build_config(args):
    cfg = ProjectConfig()
    cfg.project.input_mode = "set"
    cfg.project.input_path = args.set_path
    cfg.project.output_dir = args.out_dir
    cfg.project.backend = "cpu"
    cfg.project.mode = args.mode

    cfg.preprocess.min_max_filter_enabled = not args.no_minmax
    cfg.preprocess.min_max_filter_length = args.minmax_length

    cal = read_calibration_from_set(args.set_path)
    cfg.calibration.pixel_pitch_mm = cal.pixel_pitch_mm
    cfg.calibration.frame_dt_s = cal.frame_dt_s
    if cal.frame_dt_s is None:
        sys.exit("refusing to run: no frame_dt_s found -- results would be px/frame, not m/s")

    cfg.postprocess.range_filter.enabled = not args.no_range_filter
    cfg.postprocess.range_filter.residual_max = args.range_residual_max
    cfg.postprocess.range_filter.window_size = args.range_window
    cfg.postprocess.range_filter.insertion_max = args.range_insertion_max
    cfg.postprocess.range_filter.min_neighbours = args.range_min_neighbours
    cfg.postprocess.remove_small_groups_threshold = args.remove_small_groups

    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.verbose = True
    if args.workers is not None:
        cfg.performance.n_workers = args.workers

    if args.mode == "stereo":
        cfg.stereo = read_stereo_calibration_from_set(args.set_path)
        # Real Z (mm) of the laser sheet for THIS recording -- an
        # acquisition-time quantity with no calibration-file source (see
        # StereoSettings.sheet_z_mm's own docstring). Pick a physically
        # sensible value for your own recording -- the midpoint of the two
        # calibrated Z-planes is a reasonable default absent better
        # information, but is not guaranteed to be the true sheet position.
        cfg.stereo.sheet_z_mm = args.sheet_z
        cfg.stereo._cam0, cfg.stereo._cam1 = build_stereo_cameras(cfg.stereo)
        print(f"[info] stereo world_shape={cfg.stereo.world_shape} "
              f"scale={cfg.stereo.world_scale_px_per_mm:.6f} px/mm")
    print(f"[info] pixel_pitch={cfg.calibration.pixel_pitch_mm:.8f} mm/px "
          f"frame_dt={cfg.calibration.frame_dt_s * 1e6:.1f} us")
    return cfg


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("planar", "stereo"), required=True)
    p.add_argument("--set", dest="set_path", required=True, help="DaVis .set project (recording, not a job).")
    p.add_argument("--out", dest="out_dir", required=True, help="Output folder for the .npz files.")
    p.add_argument("--n", type=int, default=2, help="Number of pairs to process.")
    p.add_argument("--start", type=int, default=0, help="First pair index (0-based).")
    p.add_argument("--workers", type=int, default=None, help="CPU worker processes; default auto-detects.")
    p.add_argument("--sheet-z", type=float, default=None,
                   help="Stereo only: real Z (mm) of the laser sheet -- required for stereo mode.")
    p.add_argument("--no-minmax", action="store_true", help="Disable the min/max preprocessing filter.")
    p.add_argument("--minmax-length", type=int, default=4, help="Min/max filter window size L, in pixels.")
    p.add_argument("--no-range-filter", action="store_true",
                   help="Disable the local-median universal-outlier filter.")
    p.add_argument("--range-residual-max", type=float, default=2.0)
    p.add_argument("--range-window", type=int, default=3)
    p.add_argument("--range-insertion-max", type=float, default=3.0)
    p.add_argument("--range-min-neighbours", type=int, default=3)
    p.add_argument("--remove-small-groups", type=int, default=None,
                   help="Drop connected valid-vector groups smaller than this. Default: off.")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "stereo" and args.sheet_z is None:
        sys.exit("--sheet-z is required for --mode stereo (no calibration file records it)")

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_config(args)

    src = (iter_stereo_from_set(args.set_path, 0, cfg.project.stereo_frame_order)
           if args.mode == "stereo" else iter_pairs_from_set(args.set_path, 0))
    src = itertools.islice(src, args.start, args.start + args.n)

    t0 = time.time()
    runner = process_pairs_stereo if args.mode == "stereo" else process_pairs_planar
    rows = runner(src, cfg, args.out_dir, interactive_preview=False)
    dt = time.time() - t0

    write_summary(rows, args.out_dir, cfg, stereo=(args.mode == "stereo"))
    n_valid = sum(r[2] for r in rows)
    n_total = sum(r[3] for r in rows)
    print(f"\n[done] {len(rows)} pair(s) in {dt:.1f}s ({dt / max(len(rows), 1):.1f}s/pair) -- "
          f"density {100.0 * n_valid / max(n_total, 1):.2f}%")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
