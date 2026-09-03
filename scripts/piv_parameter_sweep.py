"""A/B one or more configuration variants against DaVis's own vectors, on a
handful of pairs, in a single command.

Exists so trying a parameter change is (edit VARIANTS below or add a CLI
override -> read one printed table) instead of (run a full batch -> run
scripts/compare_dataset.py -> open the CSV). Comparison math mirrors
scripts/compare_davis_lavision.py's compare(), except the resample uses
RegularGridInterpolator -- DaVis's field IS a regular grid, so this is the
same answer far faster, which is what makes a multi-variant sweep practical
even on frame sizes DaVis takes tens of seconds to process per pair.

    python scripts/piv_parameter_sweep.py --mode planar \\
        --set-file "C:\\data\\MyRecording.set" \\
        --vc7-dir "C:\\data\\MyRecording\\PIV_MPd(...)" \\
        --variants base,smoothn_off,interp_5 --n 3

Each named variant in VARIANTS is a small function applied to an otherwise
DaVis-typical ProjectConfig (see piv_batch_sample.py's module docstring for
what "DaVis-typical" means here) -- add your own to try something new.
"""
import argparse
import itertools
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_davis_lavision import load_vc7_field
from compare_stereo_preview import load_vc7_stereo_field
from piv_suite.calibration.camera_mapping import build_stereo_cameras
from piv_suite.cli.main import process_pairs_planar, process_pairs_stereo
from piv_suite.config.schema import PassSettings, ProjectConfig
from piv_suite.io.davis_set import (iter_pairs_from_set, iter_stereo_from_set,
                                    read_calibration_from_set, read_stereo_calibration_from_set)


def _set_pass_schedule(cfg, schedule):
    cfg.correlation.passes = [PassSettings(w, o) for w, o in schedule]


# Each variant is a callable applied to an otherwise-unmodified config.
# Add entries here to try something new -- nothing else needs to change.
VARIANTS = {
    "base": lambda c: None,
    "no_minmax": lambda c: setattr(c.preprocess, "min_max_filter_enabled", False),
    "minmax_len5": lambda c: setattr(c.preprocess, "min_max_filter_length", 5),
    "smoothn_2": lambda c: setattr(c.validation, "smoothn_p", 2.0),
    "smoothn_5": lambda c: setattr(c.validation, "smoothn_p", 5.0),
    "smoothn_10": lambda c: setattr(c.validation, "smoothn_p", 10.0),
    "smoothn_off": lambda c: setattr(c.validation, "smoothn", False),
    "pp_median_1.5": lambda c: setattr(c.validation, "per_pass_median_threshold", 1.5),
    "pp_median_3": lambda c: setattr(c.validation, "per_pass_median_threshold", 3.0),
    "pp_size_2": lambda c: setattr(c.validation, "per_pass_median_size", 2),
    "no_per_pass": lambda c: setattr(c.validation, "per_pass_validation", False),
    "interp_5": lambda c: setattr(c.correlation, "interpolation_order", 5),
    "rf_2.5": lambda c: setattr(c.postprocess.range_filter, "residual_max", 2.5),
    "rf_1.5": lambda c: setattr(c.postprocess.range_filter, "residual_max", 1.5),
    "no_global_std": lambda c: setattr(c.postprocess, "global_outlier_std", None),
    "extra_pass": lambda c: _set_pass_schedule(c, [(64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75), (32, 0.75)]),
    "two_coarse": lambda c: _set_pass_schedule(c, [(64, 0.5), (64, 0.5), (32, 0.75), (32, 0.75), (32, 0.75)]),
}


def build_config(mode, set_path, out_dir, workers, sheet_z):
    cfg = ProjectConfig()
    cfg.project.input_mode = "set"
    cfg.project.input_path = set_path
    cfg.project.output_dir = out_dir
    cfg.project.backend = "cpu"
    cfg.project.mode = mode
    cfg.preprocess.min_max_filter_enabled = True
    cfg.preprocess.min_max_filter_length = 4
    cal = read_calibration_from_set(set_path)
    cfg.calibration.pixel_pitch_mm = cal.pixel_pitch_mm
    cfg.calibration.frame_dt_s = cal.frame_dt_s
    cfg.postprocess.range_filter.enabled = True
    cfg.postprocess.range_filter.residual_max = 2.0
    cfg.postprocess.range_filter.window_size = 3
    cfg.postprocess.range_filter.insertion_max = 3.0
    cfg.postprocess.range_filter.min_neighbours = 3
    cfg.postprocess.remove_small_groups_threshold = None if mode == "stereo" else 5
    cfg.output.save_npz = True
    cfg.output.save_plot = False
    cfg.output.verbose = False
    cfg.performance.n_workers = workers
    if mode == "stereo":
        cfg.stereo = read_stereo_calibration_from_set(set_path)
        cfg.stereo.sheet_z_mm = sheet_z
    return cfg


def compare_one(npz_path, vc7_path, mode, px_per_mm):
    """-> dict of density/magnitude/correlation numbers for one pair."""
    d = np.load(npz_path)
    if mode == "stereo":
        x, y = d["x"] / px_per_mm, d["y"] / px_per_mm
        u, v, w, valid = d["U"] * 1000, d["V"] * 1000, d["W"] * 1000, d["valid"]
        xd, yd, ud, vd, wd = load_vc7_stereo_field(vc7_path)
    else:
        x, y = d["x"] / px_per_mm, d["y"] / px_per_mm
        u, v, w, valid = d["u"] * 1000, d["v"] * 1000, None, d["valid"]
        xd, yd, ud, vd = load_vc7_field(vc7_path)
        wd = None
    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)
    if w is not None:
        w = np.where(valid, w, np.nan)

    out = {"app_density": 100.0 * valid.sum() / valid.size,
           "dv_density": 100.0 * np.isfinite(ud).sum() / ud.size,
           "app_mag_mean": float(np.nanmean(np.hypot(u, v))),
           "dv_mag_mean": float(np.nanmean(np.hypot(ud, vd))),
           "app_mag_p99": float(np.nanpercentile(np.hypot(u, v), 99)),
           "dv_mag_p99": float(np.nanpercentile(np.hypot(ud, vd), 99))}

    xc = x - (x.min() + x.max()) / 2
    yc = y - (y.min() + y.max()) / 2
    xdc = xd - (xd.min() + xd.max()) / 2
    ydc = yd - (yd.min() + yd.max()) / 2
    if np.sign(yc[1, 0] - yc[0, 0]) != np.sign(ydc[1, 0] - ydc[0, 0]):
        ydc = -ydc
    xd1, yd1 = xdc[0, :], ydc[:, 0]
    flip = yd1[0] > yd1[-1]
    if flip:
        yd1 = yd1[::-1]

    def interp(field):
        f = field[::-1] if flip else field
        g = RegularGridInterpolator((yd1, xd1), f, bounds_error=False, fill_value=np.nan)
        return g(np.column_stack([yc.ravel(), xc.ravel()])).reshape(xc.shape)

    ur, vr = interp(ud), interp(vd)
    wr = interp(wd) if wd is not None else None

    ok = ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(ur) & ~np.isnan(vr)
    if wr is not None:
        ok = ok & ~np.isnan(w) & ~np.isnan(wr)
    if np.corrcoef(v[ok], vr[ok])[0, 1] < 0:
        vr = -vr
    out["n"] = int(ok.sum())
    out["corr_u"] = float(np.corrcoef(u[ok], ur[ok])[0, 1])
    out["corr_v"] = float(np.corrcoef(v[ok], vr[ok])[0, 1])
    out["mean_abs_diff"] = float(np.hypot(u[ok] - ur[ok], v[ok] - vr[ok]).mean())
    if wr is not None:
        out["corr_w"] = float(np.corrcoef(w[ok], wr[ok])[0, 1])
        out["mean_abs_diff_w"] = float(np.abs(w[ok] - wr[ok]).mean())
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("planar", "stereo"), required=True)
    p.add_argument("--set-file", required=True, help="DaVis .set project (recording, not a job).")
    p.add_argument("--vc7-dir", required=True, help="DaVis's own B00001.vc7, B00002.vc7, ... (1-based).")
    p.add_argument("--variants", default="base",
                   help="Comma-separated names from VARIANTS, e.g. base,no_minmax,interp_5.")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--sheet-z", type=float, default=None,
                   help="Stereo only: real Z (mm) of the laser sheet -- required for stereo mode.")
    p.add_argument("--keep", default=None, help="Keep npz output under this dir instead of a temp dir.")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "stereo" and args.sheet_z is None:
        sys.exit("--sheet-z is required for --mode stereo (no calibration file records it)")

    results = {}
    for name in [n.strip() for n in args.variants.split(",")]:
        if name not in VARIANTS:
            sys.exit(f"unknown variant {name!r} -- known: {', '.join(sorted(VARIANTS))}")
        out_dir = os.path.join(args.keep, name) if args.keep else tempfile.mkdtemp(prefix="piv_sweep_")
        os.makedirs(out_dir, exist_ok=True)
        cfg = build_config(args.mode, args.set_file, out_dir, args.workers, args.sheet_z)
        VARIANTS[name](cfg)
        if args.mode == "stereo":
            cfg.stereo._cam0, cfg.stereo._cam1 = build_stereo_cameras(cfg.stereo)
        px_per_mm = 1.0 / cfg.calibration.pixel_pitch_mm

        src = (iter_stereo_from_set(args.set_file, 0, cfg.project.stereo_frame_order)
               if args.mode == "stereo" else iter_pairs_from_set(args.set_file, 0))
        src = itertools.islice(src, args.start, args.start + args.n)
        t0 = time.time()
        runner = process_pairs_stereo if args.mode == "stereo" else process_pairs_planar
        runner(src, cfg, out_dir, interactive_preview=False)
        elapsed = time.time() - t0

        suffix = "_stereo_velocity.npz" if args.mode == "stereo" else "_velocity.npz"
        per_pair = []
        for i in range(args.start, args.start + args.n):
            npz = os.path.join(out_dir, "%04d%s" % (i, suffix))
            vc7 = os.path.join(args.vc7_dir, "B%05d.vc7" % (i + 1))
            if os.path.exists(npz) and os.path.exists(vc7):
                per_pair.append(compare_one(npz, vc7, args.mode, px_per_mm))
        if not per_pair:
            print(f"[{name}] no comparable pairs found -- skipping")
            continue
        agg = {k: float(np.mean([r[k] for r in per_pair])) for k in per_pair[0]}
        agg["seconds"] = elapsed
        results[name] = agg
        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)
        print("[%s] done in %.0fs" % (name, elapsed))

    if not results:
        sys.exit("no variant produced a comparable result")

    keys = (["app_density", "dv_density", "app_mag_mean", "app_mag_p99", "dv_mag_p99", "corr_u", "corr_v"]
            + (["corr_w"] if args.mode == "stereo" else [])
            + ["mean_abs_diff"]
            + (["mean_abs_diff_w"] if args.mode == "stereo" else [])
            + ["seconds"])
    print("\n%-14s " % "variant" + " ".join("%12s" % k for k in keys))
    for name, r in results.items():
        print("%-14s " % name + " ".join("%12.3f" % r[k] for k in keys))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
