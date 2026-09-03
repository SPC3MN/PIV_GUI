"""Local-outlier statistics for each side's own field, no cross-resampling.

The "is our field as internally consistent as DaVis's own" question --
distinct from agreement WITH DaVis (scripts/compare_dataset.py), this asks
whether each field agrees with ITSELF: does a vector look like its own local
neighbourhood, judged independently on each side. scripts/compare_dataset.py
computes this too, but only alongside a griddata-based cross-comparison whose
Delaunay triangulation of a large point cloud per pair makes a 100-pair run
take about an hour; these statistics need no resampling at all, so a few
dozen pairs run in well under a minute.

    python scripts/piv_field_quality.py --mode planar \\
        --npz-dir piv_output --vc7-dir "C:\\data\\MyRecording\\PIV_MPd(...)" \\
        --px-per-mm 19.42 --max-pairs 30
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_davis_lavision import load_vc7_field
from compare_stereo_preview import load_vc7_stereo_field
from compare_velocity_fields import field_stats


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("planar", "stereo"), required=True)
    p.add_argument("--npz-dir", required=True, help="This app's completed batch output folder.")
    p.add_argument("--vc7-dir", required=True, help="DaVis's own B00001.vc7, B00002.vc7, ... (1-based).")
    p.add_argument("--px-per-mm", type=float, required=True)
    p.add_argument("--frame-dt", type=float, default=700e-6, help="Inter-frame time in seconds.")
    p.add_argument("--max-pairs", type=int, default=25)
    return p


def main():
    args = build_arg_parser().parse_args()

    # field_stats scores residuals in NATIVE displacement units (px/frame), so
    # the m/s fields have to be divided back by mm-per-px over dt.
    native = 1.0 / (args.px_per_mm * args.frame_dt)
    suffix = "_stereo_velocity.npz" if args.mode == "stereo" else "_velocity.npz"
    ids = sorted(int(f[:4]) for f in os.listdir(args.npz_dir) if f.endswith(suffix))[:args.max_pairs]
    if not ids:
        sys.exit(f"no {suffix} files found in {args.npz_dir}")

    acc = {"app": [], "dv": []}
    for i in ids:
        d = np.load(os.path.join(args.npz_dir, "%04d%s" % (i, suffix)))
        vc7_path = os.path.join(args.vc7_dir, "B%05d.vc7" % (i + 1))
        if not os.path.exists(vc7_path):
            continue
        if args.mode == "stereo":
            u, v, valid = d["U"] * 1000, d["V"] * 1000, d["valid"]
            _, _, ud, vd, _ = load_vc7_stereo_field(vc7_path)
        else:
            u, v, valid = d["u"] * 1000, d["v"] * 1000, d["valid"]
            _, _, ud, vd = load_vc7_field(vc7_path)
        u = np.where(valid, u, np.nan)
        v = np.where(valid, v, np.nan)
        acc["app"].append(field_stats("app", u, v, native_scale=native)[0])
        acc["dv"].append(field_stats("dv", ud, vd, native_scale=native)[0])

    if not acc["app"]:
        sys.exit("no pair had both an npz file and a matching vc7 file")

    print("%s -- %d pairs" % (args.mode, len(acc["app"])))
    print("%-8s %10s %10s %12s %12s %12s" % ("", "valid %", "resid p50", "resid p99", ">2 MAD %", ">3 MAD %"))
    for who, label in (("app", "this app"), ("dv", "DaVis")):
        rows = acc[who]

        def m(k):
            return float(np.mean([r[k] for r in rows]))

        print("%-8s %10.2f %10.3f %12.3f %12.3f %12.3f"
              % (label, 100.0 * m("n_valid") / m("n_total"), m("resid_median"), m("resid_p99"),
                 m("outliers_gt_2.0_pct"), m("outliers_gt_3.0_pct")))


if __name__ == "__main__":
    main()
