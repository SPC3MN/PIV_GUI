"""Compare this app's ALREADY-COMPLETED batch PIV output against ALREADY-
COMPLETED real LaVision DaVis reference output, across an entire dataset
(potentially hundreds of pairs), producing time-series trend plots + a
CSV -- one row per pair.

Deliberately decoupled from the PIV software itself: this script never
re-runs any correlation/triangulation, so it needs no Qt/PySide6 and no
`piv_suite_gui` import at all (contrast `compare_stereo_preview.py`, which
constructs a real `PreviewPanel` to re-run the pipeline). It only reads
this app's own finished `{pair_id}_stereo_velocity.npz` /
`{pair_id}_velocity.npz` files (written by the existing CLI/GUI batch
processors -- see `cli/main.py:handle_pair_stereo`/`handle_pair`/
`handle_pair_dual_planar`) and DaVis's own finished `B00001.vc7,
B00002.vc7, ...` vector files, plus the original `.set` project purely for
calibration constants (pixel scale, frame timing) -- none of which are
recorded in the npz files themselves.

Workflow:
    1. Process a full recording through this app as normal (GUI Run, or
       the `piv-suite` CLI) with `output.save_npz` enabled (the default).
    2. Point this script at that npz output folder + the original .set +
       DaVis's own vc7 output folder:

    python scripts/compare_dataset.py \\
        --set-file "D:\\Final_Stereo\\Swirl\\On Time=6.0_...set" \\
        --npz-dir <this app's completed batch output folder> \\
        --vc7-dir "D:\\...\\StereoPIV_MPd(3x32x32_75%ov)" \\
        --mode stereo \\
        --out-dir piv_comparison_output

Resumable: writes one checkpoint JSON per pair as it's computed
(--out-dir/checkpoint/{pair_id}.json), and skips any pair whose
checkpoint already exists on a rerun -- this real dataset's drive has a
documented history of stalling mid-script, so an interrupted run must
not lose completed work. `--force` recomputes everyone anyway.
`--summarize-only` rebuilds summary.csv + the plots from whatever
checkpoints already exist, touching no npz/vc7/calibration code at all
(useful after changing only the plotting logic).

UNIT CAVEAT (real, cannot be detected after the fact): this script always
assumes the npz velocities are m/s. It has no way to tell, from the npz
file alone, whether the ORIGINAL batch run that produced them actually
had a real `frame_dt_s` available -- if it didn't,
`apply_calibration`/`combine_stereo_pair`/`combine_dual_planar_pair` all
silently stayed in native px/frame or mm/frame units instead (see each
function's own "no dt = stay in native units" docstring), and every
diff/corr number this script reports would be wrong by that missing
conversion factor, with no error raised. Confirm the original run's
calibration actually had frame timing before trusting results.

PAIR ID PRECONDITION: `pair_id` is assumed to be the zero-padded 4-digit
string (`f"{i:04d}"`) that `iter_stereo_from_set`/`iter_pairs_from_set`/
`iter_dual_planar_from_set` produce for a `.set`-mode batch run (see
`io/davis_set.py`). A batch run against loose files instead uses an
arbitrary filename stem as `pair_id` (see `io/loose_files.py`), which
this script's `int(pair_id)` DaVis-vc7-index mapping cannot handle.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_davis_lavision import compare, load_vc7_field
from compare_stereo_preview import load_vc7_stereo_field
from compare_velocity_fields import field_stats

from piv_suite.io.davis_set import (
    read_calibration_from_set, read_dual_planar_calibration_from_set,
    read_stereo_calibration_from_set,
)

MODES = ("stereo", "planar", "dual_planar")
OUTLIER_THRESHOLDS = (2.0, 3.0, 5.0, 10.0)

# Fixed column order -- every row has exactly these columns regardless of
# `status`, blank/NaN for whatever wasn't computed. See module docstring
# for why `app_*`/`dv_*` are populated independently of each other and of
# `cmp_*` (a `missing_vc7` row still carries this app's own density/
# residual trend at that index, which is useful on its own).
CSV_FIELDNAMES = ["pair_id", "pair_index", "status", "error_message"]
for _side in ("app", "dv"):
    CSV_FIELDNAMES += [
        f"{_side}_n_total", f"{_side}_n_valid", f"{_side}_pct_valid",
        f"{_side}_mag_mean", f"{_side}_mag_p99", f"{_side}_mag_max",
        f"{_side}_resid_median", f"{_side}_resid_p99",
    ]
    for _t in OUTLIER_THRESHOLDS:
        CSV_FIELDNAMES += [f"{_side}_outliers_gt_{int(_t)}", f"{_side}_outliers_gt_{int(_t)}_pct"]
CSV_FIELDNAMES += [
    "cmp_n_compared", "cmp_v_sign_flipped",
    "cmp_mean_abs_diff", "cmp_median_abs_diff", "cmp_p95_abs_diff", "cmp_corr_u", "cmp_corr_v",
    "cmp_mean_abs_diff_w", "cmp_median_abs_diff_w", "cmp_p95_abs_diff_w", "cmp_corr_w",
]

_NAN_FIELD_STATS = {
    "n_total": np.nan, "n_valid": np.nan, "pct_valid": np.nan,
    "mag_mean": np.nan, "mag_p99": np.nan, "mag_max": np.nan,
    "resid_median": np.nan, "resid_p99": np.nan,
}
for _t in OUTLIER_THRESHOLDS:
    _NAN_FIELD_STATS[f"outliers_gt_{_t}"] = np.nan
    _NAN_FIELD_STATS[f"outliers_gt_{_t}_pct"] = np.nan

_NAN_CMP = {
    "n_compared": np.nan, "v_sign_flipped": False,
    "mean_abs_diff": np.nan, "median_abs_diff": np.nan, "p95_abs_diff": np.nan,
    "corr_u": np.nan, "corr_v": np.nan,
    "mean_abs_diff_w": np.nan, "median_abs_diff_w": np.nan, "p95_abs_diff_w": np.nan, "corr_w": np.nan,
}


def _sanitize_field_stats(prefix, stats):
    """field_stats' own keys use a bare float threshold (`outliers_gt_2.0`)
    -- rename to an int-suffixed, CSV/JSON-header-friendly form
    (`outliers_gt_2`) and add the caller's app_/dv_ prefix."""
    out = {}
    for k, v in stats.items():
        if k == "label":
            continue
        for t in OUTLIER_THRESHOLDS:
            if k == f"outliers_gt_{t}":
                k = f"outliers_gt_{int(t)}"
            elif k == f"outliers_gt_{t}_pct":
                k = f"outliers_gt_{int(t)}_pct"
        out[f"{prefix}_{k}"] = v
    return out


def npz_suffix_for_mode(mode):
    return "_stereo_velocity.npz" if mode == "stereo" else "_velocity.npz"


def discover_npz_pair_ids(npz_dir, mode):
    suffix = npz_suffix_for_mode(mode)
    names = [f for f in os.listdir(npz_dir) if f.endswith(suffix)]
    if mode != "stereo":
        # planar and dual_planar share "_velocity.npz" -- exclude a
        # "_stereo_velocity.npz" false-positive if npz_dir was ever reused
        # across a stereo run too (both end in "_velocity.npz").
        names = [f for f in names if not f.endswith("_stereo_velocity.npz")]
    return sorted(f[: -len(suffix)] for f in names)


def discover_vc7_pair_ids(vc7_dir):
    ids = []
    for f in os.listdir(vc7_dir):
        m = re.match(r"^B(\d{5})\.vc7$", f)
        if m:
            ids.append(f"{int(m.group(1)) - 1:04d}")
    return sorted(ids)


def vc7_path_for_pair(vc7_dir, pair_id):
    return os.path.join(vc7_dir, f"B{int(pair_id) + 1:05d}.vc7")


def npz_path_for_pair(npz_dir, mode, pair_id):
    return os.path.join(npz_dir, f"{pair_id}{npz_suffix_for_mode(mode)}")


def _load_app_field(npz_path, mode, px_per_mm):
    """Masks u,v,(w) to NaN wherever `valid` is False -- this is this
    app's own RAW validity mask (pre-`replace_invalid_vectors`), which the
    npz's own `valid` array records regardless of whether that later fill
    step ran, since it's returned unchanged for exactly this reason (see
    pipeline.process_stereo_pair's own return contract). Deliberately does
    NOT trust the npz's `U`/`V`/`W` (or `u`/`v`) arrays as-is even though
    they may already be interpolated/filled at rejected cells -- comparing
    only genuinely-measured vectors against DaVis is a more honest
    accuracy signal, and matches the "density" convention (n_valid/n_total
    on the RAW mask) used throughout this app's own stereo-density
    investigation, not the 100%-covered-by-filling density a naive read
    of the npz's arrays alone would suggest."""
    d = np.load(npz_path)
    if mode == "stereo":
        x, y, u, v, w, valid = d["x"], d["y"], d["U"], d["V"], d["W"], d["valid"]
        x_mm, y_mm = x / px_per_mm, y / px_per_mm  # raw world-px -> mm
    elif mode == "dual_planar":
        # combine_dual_planar_pair already places these on a shared mm
        # canvas -- unlike stereo/planar, no px->mm conversion here.
        x_mm, y_mm, u, v, w, valid = d["x"], d["y"], d["u"], d["v"], None, d["valid"]
    else:  # planar
        x_mm, y_mm = d["x"] / px_per_mm, d["y"] / px_per_mm
        u, v, w, valid = d["u"], d["v"], None, d["valid"]
    # This app's own combine_stereo_pair/apply_calibration/
    # combine_dual_planar_pair all produce m/s (when frame_dt_s was known
    # -- see module docstring's UNIT CAVEAT), but load_vc7_field/
    # load_vc7_stereo_field always return mm/s -- a raw 1000x mismatch if
    # compared directly (same conversion compare_stereo_preview.py's own
    # vel_to_mm_s does before calling compare()).
    u = np.where(valid, u * 1000.0, np.nan)
    v = np.where(valid, v * 1000.0, np.nan)
    if w is not None:
        w = np.where(valid, w * 1000.0, np.nan)
    return x_mm, y_mm, u, v, w


def _load_reference_field(vc7_path, mode):
    if mode == "stereo":
        return load_vc7_stereo_field(vc7_path)  # x,y,u,v,w (w may be None)
    x, y, u, v = load_vc7_field(vc7_path)
    return x, y, u, v, None


def process_pair(pair_id, npz_dir, vc7_dir, mode, native_scale, px_per_mm):
    """Never raises. Returns a flat dict with every CSV_FIELDNAMES key
    present regardless of outcome (NaN/blank for whatever wasn't
    computed) -- see module docstring for the app_*/dv_*/cmp_* population
    rules."""
    result = {"pair_id": pair_id, "pair_index": int(pair_id), "status": "ok", "error_message": ""}
    errors = []

    npz_path = npz_path_for_pair(npz_dir, mode, pair_id)
    vc7_path = vc7_path_for_pair(vc7_dir, pair_id)

    app_field = None
    if not os.path.exists(npz_path):
        errors.append(("missing_npz", f"no npz at {npz_path}"))
    else:
        try:
            app_field = _load_app_field(npz_path, mode, px_per_mm)
        except Exception as e:
            errors.append(("npz_load_error", f"{type(e).__name__}: {e}"))

    dv_field = None
    if not os.path.exists(vc7_path):
        errors.append(("missing_vc7", f"no vc7 at {vc7_path}"))
    else:
        try:
            dv_field = _load_reference_field(vc7_path, mode)
        except Exception as e:
            errors.append(("vc7_load_error", f"{type(e).__name__}: {e}"))

    if app_field is not None:
        x, y, u, v, w = app_field
        stats, _ = field_stats("app", u, v, native_scale=native_scale, residual_thresholds=OUTLIER_THRESHOLDS)
        result.update(_sanitize_field_stats("app", stats))
    else:
        result.update(_sanitize_field_stats("app", {**_NAN_FIELD_STATS, "label": "app"}))

    if dv_field is not None:
        x_dv, y_dv, u_dv, v_dv, w_dv = dv_field
        stats, _ = field_stats("dv", u_dv, v_dv, native_scale=native_scale, residual_thresholds=OUTLIER_THRESHOLDS)
        result.update(_sanitize_field_stats("dv", stats))
    else:
        result.update(_sanitize_field_stats("dv", {**_NAN_FIELD_STATS, "label": "dv"}))

    if app_field is not None and dv_field is not None:
        cmp_ = compare(pair_id, x, y, u, v, x_dv, y_dv, u_dv, v_dv, w_app=w, w_dv=w_dv)
        result.update({f"cmp_{k}": v for k, v in cmp_.items()})
        result["status"] = "no_overlap" if cmp_["n_compared"] == 0 else "ok"
    else:
        result.update({f"cmp_{k}": v for k, v in _NAN_CMP.items()})
        status_priority = ["npz_load_error", "vc7_load_error", "missing_npz", "missing_vc7"]
        codes = [c for c, _ in errors]
        result["status"] = next(s for s in status_priority if s in codes)

    if errors:
        result["error_message"] = "; ".join(f"{c}: {m}" for c, m in errors)
    return result


def checkpoint_path(out_dir, pair_id):
    return os.path.join(out_dir, "checkpoint", f"{pair_id}.json")


def save_checkpoint(out_dir, pair_id, result, mode):
    path = checkpoint_path(out_dir, pair_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**result, "schema_version": 1, "mode": mode}, f)
    os.replace(tmp, path)  # atomic on Windows/POSIX -- never leaves a
                            # truncated checkpoint if the process is
                            # killed mid-write (this dataset's documented
                            # hang/stall risk).


def load_checkpoints(out_dir):
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    rows = []
    if not os.path.isdir(ckpt_dir):
        return rows
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ckpt_dir, fname)) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r.get("pair_index", 0))
    return rows


def assemble_summary_csv(out_dir):
    rows = load_checkpoints(out_dir)
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            row_out = {}
            for k in CSV_FIELDNAMES:
                v = r.get(k, "")
                if isinstance(v, float) and math.isnan(v):
                    v = ""
                row_out[k] = v
            w.writerow(row_out)
    return csv_path, rows


def _series(rows, key):
    idx = np.array([r["pair_index"] for r in rows], dtype=float)
    val = np.array([np.nan if r.get(key, "") == "" else float(r.get(key, np.nan)) for r in rows], dtype=float)
    order = np.argsort(idx)
    return idx[order], val[order]


def _status_rug(ax, rows):
    grey_idx = [r["pair_index"] for r in rows if r["status"] in ("missing_npz", "missing_vc7", "no_overlap")]
    red_idx = [r["pair_index"] for r in rows if r["status"] in ("npz_load_error", "vc7_load_error")]
    ylo, _ = ax.get_ylim()
    span = ax.get_ylim()[1] - ylo
    tick_y = ylo - 0.03 * span
    if grey_idx:
        ax.scatter(grey_idx, [tick_y] * len(grey_idx), marker="|", color=(0.62, 0.62, 0.62),
                   s=40, clip_on=False, label="missing/no-overlap")
    if red_idx:
        ax.scatter(red_idx, [tick_y] * len(red_idx), marker="|", color=(0.85, 0.05, 0.05),
                   s=40, clip_on=False, label="load error")


def plot_density_trends(rows, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    idx, app_pct = _series(rows, "app_pct_valid")
    _, dv_pct = _series(rows, "dv_pct_valid")
    ax1.plot(idx, app_pct, label="this app", color="tab:blue")
    ax1.plot(idx, dv_pct, label="LaVision DaVis", color="tab:orange")
    ax1.set_ylabel("valid vectors (%)")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.set_title("Density (valid vector %) per snapshot")

    idx2, n_cmp = _series(rows, "cmp_n_compared")
    _, n_app = _series(rows, "app_n_valid")
    _, n_dv = _series(rows, "dv_n_valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = 100.0 * n_cmp / np.fmax(1.0, np.fmin(n_app, n_dv))
    ax2.plot(idx2, coverage, color="tab:green")
    ax2.set_ylabel("overlap coverage (%)")
    ax2.set_xlabel("pair index")
    ax2.set_ylim(0, 105)
    ax2.set_title("Compared/overlap coverage per snapshot")
    _status_rug(ax2, rows)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_diff_magnitude_trends(rows, out_path, mode):
    has_w = mode == "stereo"
    fig, axes = plt.subplots(2 if has_w else 1, 1, figsize=(10, 7 if has_w else 4), sharex=True, squeeze=False)
    axes = axes[:, 0]

    idx, mean_d = _series(rows, "cmp_mean_abs_diff")
    _, med_d = _series(rows, "cmp_median_abs_diff")
    _, p95_d = _series(rows, "cmp_p95_abs_diff")
    ax = axes[0]
    ax.plot(idx, mean_d, label="mean", color="tab:blue")
    ax.plot(idx, med_d, label="median", color="tab:green")
    ax.plot(idx, p95_d, label="p95", color="tab:red")
    ax.set_ylabel("|diff U,V| (mm/s)")
    ax.set_title("This app vs DaVis: in-plane velocity difference per snapshot")
    ax.legend()
    _status_rug(ax, rows)

    if has_w:
        _, mean_w = _series(rows, "cmp_mean_abs_diff_w")
        _, med_w = _series(rows, "cmp_median_abs_diff_w")
        _, p95_w = _series(rows, "cmp_p95_abs_diff_w")
        ax = axes[1]
        ax.plot(idx, mean_w, label="mean", color="tab:blue")
        ax.plot(idx, med_w, label="median", color="tab:green")
        ax.plot(idx, p95_w, label="p95", color="tab:red")
        ax.set_ylabel("|diff W| (mm/s)")
        ax.set_title("This app vs DaVis: W (out-of-plane) difference per snapshot")
        ax.legend()
        _status_rug(ax, rows)

    axes[-1].set_xlabel("pair index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correlation_and_outlier_trends(rows, out_path, mode, outlier_threshold):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    idx, corr_u = _series(rows, "cmp_corr_u")
    _, corr_v = _series(rows, "cmp_corr_v")
    ax1.plot(idx, corr_u, label="corr(U)", color="tab:blue")
    ax1.plot(idx, corr_v, label="corr(V)", color="tab:orange")
    if mode == "stereo":
        _, corr_w = _series(rows, "cmp_corr_w")
        ax1.plot(idx, corr_w, label="corr(W)", color="tab:green")
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("correlation")
    ax1.set_title("This app vs DaVis: correlation per snapshot")
    ax1.legend()
    _status_rug(ax1, rows)

    t = int(outlier_threshold)
    idx2, app_out = _series(rows, f"app_outliers_gt_{t}_pct")
    _, dv_out = _series(rows, f"dv_outliers_gt_{t}_pct")
    ax2.plot(idx2, app_out, label="this app", color="tab:blue")
    ax2.plot(idx2, dv_out, label="LaVision DaVis", color="tab:orange")
    ax2.set_ylabel(f"outliers > {outlier_threshold} (%)")
    ax2.set_xlabel("pair index")
    ax2.set_title(f"Local-residual outlier rate per snapshot (threshold={outlier_threshold})")
    ax2.legend()
    _status_rug(ax2, rows)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-file", default=None, help="Calibration source only (DaVis .set project) -- "
                                                      "required unless --summarize-only.")
    p.add_argument("--multiset-index", type=int, default=0)
    p.add_argument("--npz-dir", default=None, help="This app's completed batch output folder -- "
                                                     "required unless --summarize-only.")
    p.add_argument("--vc7-dir", default=None, help="DaVis's own B00001.vc7, B00002.vc7, ... (1-based) -- "
                                                     "required unless --summarize-only.")
    p.add_argument("--mode", choices=MODES, default=None,
                    help="Required unless --summarize-only. planar and dual_planar npz files share the "
                         "same filename suffix and cannot be told apart automatically.")
    p.add_argument("--out-dir", default="piv_comparison_output")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=None, help="Optional cap; default processes every "
                                                                 "pair found on both sides.")
    p.add_argument("--outlier-threshold", type=float, default=3.0, choices=OUTLIER_THRESHOLDS,
                    help="Which of field_stats' own fixed residual thresholds to use for the outlier-rate "
                         "trend plot.")
    p.add_argument("--force", action="store_true", help="Recompute every pair, ignoring existing checkpoints.")
    p.add_argument("--summarize-only", action="store_true",
                    help="Skip all npz/vc7 loading; just rebuild summary.csv + plots from existing "
                         "checkpoint/*.json files under --out-dir.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.summarize_only:
        missing = [n for n in ("set_file", "npz_dir", "vc7_dir", "mode") if getattr(args, n) is None]
        if missing:
            parser.error(
                f"--{'/--'.join(m.replace('_', '-') for m in missing)} required unless --summarize-only")
    elif any(getattr(args, n) is not None for n in ("set_file", "npz_dir", "vc7_dir", "mode")):
        print("[info] --summarize-only: ignoring --set-file/--npz-dir/--vc7-dir/--mode")

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.summarize_only:
        os.makedirs(os.path.join(args.out_dir, "checkpoint"), exist_ok=True)

        calib = read_calibration_from_set(args.set_file, args.multiset_index)
        if calib.frame_dt_s is None:
            raise SystemExit(
                "No frame_dt_s was found for this .set -- this app's own velocities would be in "
                "mm/frame, not m/s, and couldn't be compared meaningfully against DaVis's always-m/s "
                ".vc7 output. Real timing is required for this comparison to mean anything.")

        if args.mode == "stereo":
            px_per_mm = read_stereo_calibration_from_set(args.set_file, args.multiset_index).world_scale_px_per_mm
        elif args.mode == "dual_planar":
            try:
                dp = read_dual_planar_calibration_from_set(args.set_file, args.multiset_index)
                px_per_mm = 1.0 / dp.scale_x_mm_per_px
            except Exception as e:
                px_per_mm = 1.0 / calib.pixel_pitch_mm
                print(f"[warn] couldn't read dual_planar scale ({type(e).__name__}: {e}) -- "
                      f"falling back to 1/pixel_pitch_mm={px_per_mm:.4f} px/mm for the residual-eps "
                      f"heuristic only (this does not affect the already-mm npz x,y/u,v themselves)")
        else:  # planar
            px_per_mm = 1.0 / calib.pixel_pitch_mm
        native_scale = 1.0 / (calib.frame_dt_s * px_per_mm)
        print(f"calibration: frame_dt_s={calib.frame_dt_s}  px_per_mm={px_per_mm}  mode={args.mode}")

        npz_ids, vc7_ids = set(discover_npz_pair_ids(args.npz_dir, args.mode)), set(discover_vc7_pair_ids(args.vc7_dir))
        only_npz, only_vc7 = sorted(npz_ids - vc7_ids), sorted(vc7_ids - npz_ids)
        both = sorted(npz_ids & vc7_ids, key=int)
        if only_npz:
            print(f"[warn] {len(only_npz)} npz pair(s) have no matching vc7 (e.g. {only_npz[:5]})")
        if only_vc7:
            print(f"[warn] {len(only_vc7)} vc7 pair(s) have no matching npz (e.g. {only_vc7[:5]})")
        print(f"[info] proceeding with {len(both)} pair(s) present on both sides")

        both = [p for p in both if int(p) >= args.start_index]
        if args.max_pairs is not None:
            both = both[: args.max_pairs]

        n_skipped = 0
        for i, pair_id in enumerate(both):
            if not args.force and os.path.exists(checkpoint_path(args.out_dir, pair_id)):
                n_skipped += 1
                continue
            result = process_pair(pair_id, args.npz_dir, args.vc7_dir, args.mode, native_scale, px_per_mm)
            save_checkpoint(args.out_dir, pair_id, result, args.mode)
            print(f"  [{i + 1}/{len(both)}] pair {pair_id}: status={result['status']}", flush=True)
        if n_skipped:
            print(f"[info] skipped {n_skipped} pair(s) with an existing checkpoint (use --force to redo)")

    csv_path, rows = assemble_summary_csv(args.out_dir)
    print(f"wrote {csv_path} ({len(rows)} row(s))")
    if not rows:
        print("[warn] no rows to plot")
        return

    mode = args.mode
    if mode is None:
        # --summarize-only: recover the mode from whatever a prior run's
        # checkpoints recorded (added to every checkpoint by save_checkpoint),
        # rather than guessing from the data.
        mode = next((r["mode"] for r in rows if r.get("mode")), "planar")

    plot_density_trends(rows, os.path.join(args.out_dir, "density_trends.png"))
    plot_diff_magnitude_trends(rows, os.path.join(args.out_dir, "diff_magnitude_trends.png"), mode)
    plot_correlation_and_outlier_trends(rows, os.path.join(args.out_dir, "correlation_and_outlier_trends.png"),
                                         mode, args.outlier_threshold)
    print(f"wrote {args.out_dir}/density_trends.png, diff_magnitude_trends.png, "
          f"correlation_and_outlier_trends.png")


if __name__ == "__main__":
    main()
