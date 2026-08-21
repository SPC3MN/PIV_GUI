"""Side-by-side velocity-field comparison: this program vs a real
LaVision DaVis export, with an explicit OUTLIER analysis.

Complements scripts/compare_davis_lavision.py, which answers "how well
do the two agree on average" (correlation coefficients, mean/median
absolute difference). This one answers a different question that
aggregate correlation hides: "is this program leaving spurious vectors
in the field that DaVis rejected?" -- i.e. outliers, which can be a
small fraction of vectors (barely moving corr()) while completely
dominating what the field LOOKS like in a quiver plot, and while badly
skewing downstream derivatives (structure functions, dissipation-rate
estimates) that are sensitive to isolated bad vectors.

Outlier metric: each vector's NORMALIZED LOCAL RESIDUAL -- its distance
from its own 3x3 neighborhood median, normalized by that neighborhood's
median absolute deviation. This is the same Westerweel & Scarano
"universal outlier detection" quantity that processing.postprocess.
range_filter thresholds on (and that DaVis's own "remove if residual >
removal factor" step uses), so a vector reported as an outlier here is
an outlier by exactly the criterion both programs claim to apply --
not by some third definition invented for this script.

Run:
    python scripts/compare_velocity_fields.py --lavision-dir "...\\Lavision_Sample"

Produces per-pair PNGs (magnitude maps, outlier-location maps, residual
histograms) plus a printed table, into --out-dir (default:
piv_comparison_output/).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_davis_lavision import (
    PX_PER_MM, build_config, find_im7_files, find_vc7_for, load_vc7_field,
    read_pair, run_this_app,
)
from piv_suite.config.schema import ProjectConfig
from piv_suite.io.buffers import frames_from_buffer


def gui_default_config():
    """ProjectConfig exactly as the GUI presents it on a fresh start --
    every schema default untouched. This is what a user actually gets by
    launching the app and hitting Preview without changing settings, and
    it is NOT the same as build_config()'s DaVis-matched setup (see
    build_config in compare_davis_lavision.py): notably
    min_max_filter_enabled defaults False (vs True/length-4 for this
    dataset's DaVis job) and range_filter.residual_max defaults 3.0 (vs
    DaVis's own removal factor of 2.0). Included here so the comparison
    shows what those default choices actually cost on real data, rather
    than only ever reporting the tuned configuration's numbers."""
    return ProjectConfig()


def normalized_local_residual(u, v, size=1, eps=0.1):
    """Westerweel & Scarano universal outlier detection statistic, per
    vector -- the same quantity processing.postprocess.range_filter
    thresholds on. NaN (already-rejected/absent) vectors are excluded
    from their neighbours' medians rather than treated as zeros, so a
    vector next to a hole isn't penalized for the hole.

    THE STATISTIC IS NOT SCALE-INVARIANT, because of `eps`. Everything
    else here is a ratio of like-united quantities and would be, but eps
    is an absolute floor added to the neighbourhood MAD (it represents
    expected measurement noise, conventionally ~0.1 PIXEL, and exists to
    stop a locally-uniform neighbourhood with a near-zero MAD from
    dividing a tiny deviation into a huge residual). So eps only carries
    its intended meaning if u/v are in px/frame: on the same field
    expressed in mm/s (~70x larger numbers for this dataset) eps becomes
    negligible, the floor stops doing its job, and the reported outlier
    rate inflates several-fold. Measured directly: the same real field
    scored 2.9% outliers (>3) in px/frame vs 27% in mm/s. Callers must
    pass px/frame (see main(), which converts every field back to
    px/frame before scoring) or scale eps to match their units.

    Returns an array of the same shape, NaN where the vector itself is
    NaN (nothing to score)."""
    from numpy.lib.stride_tricks import sliding_window_view
    import warnings

    k = 2 * size + 1
    center = k * k // 2

    def windows_of(a):
        padded = np.pad(a.astype(np.float64), size, mode="constant", constant_values=np.nan)
        return sliding_window_view(padded, (k, k)).reshape(a.shape[0], a.shape[1], k * k)

    wu, wv = windows_of(u), windows_of(v)
    # exclude each vector itself from its own neighbourhood statistics
    wu_excl, wv_excl = wu.copy(), wv.copy()
    wu_excl[..., center] = np.nan
    wv_excl[..., center] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        um = np.nanmedian(wu_excl, axis=-1)
        vm = np.nanmedian(wv_excl, axis=-1)
        rm_u = np.nanmedian(np.abs(wu_excl - um[..., None]), axis=-1)
        rm_v = np.nanmedian(np.abs(wv_excl - vm[..., None]), axis=-1)

    with np.errstate(invalid="ignore", divide="ignore"):
        r_u = np.abs(u - um) / (rm_u + eps)
        r_v = np.abs(v - vm) / (rm_v + eps)
        return np.sqrt(r_u ** 2 + r_v ** 2)


def field_stats(label, u, v, residual_thresholds=(2.0, 3.0, 5.0, 10.0)):
    finite = np.isfinite(u) & np.isfinite(v)
    n_total = u.size
    n_valid = int(finite.sum())
    mag = np.hypot(u, v)

    # Score residuals on the px/frame form of the field, never the mm/s
    # form -- normalized_local_residual's eps floor is defined in pixels
    # and silently stops working otherwise (see its docstring).
    resid = normalized_local_residual(u / PX_PER_MM_TO_MM_S, v / PX_PER_MM_TO_MM_S)
    resid_valid = resid[finite & np.isfinite(resid)]

    stats = {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "pct_valid": 100.0 * n_valid / n_total,
        "mag_mean": np.nanmean(mag),
        "mag_p99": np.nanpercentile(mag[finite], 99) if n_valid else np.nan,
        "mag_max": np.nanmax(mag[finite]) if n_valid else np.nan,
        "resid_median": np.median(resid_valid) if resid_valid.size else np.nan,
        "resid_p99": np.percentile(resid_valid, 99) if resid_valid.size else np.nan,
    }
    for t in residual_thresholds:
        n_out = int((resid_valid > t).sum())
        stats[f"outliers_gt_{t}"] = n_out
        stats[f"outliers_gt_{t}_pct"] = 100.0 * n_out / max(1, resid_valid.size)
    return stats, resid


def print_stats_table(all_stats, thresholds=(2.0, 3.0, 5.0, 10.0)):
    labels = [s["label"] for s in all_stats]
    width = max(len(l) for l in labels) + 2

    def row(name, fmt, key):
        cells = "".join(f"{fmt.format(s[key]):>18}" for s in all_stats)
        print(f"  {name:<28}{cells}")

    print(f"  {'':<28}" + "".join(f"{l:>18}" for l in labels))
    print("  " + "-" * (28 + 18 * len(all_stats)))
    row("valid vectors", "{:.0f}", "n_valid")
    row("valid %", "{:.2f}%", "pct_valid")
    row("mean |velocity|", "{:.1f}", "mag_mean")
    row("p99 |velocity|", "{:.1f}", "mag_p99")
    row("max |velocity|", "{:.1f}", "mag_max")
    row("median local residual", "{:.2f}", "resid_median")
    row("p99 local residual", "{:.2f}", "resid_p99")
    for t in thresholds:
        row(f"outliers (residual > {t})", "{:.0f}", f"outliers_gt_{t}")
        row(f"   as % of valid", "{:.3f}%", f"outliers_gt_{t}_pct")


def plot_comparison(out_path, fields, resids, pair_id, outlier_threshold=3.0):
    """fields/resids: list of (label, u, v) / (label, residual). One
    column per field: velocity magnitude on top, outlier locations
    below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fields)
    fig, axes = plt.subplots(3, n, figsize=(6.2 * n, 14), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(3, 1)

    mags = [np.hypot(u, v) for _, u, v in fields]
    # Scale every column to the SAME robust range, taken from the
    # reference (last) field -- DaVis. A shared scale is what makes the
    # columns visually comparable at all; taking it from the reference
    # (rather than from the pooled max) keeps one column's outliers from
    # compressing every other column into a flat dark rectangle.
    ref_mag = mags[-1]
    vmax = np.nanpercentile(ref_mag[np.isfinite(ref_mag)], 99)

    for col, ((label, u, v), (_, resid), mag) in enumerate(zip(fields, resids, mags)):
        ax = axes[0, col]
        im = ax.imshow(mag, origin="upper", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{label}\nvelocity magnitude", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="|V| (mm/s)")

        ax = axes[1, col]
        outliers = np.isfinite(resid) & (resid > outlier_threshold)
        n_out = int(outliers.sum())
        finite = np.isfinite(u) & np.isfinite(v)
        pct = 100.0 * n_out / max(1, int(finite.sum()))
        # Three-state map (imshow, not scatter -- at ~190k grid points a
        # scatter marker large enough to see swamps the plot into solid
        # red regardless of the real outlier density): grey = clean
        # vector, red = outlier, white = no vector at all.
        disp = np.full(u.shape + (3,), 1.0)          # white background = no vector
        disp[finite] = (0.62, 0.62, 0.62)             # grey = clean
        disp[outliers] = (0.85, 0.05, 0.05)           # red = outlier
        ax.imshow(disp, origin="upper", interpolation="nearest")
        ax.set_title(f"outliers: local residual > {outlier_threshold}\n"
                      f"{n_out} vectors ({pct:.1f}% of valid)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2, col]
        rv = resid[np.isfinite(resid)]
        ax.hist(np.clip(rv, 0, 15), bins=80, color="steelblue", log=True)
        ax.axvline(outlier_threshold, color="red", ls="--", lw=1.2,
                    label=f"threshold {outlier_threshold}")
        ax.set_xlabel("normalized local residual (clipped at 15)")
        ax.set_ylabel("count (log)")
        ax.set_title(f"residual distribution (median {np.median(rv):.2f})", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle(f"Velocity-field comparison -- {pair_id}", fontsize=14)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lavision-dir",
                        help="DaVis sample dir containing B*.im7 and PIV_MP*/PostProc/B*.vc7")
    parser.add_argument("--set-file",
                        help="Alternative input: a DaVis .set (e.g. a StreamSet of .ims "
                             "recordings) read through the SAME io.davis_set path the GUI "
                             "uses, rather than loose .im7 files. Requires --vc7-dir.")
    parser.add_argument("--vc7-dir",
                        help="With --set-file: directory of DaVis's own result vectors "
                             "(B00001.vc7, B00002.vc7, ... 1-based, matching set order).")
    parser.add_argument("--start-index", type=int, default=0,
                        help="With --set-file: first pair index to process (0-based).")
    parser.add_argument("--out-dir", default="piv_comparison_output")
    parser.add_argument("--outlier-threshold", type=float, default=3.0,
                        help="normalized local residual above which a vector is called an outlier")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--skip-gui-defaults", action="store_true",
                        help="only run the DaVis-matched (tuned) config, not the GUI-default one")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if bool(args.set_file) == bool(args.lavision_dir):
        parser.error("give exactly one of --lavision-dir or --set-file")
    if args.set_file and not args.vc7_dir:
        parser.error("--set-file requires --vc7-dir")

    if args.lavision_dir:
        im7_files = find_im7_files(args.lavision_dir)
        if args.max_pairs:
            im7_files = im7_files[:args.max_pairs]

        def _iter_loose():
            for im7_path in im7_files:
                pid = os.path.splitext(os.path.basename(im7_path))[0]
                fa, fb = read_pair(im7_path)
                yield pid, fa, fb, find_vc7_for(args.lavision_dir, im7_path)

        pair_iter = _iter_loose()
    else:
        # Read through io.davis_set -- the SAME ingestion path the GUI and
        # CLI use for a .set project, so this exercises the real code path
        # rather than a loose-file shortcut that happens to look similar.
        from piv_suite.io.davis_set import _open_dataset

        def _iter_set():
            dataset, owns = _open_dataset(args.set_file, 0)
            try:
                n = len(dataset)
                start = args.start_index
                end = min(n, start + (args.max_pairs or n))
                print(f"set contains {n} pair(s); processing indices {start}..{end - 1}")
                for i in range(start, end):
                    buf = dataset[i]
                    fa, fb = frames_from_buffer(buf)
                    # DaVis names its per-pair results B00001.vc7 onward,
                    # 1-based against set order -- index i maps to i+1.
                    vc7 = os.path.join(args.vc7_dir, f"B{i + 1:05d}.vc7")
                    if not os.path.exists(vc7):
                        raise FileNotFoundError(f"no DaVis result at {vc7} for set index {i}")
                    yield f"{i:04d}", fa, fb, vc7
            finally:
                if owns:
                    dataset.close()

        pair_iter = _iter_set()

    configs = [("this app (DaVis-matched)", build_config())]
    if not args.skip_gui_defaults:
        configs.insert(0, ("this app (GUI defaults)", gui_default_config()))

    for pair_id, frame_a, frame_b, vc7_path in pair_iter:
        print(f"\n=== {pair_id} ===")

        fields = []
        for label, cfg in configs:
            print(f"  running {label} ...")
            fa, fb = frame_a.copy(), frame_b.copy()
            x_mm, y_mm, u, v, _elapsed = run_this_app(cfg, fa, fb)
            if cfg.calibration.pixel_pitch_mm is None:
                # GUI-default config leaves apply_calibration a no-op, so
                # what comes back is px/frame -- except run_this_app has
                # already multiplied by 1000 on the way out (its own
                # m/s -> mm/s step, which is meaningless for an
                # uncalibrated run). Undo that 1000 first, THEN convert
                # px/frame -> mm/s, so the magnitude maps/statistics are
                # comparable at all. Pure unit handling applied after the
                # fact, not a change to what was computed -- and it does
                # not affect the residual statistic either way, which is
                # scale-invariant by construction (a ratio of like-united
                # quantities).
                u = (u / 1000.0) * PX_PER_MM_TO_MM_S
                v = (v / 1000.0) * PX_PER_MM_TO_MM_S
            fields.append((label, u, v))

        _x_dv, _y_dv, u_dv, v_dv = load_vc7_field(vc7_path)
        fields.append(("LaVision DaVis", u_dv, v_dv))

        all_stats, all_resids = [], []
        for label, u, v in fields:
            stats, resid = field_stats(label, u, v)
            all_stats.append(stats)
            all_resids.append((label, resid))

        print()
        print_stats_table(all_stats)

        png = os.path.join(args.out_dir, f"{pair_id}_field_comparison.png")
        plot_comparison(png, fields, all_resids, pair_id, args.outlier_threshold)
        print(f"\n  wrote {png}")


# px/frame -> mm/s for this dataset (pixel pitch 1/19.42 mm/px, dt 700us)
PX_PER_MM_TO_MM_S = (1.0 / PX_PER_MM) / 700e-6


if __name__ == "__main__":
    main()
