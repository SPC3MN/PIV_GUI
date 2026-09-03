r"""Build comparison figures for a batch run against DaVis: per-pair density
and correlation trends, ensemble-mean fields side by side, a parity scatter,
and centreline profiles.

Reads this app's finished npz batch output and DaVis's own vc7 vectors for the
same pair indices, resamples DaVis onto this app's grid (RegularGridInterpolator
-- DaVis's field is a regular grid), and writes PNGs plus a summary.json of the
headline numbers so a report can quote measured values rather than restated
ones.

    python scripts/make_comparison_plots.py --mode planar \
        --npz-dir piv_output --vc7-dir "C:\data\MyRecording\PIV_MPd(...)" \
        --px-per-mm 19.42 --out-dir comparison_figures --label "My recording"
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from compare_davis_lavision import load_vc7_field
from compare_stereo_preview import load_vc7_stereo_field

INK = "#12212B"
APP_C = "#0F6E78"
DV_C = "#C2571A"
GRID_C = "#D8DEE3"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def load_pair(npz_path, vc7_path, mode, px_per_mm):
    """-> (x_mm, y_mm, app u/v/w masked to valid, DaVis u/v/w on the app grid)."""
    d = np.load(npz_path)
    if mode == "stereo":
        x, y = d["x"] / px_per_mm, d["y"] / px_per_mm
        u, v, w, valid = d["U"] * 1000, d["V"] * 1000, d["W"] * 1000, d["valid"]
        xd, yd, ud, vd, wd = load_vc7_stereo_field(vc7_path)
    else:
        x, y = d["x"] / px_per_mm, d["y"] / px_per_mm
        u, v, valid = d["u"] * 1000, d["v"] * 1000, d["valid"]
        w, wd = None, None
        xd, yd, ud, vd = load_vc7_field(vc7_path)

    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)
    if w is not None:
        w = np.where(valid, w, np.nan)

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
    # DaVis's V carries the opposite sign convention on these projects.
    ok = ~np.isnan(u) & ~np.isnan(ur)
    if ok.sum() > 100 and np.corrcoef(v[ok], vr[ok])[0, 1] < 0:
        vr = -vr
    # DaVis's density has to be measured on DaVis's OWN grid. Measuring it
    # after resampling onto this app's grid flatters DaVis: the app's grid is
    # inset half an interrogation window from the canvas edge, so it simply
    # never asks about the border cells where DaVis's own field is invalid
    # (99.93% vs a true 98.11% on one real planar pair).
    dv_density = 100.0 * np.isfinite(ud).sum() / ud.size
    return xc, yc, u, v, w, ur, vr, wr, valid, dv_density


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("planar", "stereo"), required=True)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--vc7-dir", required=True)
    p.add_argument("--px-per-mm", type=float, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    suffix = "_stereo_velocity.npz" if args.mode == "stereo" else "_velocity.npz"
    ids = sorted(int(f[:4]) for f in os.listdir(args.npz_dir) if f.endswith(suffix))
    print("%d pair(s) found in %s" % (len(ids), args.npz_dir))
    if not ids:
        print("nothing to plot -- skipping")
        return

    rows = []
    sum_app = sum_dv = None
    n_app = n_dv = None
    for k, i in enumerate(ids):
        npz = os.path.join(args.npz_dir, "%04d%s" % (i, suffix))
        vc7 = os.path.join(args.vc7_dir, "B%05d.vc7" % (i + 1))
        if not os.path.exists(vc7):
            continue
        x, y, u, v, w, ur, vr, wr, valid, dv_density = load_pair(npz, vc7, args.mode, args.px_per_mm)
        ok = ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(ur) & ~np.isnan(vr)
        if wr is not None:
            ok = ok & ~np.isnan(w) & ~np.isnan(wr)
        row = {"pair": i,
               "app_density": 100.0 * valid.sum() / valid.size,
               "app_n_valid": int(valid.sum()), "app_n_total": int(valid.size),
               "dv_density": dv_density,
               "corr_u": float(np.corrcoef(u[ok], ur[ok])[0, 1]),
               "corr_v": float(np.corrcoef(v[ok], vr[ok])[0, 1]),
               "mean_abs_diff": float(np.hypot(u[ok] - ur[ok], v[ok] - vr[ok]).mean()),
               "app_mag_mean": float(np.nanmean(np.hypot(u, v))),
               "dv_mag_mean": float(np.nanmean(np.hypot(ur, vr))),
               "n": int(ok.sum())}
        if wr is not None:
            row["corr_w"] = float(np.corrcoef(w[ok], wr[ok])[0, 1])
            row["mean_abs_diff_w"] = float(np.abs(w[ok] - wr[ok]).mean())
        rows.append(row)

        # Running ensemble means, counting only cells each side actually measured.
        comps_app = [u, v] + ([w] if w is not None else [])
        comps_dv = [ur, vr] + ([wr] if wr is not None else [])
        if sum_app is None:
            sum_app = [np.zeros_like(u) for _ in comps_app]
            sum_dv = [np.zeros_like(u) for _ in comps_dv]
            n_app = np.zeros_like(u)
            n_dv = np.zeros_like(u)
        m_app = ~np.isnan(u)
        m_dv = ~np.isnan(ur)
        for s, c in zip(sum_app, comps_app):
            s += np.where(m_app, np.nan_to_num(c), 0.0)
        for s, c in zip(sum_dv, comps_dv):
            s += np.where(m_dv, np.nan_to_num(c), 0.0)
        n_app += m_app
        n_dv += m_dv
        if (k + 1) % 10 == 0:
            print("  ...%d/%d" % (k + 1, len(ids)))

    xs = np.array([r["pair"] for r in rows])
    mean_app = [np.where(n_app > 0, s / np.maximum(n_app, 1), np.nan) for s in sum_app]
    mean_dv = [np.where(n_dv > 0, s / np.maximum(n_dv, 1), np.nan) for s in sum_dv]

    # ---- figure 1: density + correlation per pair ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    ax = axes[0]
    ax.plot(xs, [r["app_density"] for r in rows], color=APP_C, lw=1.4, label="this app")
    ax.plot(xs, [r["dv_density"] for r in rows], color=DV_C, lw=1.4, label="DaVis")
    _style(ax, "Vector density per pair", ylabel="valid vectors (%)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(xs, [r["corr_u"] for r in rows], color=APP_C, lw=1.2, label="corr(U)")
    ax.plot(xs, [r["corr_v"] for r in rows], color=DV_C, lw=1.2, label="corr(V)")
    if "corr_w" in rows[0]:
        ax.plot(xs, [r["corr_w"] for r in rows], color="#6B4FA8", lw=1.2, label="corr(W)")
    _style(ax, "Agreement with DaVis, per pair", xlabel="pair index", ylabel="correlation")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("%s -- %d pairs vs DaVis" % (args.label or args.mode, len(rows)), color=INK)
    fig.savefig(os.path.join(args.out_dir, "density_and_correlation.png"), dpi=140)
    plt.close(fig)

    # ---- figure 2: ensemble-mean fields ----
    names = ["U", "V"] + (["W"] if len(mean_app) == 3 else [])
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3.1 * len(names)), layout="constrained")
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        a, b = mean_app[r], mean_dv[r]
        both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
        lo, hi = np.percentile(both, [1, 99])
        lim = max(abs(lo), abs(hi))
        for c, (field, title) in enumerate([(a, "this app"), (b, "DaVis"), (a - b, "difference")]):
            ax = axes[r, c]
            vmax = lim if c < 2 else np.nanpercentile(np.abs(a - b), 99)
            im = ax.pcolormesh(x, y, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_aspect("equal")
            ax.set_title("mean %s -- %s" % (name, title), color=INK, fontsize=9)
            ax.tick_params(labelsize=7, colors=INK)
            ax.set_xlabel("x (mm)", color=INK, fontsize=8)
            if c == 0:
                ax.set_ylabel("y (mm)", color=INK, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle("Ensemble mean over %d pairs (mm/s)" % len(rows), color=INK)
    fig.savefig(os.path.join(args.out_dir, "ensemble_mean_fields.png"), dpi=110)
    plt.close(fig)

    # ---- figure 2b: parity -- every compared vector, this app vs DaVis ----
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.8), layout="constrained")
    axes = np.atleast_1d(axes)
    for r, name in enumerate(names):
        a, b = mean_app[r], mean_dv[r]
        ok = np.isfinite(a) & np.isfinite(b)
        ax = axes[r]
        lim = np.nanpercentile(np.abs(np.concatenate([a[ok], b[ok]])), 99.5)
        ax.hexbin(b[ok], a[ok], gridsize=70, bins="log", cmap="magma_r",
                  extent=(-lim, lim, -lim, lim))
        ax.plot([-lim, lim], [-lim, lim], color=DV_C, lw=1.0, ls="--")
        ax.set_aspect("equal")
        _style(ax, "mean %s   (r = %.4f)" % (name, np.corrcoef(a[ok], b[ok])[0, 1]),
               xlabel="DaVis (mm/s)", ylabel="this app (mm/s)" if r == 0 else None)
    fig.suptitle("Ensemble-mean parity, %d pairs" % len(rows), color=INK)
    fig.savefig(os.path.join(args.out_dir, "parity.png"), dpi=120)
    plt.close(fig)

    # ---- figure 3: mean profiles through the field centre ----
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), layout="constrained")
    axes = np.atleast_1d(axes)
    mid = mean_app[0].shape[0] // 2
    for r, name in enumerate(names):
        ax = axes[r]
        ax.plot(x[mid], mean_app[r][mid], color=APP_C, lw=1.4, label="this app")
        ax.plot(x[mid], mean_dv[r][mid], color=DV_C, lw=1.4, ls="--", label="DaVis")
        _style(ax, "mean %s, horizontal centreline" % name, xlabel="x (mm)", ylabel="mm/s")
        if r == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(args.out_dir, "mean_profiles.png"), dpi=140)
    plt.close(fig)

    def agg(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else None

    summary = {
        "mode": args.mode, "label": args.label, "n_pairs": len(rows),
        "app_density_mean": agg("app_density"), "dv_density_mean": agg("dv_density"),
        "corr_u_mean": agg("corr_u"), "corr_v_mean": agg("corr_v"), "corr_w_mean": agg("corr_w"),
        "mean_abs_diff_mean": agg("mean_abs_diff"), "mean_abs_diff_w_mean": agg("mean_abs_diff_w"),
        "app_mag_mean": agg("app_mag_mean"), "dv_mag_mean": agg("dv_mag_mean"),
        "per_pair": rows,
    }
    # Ensemble-mean-field agreement -- the "final mean results" number.
    for r, name in enumerate(names):
        a, b = mean_app[r], mean_dv[r]
        ok = np.isfinite(a) & np.isfinite(b)
        summary["ensemble_corr_%s" % name] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        summary["ensemble_mean_abs_diff_%s" % name] = float(np.abs(a[ok] - b[ok]).mean())
        summary["ensemble_app_mean_%s" % name] = float(a[ok].mean())
        summary["ensemble_dv_mean_%s" % name] = float(b[ok].mean())
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n== %s (%d pairs) ==" % (args.label or args.mode, len(rows)))
    print("density      app %.2f%%   DaVis %.2f%%" % (summary["app_density_mean"], summary["dv_density_mean"]))
    print("per-pair     corr(U) %.4f  corr(V) %.4f%s   mean|diff| %.2f mm/s"
          % (summary["corr_u_mean"], summary["corr_v_mean"],
             "  corr(W) %.4f" % summary["corr_w_mean"] if summary["corr_w_mean"] else "",
             summary["mean_abs_diff_mean"]))
    for name in names:
        print("ensemble %-2s  corr %.4f  mean|diff| %.2f mm/s  (app mean %.2f, DaVis mean %.2f)"
              % (name, summary["ensemble_corr_%s" % name], summary["ensemble_mean_abs_diff_%s" % name],
                 summary["ensemble_app_mean_%s" % name], summary["ensemble_dv_mean_%s" % name]))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
