"""Stereo (U, V, W) quiver plotting, colored by out-of-plane W. Migrated
from stereo_common.plot_and_save_stereo, generalized from a `ctrl`
namespace to explicit kwargs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_stereo_figure(x, y, U, V, W, valid, title, quiver_scale=1000):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.quiver(x[valid], y[valid], U[valid], V[valid], W[valid],
                    cmap="coolwarm", scale=quiver_scale)
    fig.colorbar(sc, ax=ax, label="W (out-of-plane)")
    ax.set_title(title)
    ax.set_xlabel("x (world px)")
    ax.set_ylabel("y (world px)")
    # NOT ax.invert_yaxis() -- see plotting/planar.py's make_planar_figure
    # for the full account. Same root cause here: `y` (and this stereo
    # pair's dewarped `x, y`) comes from the same engines.cpu_engine/
    # gpu_engine coordinate transform regardless of whether the frame was
    # raw or dewarped onto a world grid, so it already renders correctly
    # under matplotlib's default (non-inverted) orientation.
    fig.tight_layout()
    return fig


def plot_and_save_stereo(x, y, U, V, W, valid, out_path, title,
                          quiver_scale=1000, plot_dpi=150, show_plots=False):
    fig = make_stereo_figure(x, y, U, V, W, valid, title, quiver_scale=quiver_scale)
    fig.savefig(out_path, dpi=plot_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)
