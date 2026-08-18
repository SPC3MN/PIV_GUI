"""Planar (u, v) quiver plotting. Migrated from piv_common.plot_and_save_planar,
generalized from a `ctrl` namespace to explicit kwargs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_planar_figure(x, y, u, v, valid, title, quiver_scale=1000):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(x[valid], y[valid], u[valid], v[valid], color="red", scale=quiver_scale)
    ax.set_title(title)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_and_save_planar(x, y, u, v, valid, out_path, title,
                          quiver_scale=1000, plot_dpi=150, show_plots=False):
    fig = make_planar_figure(x, y, u, v, valid, title, quiver_scale=quiver_scale)
    fig.savefig(out_path, dpi=plot_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)
