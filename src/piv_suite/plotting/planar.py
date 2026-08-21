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
    # NOT ax.invert_yaxis(). `y` here already comes flipped out of
    # engines.cpu_engine.init_cpu_processor / gpu_engine's equivalent
    # (`y = frame_shape[0] * scaling_par - y`, converting from openpiv's
    # raw row-major convention -- row 0 = smallest y -- to a y-up
    # convention where row 0, the physical TOP of the image, already gets
    # the LARGEST y value). That alone renders correctly under
    # matplotlib's default axis orientation (larger y plotted higher).
    # Calling invert_yaxis() on top of it flips a second time, rendering
    # the field upside down -- confirmed with a synthetic marker placed
    # at the physical top of a real frame, which rendered at the BOTTOM
    # of the plot before this fix. Reported by a user comparing this
    # program's preview against a real LaVision DaVis render of the same
    # data: the flow pattern looked structurally right but vertically
    # mirrored, which a double-flip explains exactly (the underlying u/v
    # values were already confirmed correct against DaVis -- corr(U) and
    # corr(V) both positive -- so this was a display-only bug, not a
    # sign error in the computed vectors).
    fig.tight_layout()
    return fig


def plot_and_save_planar(x, y, u, v, valid, out_path, title,
                          quiver_scale=1000, plot_dpi=150, show_plots=False):
    fig = make_planar_figure(x, y, u, v, valid, title, quiver_scale=quiver_scale)
    fig.savefig(out_path, dpi=plot_dpi)
    if show_plots:
        plt.show()
    plt.close(fig)
