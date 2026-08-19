"""In-memory preview rendering for the GUI -- replaces
piv_common.preview_first_snapshot()'s blocking terminal y/N prompt with a
figure the GUI's preview_panel can embed (FigureCanvasQTAgg) and let the
user approve/re-tweak-and-re-preview before committing to a batch run.

Unlike the batch run's per-pair saved plots (plotting.planar/plotting.stereo,
a fixed quiver-only figure), the preview figure supports the GUI's
configurable view: per-component (U, V, and W for stereo) filled contours
with auto or manually-scaled colorbars, an optional vector overlay, and a
choice of colormap -- since a single preview is looked at interactively
this can afford to be heavier/more informative than the plot saved for
every pair of a large batch.

The CLI keeps the original terminal-prompt behavior (piv_suite.cli.main
calls preview_first_snapshot_cli, not this module) since there's no GUI
event loop to embed a canvas into there.
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A short, GUI-facing list of colormaps that are meaningful for signed
# velocity data (diverging) as well as speed/magnitude-like data
# (sequential) -- not matplotlib's entire registry, which is mostly noise
# for this use case.
AVAILABLE_COLORMAPS = ["viridis", "plasma", "turbo", "jet", "coolwarm", "RdBu", "seismic"]

_COMPONENT_LABELS = {"U": "U", "V": "V", "W": "W (out-of-plane)"}


def _resolve_range(masked_data, vrange):
    """vrange is (vmin, vmax), either entry possibly None for "auto from
    data". Returns a concrete (vmin, vmax), falling back to (0, 1) if
    there's no valid data at all to auto-scale from."""
    vmin, vmax = vrange if vrange is not None else (None, None)
    if vmin is None or vmax is None:
        finite = masked_data.compressed()
        auto_min = float(finite.min()) if finite.size else 0.0
        auto_max = float(finite.max()) if finite.size else 1.0
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax
    if vmin >= vmax:
        vmax = vmin + 1e-9  # contourf rejects a zero-width/inverted level range
    return vmin, vmax


def _plot_component(ax, x, y, data, valid, name, show_contour, show_vectors,
                     cmap, vrange, quiver_u, quiver_v, quiver_scale):
    masked = np.ma.masked_where(~valid, data)
    if show_contour:
        vmin, vmax = _resolve_range(masked, vrange)
        levels = np.linspace(vmin, vmax, 21)
        cs = ax.contourf(x, y, masked, levels=levels, cmap=cmap, extend="both")
        ax.figure.colorbar(cs, ax=ax, label=_COMPONENT_LABELS.get(name, name))
    if show_vectors:
        # black-on-contour reads clearly against any of AVAILABLE_COLORMAPS;
        # red-on-blank matches the batch-run quiver plots' look when
        # contours are off.
        color = "black" if show_contour else "red"
        ax.quiver(x[valid], y[valid], quiver_u[valid], quiver_v[valid], color=color, scale=quiver_scale)
    ax.set_title(_COMPONENT_LABELS.get(name, name))
    ax.invert_yaxis()


def make_preview_figure(mode, x, y, u, v, valid, title, w=None, quiver_scale=1000,
                         show_contour=True, show_vectors=False, cmap="viridis", ranges=None):
    """Build (not save/show) a matplotlib Figure for one pair, for the GUI
    to embed directly. mode is "planar" or "stereo".

    show_contour/show_vectors can both be on at once (vectors drawn on
    top of the contour) or both off (an empty-but-labeled axes -- the
    caller's status label still reports the numeric summary).
    `ranges` is an optional {"U"/"V"/"W": (vmin, vmax)} dict; either side
    of a tuple, or the whole entry, may be None for "auto-scale from this
    pair's data" (the default when `ranges` itself is None).
    """
    ranges = ranges or {}
    components = [("U", u), ("V", v)]
    if mode == "stereo" and w is not None:
        components.append(("W", w))

    n = len(components)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axlabel = ("x (world px)", "y (world px)") if mode == "stereo" else ("pixels", "pixels")
    for ax, (name, data) in zip(axes[0], components):
        _plot_component(ax, x, y, data, valid, name, show_contour, show_vectors,
                         cmap, ranges.get(name), u, v, quiver_scale)
        ax.set_xlabel(axlabel[0])
        ax.set_ylabel(axlabel[1])
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def preview_first_snapshot_cli(png_path, prompt="Continue processing the remaining pairs in this set?"):
    """CLI counterpart -- unchanged from piv_common.preview_first_snapshot:
    open the just-saved preview PNG and block on a y/n terminal prompt.
    Exits the process if the user declines."""
    print(f"[info] first-snapshot preview saved to '{png_path}'")
    try:
        if sys.platform == "win32":
            os.startfile(png_path)
        else:
            import subprocess
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, png_path])
    except OSError as e:
        print(f"[warn] couldn't auto-open preview image ({e}) -- open it "
              f"manually: {png_path}")

    while True:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return
        if answer in ("", "n", "no"):
            sys.exit("Aborted after reviewing the first snapshot.")
        print("Please answer 'y' or 'n'.")
