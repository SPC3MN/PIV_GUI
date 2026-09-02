"""In-memory preview rendering for the GUI -- replaces
piv_common.preview_first_snapshot()'s blocking terminal y/N prompt with a
figure the GUI's preview_panel can embed (FigureCanvasQTAgg) and let the
user approve/re-tweak-and-re-preview before committing to a batch run.

Renders a SINGLE velocity-magnitude field (sqrt(u**2+v**2), or
sqrt(u**2+v**2+w**2) when a stereo w is given) rather than the previous
one-subplot-per-component layout -- magnitude ("does this flow look
physically sane") is what a sanity-check preview is actually for, and
per-component subplots with independent manual color ranges never earned
their UI complexity here. The presentation is fixed, not user-
configurable: the filled contour is always on (there's nothing else TO
turn off once there's only one field to show), it's always auto-scaled
to that field's own min/max (comparing pairs on a pinned scale was never
actually exercised), and the colormap is hard-coded to "turbo" (a
perceptually-uniform sequential map, right for a magnitude/speed field
the way a diverging map is right for a signed single component -- not a
meaningful per-user choice any more with only one field). The one thing
still toggleable is an optional (u, v) quiver overlay -- see
preview_panel.py for why it's only enabled once a real preview result
exists to draw it on top of.

units labels the colorbar explicitly (e.g. "m/s" or "px/frame") instead
of assuming -- a correctly-computed m/s field with an unlabeled colorbar
reads as "these numbers look wrong" even when they aren't, which is very
plausibly what actually prompted a real "the preview units look
incorrect" report before this label existed at all. See
preview_panel.py's _compute_planar/_compute_dual_planar/_compute_stereo
for how `units` is decided from whether real calibration was available.

Unlike the batch run's per-pair saved plots (plotting.planar/
plotting.stereo, a fixed quiver-only figure), this can afford a heavier
single interactive render since only one pair (or one averaged range,
see preview_panel.py's _compute_range) is ever shown at a time.

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

# Hard-coded, not a GUI choice any more -- see this module's docstring.
MAGNITUDE_CMAP = "turbo"


def _auto_range(masked_data):
    """(vmin, vmax) auto-scaled from masked_data's own valid values.
    Magnitude is never negative so this can't invert the way a signed
    component's range could, but still needs a fallback for "no valid
    data at all" (every cell rejected) and a nudge for contourf's own
    zero-width-level-range rejection when every valid cell is identical."""
    finite = masked_data.compressed()
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-9
    return vmin, vmax


def make_preview_figure(mode, x, y, u, v, valid, title, w=None, units="m/s",
                         show_vectors=False, quiver_scale=1000, figsize=(7, 6)):
    """Build (not save/show) a matplotlib Figure showing ONE velocity-
    magnitude field for the GUI to embed directly. mode is "planar" or
    "stereo" -- used only to label the axes (stereo's grid is DaVis's
    dewarped "world px" canvas; planar's is the raw frame's own pixel
    grid), not to branch the plotting logic itself the way the previous
    per-component version did.

    magnitude = sqrt(u**2+v**2), or sqrt(u**2+v**2+w**2) when w is given
    (stereo's 3-component result) -- w=None vs w=array IS the planar/
    stereo distinction for the actual math, matching what the caller has
    on hand; `mode` only affects the axis label text.

    show_vectors overlays a quiver of the in-plane (u, v) direction on
    top of the magnitude contour -- always ON TOP, never an alternative
    to it (there's no contour-off mode any more; magnitude IS the plot).

    figsize is in inches and should be derived from the widget the figure
    will be shown in (see preview_panel._figsize_for_canvas), so the plot
    fills the space available instead of being letterboxed inside a
    hard-coded one. It only sets the CANVAS shape -- the data's own shape
    is preserved independently by the equal aspect below."""
    magnitude = np.sqrt(u**2 + v**2) if w is None else np.sqrt(u**2 + v**2 + w**2)
    masked = np.ma.masked_where(~valid, magnitude)
    vmin, vmax = _auto_range(masked)
    levels = np.linspace(vmin, vmax, 21)

    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(x, y, masked, levels=levels, cmap=MAGNITUDE_CMAP, extend="both")
    # EQUAL ASPECT, and it is not cosmetic. A PIV field is a picture of a
    # physical region, so the preview's whole job -- "does this flow look
    # right" -- depends on shape being preserved. matplotlib's default
    # aspect="auto" stretches the data to fill whatever axes box it is given:
    # measured at 1.881x on a real 379x704 stereo grid, i.e. a round vortex
    # rendered 88% taller than wide, which reads as a physically wrong flow.
    ax.set_aspect("equal")
    fig.colorbar(cs, ax=ax, label=f"|V| ({units})")
    if show_vectors:
        ax.quiver(x[valid], y[valid], u[valid], v[valid], color="black", scale=quiver_scale)
    axlabel = ("x (world px)", "y (world px)") if mode == "stereo" else ("pixels", "pixels")
    ax.set_xlabel(axlabel[0])
    ax.set_ylabel(axlabel[1])
    # NOT ax.invert_yaxis() -- see plotting/planar.py's make_planar_figure
    # for the full account (same root cause, same fix, every plotting
    # module shares this). `y` already comes flipped out of
    # engines.cpu_engine/gpu_engine (row 0, the image's physical top,
    # gets the LARGEST y value) specifically so it renders correctly
    # under matplotlib's default axis orientation; inverting again here
    # rendered every preview upside down. Confirmed with a synthetic
    # marker at the physical top of a real frame, which rendered at the
    # BOTTOM of the plot before this fix -- and reported independently by
    # a user comparing this preview against a real DaVis render of the
    # same data (the flow pattern looked structurally right but
    # vertically mirrored).
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
