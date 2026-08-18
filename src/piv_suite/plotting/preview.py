"""In-memory preview rendering for the GUI -- replaces
piv_common.preview_first_snapshot()'s blocking terminal y/N prompt with a
figure the GUI's preview_panel can embed (FigureCanvasQTAgg) and let the
user approve/re-tweak-and-re-preview before committing to a batch run.

The CLI keeps the original terminal-prompt behavior (piv_suite.cli.main
calls preview_first_snapshot_cli, not this module) since there's no GUI
event loop to embed a canvas into there.
"""

import os
import sys


def make_preview_figure(mode, x, y, u, v, valid, title, w=None, quiver_scale=1000):
    """Build (not save/show) a matplotlib Figure for the first pair of a
    run, for the GUI to embed directly. mode is "planar" or "stereo"."""
    if mode == "stereo":
        from .stereo import make_stereo_figure
        return make_stereo_figure(x, y, u, v, w, valid, title, quiver_scale=quiver_scale)
    from .planar import make_planar_figure
    return make_planar_figure(x, y, u, v, valid, title, quiver_scale=quiver_scale)


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
