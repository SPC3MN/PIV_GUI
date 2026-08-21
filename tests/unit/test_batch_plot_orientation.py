"""Regression test for plotting/planar.py and plotting/stereo.py's
y-axis orientation -- see test_preview_plotting.py's
test_preview_does_not_double_flip_y_orientation for the full root-cause
account (both these batch-run plotters shared the same double-flip bug
as the preview panel, since all three build on the same
engines.cpu_engine/gpu_engine y-coordinate convention)."""

import numpy as np

from piv_suite.plotting.planar import make_planar_figure
from piv_suite.plotting.stereo import make_stereo_figure


def _field(ny=10, nx=6):
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    return x, y, np.ones((ny, nx), dtype=bool)


def test_planar_batch_plot_is_not_y_inverted():
    x, y, valid = _field()
    u, v = np.zeros_like(x), np.zeros_like(x)
    fig = make_planar_figure(x, y, u, v, valid, title="t")
    ylim = fig.axes[0].get_ylim()
    assert ylim[0] < ylim[1], f"y-axis is inverted (ylim={ylim})"


def test_stereo_batch_plot_is_not_y_inverted():
    x, y, valid = _field()
    U, V, W = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    fig = make_stereo_figure(x, y, U, V, W, valid, title="t")
    ylim = fig.axes[0].get_ylim()
    assert ylim[0] < ylim[1], f"y-axis is inverted (ylim={ylim})"
