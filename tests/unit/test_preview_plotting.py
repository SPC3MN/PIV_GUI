"""Regression tests for plotting/preview.py's single-magnitude-field
preview figure -- auto-scaled colorbar (labeled with real units), optional
vector overlay, hard-coded colormap. No Qt/GUI needed, matplotlib alone
(Agg backend, set in the module under test)."""

import numpy as np
from matplotlib.quiver import Quiver

from piv_suite.plotting.preview import MAGNITUDE_CMAP, _auto_range, make_preview_figure


def _planar_field(nx=5, ny=4):
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    u = x.copy()
    v = y.copy()
    valid = np.ones_like(x, dtype=bool)
    return x, y, u, v, valid


def test_preview_does_not_double_flip_y_orientation():
    # Regression test for a real display bug, reported by a user
    # comparing this program's preview against a real LaVision DaVis
    # render of the same data: the flow pattern looked structurally
    # right but vertically mirrored. Root cause: engines.cpu_engine/
    # gpu_engine already flip y into a convention where the image's
    # physical TOP row gets the LARGEST y value (specifically so it
    # renders correctly under matplotlib's default, non-inverted y-axis
    # orientation) -- the plotting code used to call ax.invert_yaxis() on
    # top of that, flipping a second time. This builds a field with an
    # unambiguous marker at the LARGEST y (standing in for the image's
    # physical top, per that convention) and checks it renders in the
    # upper half of the axes, not the lower half.
    ny, nx = 10, 6
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    v[-1, :] = 100.0  # marker at the LARGEST y (last grid row)
    valid = np.ones((ny, nx), dtype=bool)

    fig = make_preview_figure("planar", x, y, u, v, valid, title="t")
    data_axes = fig.axes[0]
    ylim = data_axes.get_ylim()
    # NOT inverted: get_ylim()[0] (bottom of the displayed axes) must be
    # the SMALLER data value, matching matplotlib's default orientation
    # (larger y plotted higher) -- an inverted axes would report ylim
    # with the larger value first.
    assert ylim[0] < ylim[1], (
        f"y-axis is inverted (ylim={ylim}) -- the marker at the largest "
        "y (the image's physical top, per engines.cpu_engine's "
        "convention) would render at the BOTTOM of the plot"
    )


def test_default_produces_one_data_axes_and_one_colorbar():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t")
    # 1 data axes + 1 colorbar axes -- a single magnitude field, not one
    # subplot per component the way the previous version had.
    assert len(fig.axes) == 2


def test_magnitude_is_planar_pythagorean_when_no_w_given():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t")
    cs = fig.axes[0].collections[0] if fig.axes[0].collections else fig.axes[0]._children[0]
    expected_max = float(np.sqrt(u**2 + v**2).max())
    assert cs.get_clim()[1] == expected_max


def test_magnitude_includes_w_for_stereo():
    x, y, u, v, valid = _planar_field()
    w = np.full_like(u, 3.0)  # constant w so the 3-component sum is easy to check by hand
    fig_planar = make_preview_figure("planar", x, y, u, v, valid, title="t")
    fig_stereo = make_preview_figure("stereo", x, y, u, v, valid, title="t", w=w)
    cs_planar = fig_planar.axes[0].collections[0]
    cs_stereo = fig_stereo.axes[0].collections[0]
    # adding a nonzero w must increase the magnitude's max -- confirms w
    # is actually folded into sqrt(u**2+v**2+w**2), not silently ignored.
    assert cs_stereo.get_clim()[1] > cs_planar.get_clim()[1]
    expected_max = float(np.sqrt(u**2 + v**2 + w**2).max())
    assert cs_stereo.get_clim()[1] == expected_max


def test_colormap_is_hard_coded_to_turbo():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t")
    cs = fig.axes[0].collections[0] if fig.axes[0].collections else fig.axes[0]._children[0]
    assert cs.get_cmap().name == "turbo" == MAGNITUDE_CMAP


def test_colorbar_label_includes_units():
    x, y, u, v, valid = _planar_field()
    fig_ms = make_preview_figure("planar", x, y, u, v, valid, title="t", units="m/s")
    fig_px = make_preview_figure("planar", x, y, u, v, valid, title="t", units="px/frame")
    # the colorbar axes is whichever one isn't the data axes (index 0)
    assert "m/s" in fig_ms.axes[1].get_ylabel()
    assert "px/frame" in fig_px.axes[1].get_ylabel()


def test_vectors_toggle_adds_a_quiver_without_extra_axes():
    x, y, u, v, valid = _planar_field()
    fig_off = make_preview_figure("planar", x, y, u, v, valid, title="t", show_vectors=False)
    fig_on = make_preview_figure("planar", x, y, u, v, valid, title="t", show_vectors=True)
    assert not any(isinstance(c, Quiver) for c in fig_off.axes[0].get_children())
    assert any(isinstance(c, Quiver) for c in fig_on.axes[0].get_children())
    # the contour (and its colorbar) is always there regardless -- there's
    # no "contours off" mode any more, magnitude IS the plot.
    assert len(fig_off.axes) == len(fig_on.axes) == 2


def test_auto_range_scales_to_data_min_max():
    data = np.ma.masked_array(np.arange(4, dtype=float), mask=False)
    assert _auto_range(data) == (0.0, 3.0)


def test_auto_range_falls_back_when_everything_is_masked():
    data = np.ma.masked_array(np.zeros(4), mask=True)
    vmin, vmax = _auto_range(data)
    assert vmin == 0.0
    assert vmax > vmin  # contourf rejects a zero-width level range


def test_empty_valid_mask_does_not_raise():
    x, y, u, v, valid = _planar_field()
    valid[:] = False
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t", show_vectors=True)
    assert len(fig.axes) == 2
