"""Regression tests for plotting/preview.py's single-magnitude-field
preview figure -- auto-scaled colorbar (labeled with real units), optional
vector overlay, hard-coded colormap. No Qt/GUI needed, matplotlib alone
(Agg backend, set in the module under test)."""

import numpy as np
import pytest
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


# ---- geometric fidelity ----
#
# A PIV field is a picture of a physical region, so the preview has to
# preserve shape: a round vortex must render round. Before these tests the
# figure used a fixed 7x6 canvas with matplotlib's default aspect="auto",
# which stretches the data to fill the axes -- measured at 1.881x on a real
# stereo grid (379x704), i.e. a circle rendered 88% taller than wide.

def _axes_scale_ratio(fig):
    """(inches-per-data-unit in y) / (same in x) for the data axes. 1.0 means
    a circle in the data renders as a circle on paper."""
    ax = fig.axes[0]
    box = ax.get_position()
    fw, fh = fig.get_size_inches()
    sx = (box.width * fw) / abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    sy = (box.height * fh) / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    return sy / sx


def _wide_field(ny=379, nx=704):
    """A real stereo vector grid's shape -- deliberately far from square, which
    is what makes the distortion visible."""
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    u = np.ones_like(x)
    v = np.zeros_like(x)
    return x, y, u, v, np.ones_like(x, dtype=bool)


def test_preview_preserves_geometry_on_a_non_square_grid():
    x, y, u, v, valid = _wide_field()
    fig = make_preview_figure("stereo", x, y, u, v, valid, "t")
    assert fig.axes[0].get_aspect() == 1.0
    assert _axes_scale_ratio(fig) == pytest.approx(1.0, abs=1e-6)


def test_preview_preserves_geometry_on_a_tall_grid():
    """The opposite extreme must not be distorted either."""
    x, y, u, v, valid = _wide_field(ny=700, nx=200)
    fig = make_preview_figure("stereo", x, y, u, v, valid, "t")
    assert _axes_scale_ratio(fig) == pytest.approx(1.0, abs=1e-6)


def test_a_circular_feature_stays_circular():
    """The property that actually matters, stated in the terms a user would:
    measure the rendered width and height of a round blob's contour."""
    ny, nx = 379, 704
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.sqrt((x - nx / 2) ** 2 + (y - ny / 2) ** 2)
    u = np.exp(-(r / 120.0) ** 2)
    fig = make_preview_figure("stereo", x, y, u, np.zeros_like(u),
                              np.ones_like(u, dtype=bool), "t")
    ax = fig.axes[0]
    # Half-max contour of a radially symmetric blob: equal extent both ways.
    cs = ax.contour(x, y, u, levels=[0.5])
    pts = np.vstack([p.vertices for p in cs.get_paths()])
    width = np.ptp(pts[:, 0]) * (ax.get_position().width * fig.get_size_inches()[0]
                                 / abs(ax.get_xlim()[1] - ax.get_xlim()[0]))
    height = np.ptp(pts[:, 1]) * (ax.get_position().height * fig.get_size_inches()[1]
                                  / abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
    assert width == pytest.approx(height, rel=0.02)


def test_figure_size_follows_the_requested_canvas_shape():
    """The preview fills a resizable widget, so the figure must take its shape
    from the caller instead of a hard-coded 7x6."""
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, "t", figsize=(11.0, 4.0))
    assert tuple(fig.get_size_inches()) == pytest.approx((11.0, 4.0))
