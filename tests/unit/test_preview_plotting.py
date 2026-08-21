"""Regression tests for plotting/preview.py's configurable preview figure
-- per-component contours, optional vector overlay, and manual color-range
overrides. No Qt/GUI needed, matplotlib alone (Agg backend, set in the
module under test)."""

import numpy as np

from piv_suite.plotting.preview import _resolve_range, make_preview_figure


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
    # orientation) -- _plot_component used to call ax.invert_yaxis() on
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

    fig = make_preview_figure("planar", x, y, u, v, valid, title="t",
                               show_contour=True, show_vectors=False)
    # Locate the V axes by its title rather than assuming a fixed index
    # (colorbar axes are interleaved among the data axes).
    v_axes = next(ax for ax in fig.axes if ax.get_title() == "V")
    ylim = v_axes.get_ylim()
    # NOT inverted: get_ylim()[0] (bottom of the displayed axes) must be
    # the SMALLER data value, matching matplotlib's default orientation
    # (larger y plotted higher) -- an inverted axes would report ylim
    # with the larger value first.
    assert ylim[0] < ylim[1], (
        f"y-axis is inverted (ylim={ylim}) -- the marker at the largest "
        "y (the image's physical top, per engines.cpu_engine's "
        "convention) would render at the BOTTOM of the plot"
    )


def test_planar_defaults_produce_two_axes_with_contours_no_vectors():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t")
    axes = fig.axes
    # 2 data axes (U, V) + 1 colorbar axes each = 4
    assert len(axes) == 4
    # a contourf collection exists (indirect: the axes have QuadContourSet
    # collections) but simplest reliable signal is that colorbars were
    # created, i.e. figure has more axes than just the 2 data ones.
    assert len(fig.axes) > 2


def test_stereo_adds_a_third_w_axes():
    x, y, u, v, valid = _planar_field()
    w = u + v
    fig_planar = make_preview_figure("planar", x, y, u, v, valid, title="t")
    fig_stereo = make_preview_figure("stereo", x, y, u, v, valid, title="t", w=w)
    assert len(fig_stereo.axes) > len(fig_planar.axes)


def test_vectors_only_no_contour_has_no_colorbar_axes():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t",
                               show_contour=False, show_vectors=True)
    # exactly 2 axes (U, V) -- no colorbar axes since contours are off
    assert len(fig.axes) == 2


def test_manual_range_overrides_auto_scale():
    # data's own min/max is 0..3 (see _planar_field) -- a manual range
    # must win over that, not just clamp/ignore it.
    data = np.ma.masked_array(np.arange(4, dtype=float), mask=False)
    assert _resolve_range(data, (-10, 10)) == (-10, 10)
    assert _resolve_range(data, None) == (0.0, 3.0)
    assert _resolve_range(data, (-10, None)) == (-10, 3.0)
    assert _resolve_range(data, (None, 10)) == (0.0, 10)


def test_manual_range_is_used_by_the_built_figure():
    x, y, u, v, valid = _planar_field()
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t",
                               ranges={"U": (-10, 10)})
    cs = fig.axes[0].collections[0] if fig.axes[0].collections else fig.axes[0]._children[0]
    assert cs.get_clim() == (-10, 10)


def test_empty_valid_mask_does_not_raise():
    x, y, u, v, valid = _planar_field()
    valid[:] = False
    fig = make_preview_figure("planar", x, y, u, v, valid, title="t",
                               show_contour=True, show_vectors=True)
    assert len(fig.axes) >= 2
