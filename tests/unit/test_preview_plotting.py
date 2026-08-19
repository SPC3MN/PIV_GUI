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
