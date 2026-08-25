"""Tests for preview_panel.py's range-preview averaging (Task 4): running
several consecutive pairs through the existing single-pair compute path
and averaging them into ONE field. _compute_range/_average_results are
exercised directly (no QThread/main_window needed) -- they're plain
synchronous methods/functions once the per-pair compute calls themselves
are stubbed out, matching this repo's existing pattern of monkeypatching
_compute_planar/_compute_stereo for fast, non-real-data GUI tests (see
test_gui_smoke.py's test_preview_shows_progress_bar_while_running_and_hides_after).
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from piv_suite_gui.widgets.preview_panel import PreviewPanel, _average_results


def _dict_planar(index, valid_mask, u_val, v_val):
    valid = np.array(valid_mask, dtype=bool)
    u = np.full(valid.shape, u_val, dtype=float)
    v = np.full(valid.shape, v_val, dtype=float)
    return dict(kind="planar", pair_id=f"{index:04d}", x=np.zeros(valid.shape), y=np.zeros(valid.shape),
                u=u, v=v, valid=valid, elapsed=0.1, n_valid=int(valid.sum()), n_total=int(valid.size),
                n_range=1, n_std=2, units="m/s")


def test_average_results_nanmean_respects_each_pairs_own_valid_mask():
    # pair 0's [0,0] cell is invalid (must not pollute the average with
    # garbage), pair 1's is valid -- the averaged cell must come out as
    # pair 1's own value alone, not a mix including pair 0's un-trusted data.
    r0 = _dict_planar(0, [[False, True], [True, True]], u_val=10.0, v_val=0.0)
    r1 = _dict_planar(1, [[True, True], [True, True]], u_val=2.0, v_val=0.0)
    avg = _average_results([r0, r1])

    assert avg["u"][0, 0] == pytest.approx(2.0)  # only pair 1 contributed here
    assert avg["u"][1, 1] == pytest.approx(6.0)  # both pairs contributed: mean(10, 2)
    assert avg["valid"][0, 0] == True   # noqa: E712 -- either pair being valid is enough
    assert avg["v"].shape == (2, 2)


def test_average_results_sums_counters_and_elapsed():
    r0 = _dict_planar(0, [[True]], u_val=1.0, v_val=1.0)
    r1 = _dict_planar(1, [[True]], u_val=3.0, v_val=3.0)
    avg = _average_results([r0, r1])

    assert avg["elapsed"] == pytest.approx(0.2)
    assert avg["n_range"] == 2  # 1 + 1
    assert avg["n_std"] == 4    # 2 + 2
    assert avg["units"] == "m/s"
    assert avg["pair_id"] == "0000..0001 (avg of 2)"


def test_average_results_includes_w_only_for_stereo():
    r0 = _dict_planar(0, [[True]], u_val=1.0, v_val=1.0)
    r0["kind"] = "stereo"
    r0["w"] = np.array([[5.0]])
    r1 = _dict_planar(1, [[True]], u_val=3.0, v_val=3.0)
    r1["kind"] = "stereo"
    r1["w"] = np.array([[7.0]])
    avg = _average_results([r0, r1])

    assert avg["kind"] == "stereo"
    assert avg["w"] == pytest.approx(np.array([[6.0]]))

    # a planar pair (no w key at all) must come back with w=None, not KeyError
    p0 = _dict_planar(0, [[True]], u_val=1.0, v_val=1.0)
    p1 = _dict_planar(1, [[True]], u_val=3.0, v_val=3.0)
    avg_planar = _average_results([p0, p1])
    assert avg_planar["w"] is None


def test_compute_range_count_one_returns_the_single_result_unchanged(qtbot):
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    seen = []

    def fake_compute_planar(project, preprocess, correlation, validation, post, calibration, index):
        seen.append(index)
        return _dict_planar(index, [[True]], u_val=1.0, v_val=1.0)

    panel._compute_planar = fake_compute_planar
    project = type("P", (), {"mode": "planar", "dual_camera": False})()
    progress_calls = []

    result = panel._compute_range(project, None, None, None, None, None, None, None,
                                   start_index=5, count=1, progress_cb=lambda d, t: progress_calls.append((d, t)))

    assert seen == [5]  # only the selected pair, not a range
    assert result["pair_id"] == "0005"  # untouched single-pair result, no "avg of" wrapping
    assert progress_calls == [(1, 1)]


def test_compute_range_count_greater_than_one_runs_each_pair_and_averages(qtbot):
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    seen = []

    def fake_compute_planar(project, preprocess, correlation, validation, post, calibration, index):
        seen.append(index)
        return _dict_planar(index, [[True]], u_val=float(index), v_val=0.0)

    panel._compute_planar = fake_compute_planar
    project = type("P", (), {"mode": "planar", "dual_camera": False})()
    progress_calls = []

    result = panel._compute_range(project, None, None, None, None, None, None, None,
                                   start_index=2, count=3, progress_cb=lambda d, t: progress_calls.append((d, t)))

    assert seen == [2, 3, 4]  # 3 CONSECUTIVE pairs starting at start_index
    assert result["u"][0, 0] == pytest.approx((2.0 + 3.0 + 4.0) / 3)  # nanmean across the range
    assert "avg of 3" in result["pair_id"]
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


def test_compute_range_routes_by_project_mode_and_dual_camera(qtbot):
    panel = PreviewPanel()
    qtbot.addWidget(panel)
    called = {"planar": 0, "dual_planar": 0, "stereo": 0}
    panel._compute_planar = lambda *a: (called.__setitem__("planar", called["planar"] + 1),
                                         _dict_planar(0, [[True]], 1.0, 1.0))[1]
    panel._compute_dual_planar = lambda *a: (called.__setitem__("dual_planar", called["dual_planar"] + 1),
                                              _dict_planar(0, [[True]], 1.0, 1.0))[1]
    panel._compute_stereo = lambda *a: (called.__setitem__("stereo", called["stereo"] + 1),
                                         _dict_planar(0, [[True]], 1.0, 1.0))[1]

    stereo_project = type("P", (), {"mode": "stereo", "dual_camera": False})()
    panel._compute_range(stereo_project, None, None, None, None, None, None, None, 0, 1, lambda d, t: None)
    assert called == {"planar": 0, "dual_planar": 0, "stereo": 1}

    dual_project = type("P", (), {"mode": "planar", "dual_camera": True})()
    panel._compute_range(dual_project, None, None, None, None, None, None, None, 0, 1, lambda d, t: None)
    assert called == {"planar": 0, "dual_planar": 1, "stereo": 1}

    planar_project = type("P", (), {"mode": "planar", "dual_camera": False})()
    panel._compute_range(planar_project, None, None, None, None, None, None, None, 0, 1, lambda d, t: None)
    assert called == {"planar": 1, "dual_planar": 1, "stereo": 1}
