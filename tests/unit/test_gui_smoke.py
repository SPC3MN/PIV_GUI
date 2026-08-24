"""GUI smoke tests -- panel construction and signal/slot wiring don't
raise. Full interaction testing is lower priority for this single-user
tool; core pipeline correctness is covered by the other unit tests plus
the manual end-to-end preview/run checks in the project notes."""

import pytest

pytest.importorskip("PySide6")

from piv_suite_gui.main_window import MainWindow


def test_main_window_constructs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.project_panel is not None
    assert window.settings_panel is not None
    assert window.preview_panel is not None
    assert window.run_panel is not None


def test_preview_shows_progress_bar_while_running_and_hides_after(qtbot):
    # A slow preview (real PIV correlation, up to a minute on a full-
    # resolution frame) used to run inline on the GUI thread, giving no
    # feedback beyond a static "Running preview..." label and leaving the
    # window unresponsive to Windows' own paint/ping messages for the
    # whole call -- reported from real use as the window showing "(Not
    # Responding)". The compute now runs on a QThread (see
    # preview_panel._PreviewWorker); this stubs _compute_planar with an
    # artificial delay to check the bar/button reflect "running" while
    # the worker thread is actually still going, and "idle" only once it
    # has genuinely finished -- not just that _do_preview() returned
    # (which now happens immediately, before the worker completes).
    import time
    import numpy as np
    from piv_suite_gui.widgets.preview_panel import PreviewPanel

    panel = PreviewPanel()
    qtbot.addWidget(panel)
    panel.show()
    assert panel.progress_bar.isVisible() is False

    def fake_compute_planar(project, preprocess, correlation, validation, post, calibration, index):
        time.sleep(0.2)  # long enough for the test to observe "running" state
        grid = np.zeros((3, 3))
        valid = np.ones((3, 3), dtype=bool)
        return dict(kind="planar", pair_id="0000", x=grid, y=grid, u=grid, v=grid,
                    valid=valid, elapsed=0.0, n_valid=9, n_total=9, n_range=0, n_std=0)

    panel._compute_planar = fake_compute_planar
    panel.window = lambda: type("W", (), {
        "project_panel": type("P", (), {
            "get_project_settings": lambda self: type("S", (), {"mode": "planar"})(),
            "get_preprocess_settings": lambda self: None,
        })(),
        "settings_panel": type("SP", (), {
            "get_correlation_settings": lambda self: None,
            "get_validation_settings": lambda self: None,
            "get_postprocess_settings": lambda self: None,
            "get_calibration_settings": lambda self: None,
        })(),
    })()
    panel.pair_combo.addItem("0000")  # skip _refresh_pairs -- the fake window has no real project I/O

    panel._do_preview()

    # _do_preview() itself returns right after starting the worker thread
    # -- the bar/button state it set synchronously beforehand must
    # already show "running" here, DURING the fake compute's sleep.
    assert panel.progress_bar.isVisible() is True
    assert panel.preview_btn.isEnabled() is False

    qtbot.waitUntil(lambda: not panel.progress_bar.isVisible(), timeout=2000)
    assert panel.preview_btn.isEnabled() is True


def test_header_badge_reflects_backend_availability(qtbot):
    from piv_suite.engines.registry import is_gpu_available

    window = MainWindow()
    qtbot.addWidget(window)
    text = window.header.backend_badge.text()
    if is_gpu_available():
        assert "GPU" in text
    else:
        assert "CPU only" in text


def test_header_run_button_mirrors_run_panel_state(qtbot):
    # The header's Run button is a second surface for run_panel's action,
    # so it must never be clickable when run_panel's own button isn't --
    # including while a batch is already in flight, which is why
    # run_panel emits on every enabled-state change rather than the
    # header just listening to `previewed` directly.
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.header.run_btn.isEnabled() is False

    window.preview_panel.previewed.emit(True)
    assert window.run_panel.run_btn.isEnabled() is True
    assert window.header.run_btn.isEnabled() is True

    window.run_panel.set_run_enabled(False)  # what _start_run does
    assert window.header.run_btn.isEnabled() is False


def test_header_run_forwards_to_run_panel_and_shows_run_tab(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    # swap the real _start_run connection for a recorder -- this test is
    # about the forwarding wiring, not about launching a batch thread
    window.run_panel.run_btn.clicked.disconnect()
    started = []
    window.run_panel.run_btn.clicked.connect(lambda: started.append(True))

    # while disabled, forwarding must not start anything, but should still
    # bring the Run tab forward so the user sees the disabled button
    window.header.run_requested.emit()
    assert window.tabs.currentWidget() is window.run_panel
    assert started == []

    window.tabs.setCurrentWidget(window.preview_panel)
    window.preview_panel.previewed.emit(True)
    window.header.run_requested.emit()
    assert window.tabs.currentWidget() is window.run_panel
    assert started == [True]


def test_status_bar_shows_backend_and_version(qtbot):
    from piv_suite import __version__

    window = MainWindow()
    qtbot.addWidget(window)
    assert __version__ in window.header.version_text()
    assert window.status_backend_label.text()  # non-empty backend summary


def test_preview_panel_pair_selector_and_plot_options_exist(qtbot):
    from piv_suite_gui.widgets.preview_panel import PreviewPanel

    panel = PreviewPanel()
    qtbot.addWidget(panel)

    assert panel.pair_combo.count() == 0
    assert panel.show_contour_check.isChecked() is True   # contours on by default
    assert panel.show_vectors_check.isChecked() is False  # vectors off by default
    assert panel.cmap_combo.count() > 1

    # U/V/W range rows, each auto-scaled (spins disabled) by default
    for name in ("U", "V", "W"):
        _label, auto_check, min_spin, max_spin = panel._range_rows[name]
        assert auto_check.isChecked() is True
        assert min_spin.isEnabled() is False
        assert max_spin.isEnabled() is False


def test_preview_panel_range_row_toggles_manual_spins(qtbot):
    from piv_suite_gui.widgets.preview_panel import PreviewPanel

    panel = PreviewPanel()
    qtbot.addWidget(panel)

    _label, auto_check, min_spin, max_spin = panel._range_rows["U"]
    auto_check.setChecked(False)
    assert min_spin.isEnabled() is True
    assert max_spin.isEnabled() is True

    min_spin.setValue(-2.5)
    max_spin.setValue(2.5)
    ranges = panel._get_ranges()
    assert ranges["U"] == (-2.5, 2.5)
    assert "V" not in ranges  # still auto


def test_preview_panel_w_range_row_hidden_until_stereo(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    pp = window.preview_panel
    label, auto_check, min_spin, max_spin = pp._range_rows["W"]
    assert not label.isVisible()

    window.project_panel.stereo_radio.setChecked(True)
    assert label.isVisible()


def test_loose_suffix_extension_comes_from_glob(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    pp = window.project_panel
    pp.mode_loose.setChecked(True)

    # defaults: suffix fields show just the marker, no extension
    assert pp.suffix_a_edit.text() == "_a"
    assert pp.suffix_b_edit.text() == "_b"
    assert pp.suffix_cam0_edit.text() == "_cam1"
    assert pp.suffix_cam1_edit.text() == "_cam2"

    # combined with the default glob's extension at read time
    settings = pp.get_project_settings()
    assert settings.suffix_a == "_a.im7"
    assert settings.suffix_b == "_b.im7"
    assert settings.suffix_cam0 == "_cam1.im7"
    assert settings.suffix_cam1 == "_cam2.im7"

    # changing Glob's extension re-derives every suffix without touching
    # the marker fields themselves
    pp.loose_glob_edit.setText("*.tif")
    assert pp.suffix_a_edit.text() == "_a"  # unchanged
    settings = pp.get_project_settings()
    assert settings.suffix_a == "_a.tif"
    assert settings.suffix_b == "_b.tif"
    assert settings.suffix_cam0 == "_cam1.tif"
    assert settings.suffix_cam1 == "_cam2.tif"


def test_loose_suffix_typed_extension_is_overridden_by_glob(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    pp = window.project_panel
    pp.mode_loose.setChecked(True)
    pp.loose_glob_edit.setText("*.tif")
    pp.suffix_a_edit.setText("_a.im7")  # stale/wrong extension typed by hand

    settings = pp.get_project_settings()
    assert settings.suffix_a == "_a.tif"  # glob's extension wins, not duplicated


def test_loose_suffix_set_from_strips_extension_back_off(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    pp = window.project_panel
    from piv_suite.config.schema import ProjectSettings

    project = ProjectSettings(
        input_mode="loose", input_path="x", output_dir="y", backend="cpu", mode="planar",
        loose_glob="*.tif", suffix_a="_a.tif", suffix_b="_b.tif",
        suffix_cam0="_cam1.tif", suffix_cam1="_cam2.tif",
    )
    pp.set_from(project)
    assert pp.suffix_a_edit.text() == "_a"
    assert pp.suffix_b_edit.text() == "_b"
    assert pp.suffix_cam0_edit.text() == "_cam1"
    assert pp.suffix_cam1_edit.text() == "_cam2"
    # round-trips back to the same full suffix
    assert pp.get_project_settings().suffix_a == "_a.tif"


def test_settings_panel_greys_out_inapplicable_backend_fields(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    pp = window.project_panel

    gpu_only = [sp.batch_size_check, sp.tiling_check, sp.n_tiles_y_spin,
                sp.n_tiles_x_spin, sp.tile_margin_check]
    cpu_only = [sp.correlation_method_combo, sp.deformation_method_combo,
                sp.interpolation_order_spin, sp.filter_method_combo]

    # default backend is CPU -- GPU-only fields start disabled
    assert pp.backend == "cpu"
    for w in gpu_only:
        assert w.isEnabled() is False
    for w in cpu_only:
        assert w.isEnabled() is True

    # a checkbox-gated GPU spin stays disabled even if its own checkbox
    # was checked while GPU was selected, once backend flips back to CPU
    pp.gpu_radio.setChecked(True)
    sp.batch_size_check.setChecked(True)
    assert sp.batch_size_spin.isEnabled() is True
    pp.cpu_radio.setChecked(True)
    assert sp.batch_size_spin.isEnabled() is False
    for w in gpu_only:
        assert w.isEnabled() is False
    for w in cpu_only:
        assert w.isEnabled() is True

    pp.gpu_radio.setChecked(True)
    for w in gpu_only:
        assert w.isEnabled() is True
    for w in cpu_only:
        assert w.isEnabled() is False
    assert sp.batch_size_spin.isEnabled() is True  # checkbox was still checked


def test_project_panel_default_settings(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    settings = window.project_panel.get_project_settings()
    assert settings.backend == "cpu"
    assert settings.mode == "planar"
    assert settings.input_mode == "set"


def test_settings_panel_default_passes_match_canonical_schema(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    from piv_suite.config.schema import CorrelationSettings
    passes = window.settings_panel.get_correlation_settings().passes
    default_passes = CorrelationSettings().passes
    assert [(p.window_size, p.overlap_fraction) for p in passes] == \
           [(p.window_size, p.overlap_fraction) for p in default_passes]


def test_run_button_disabled_until_preview(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert not window.run_panel.run_btn.isEnabled()
    window.preview_panel.previewed.emit(True)
    assert window.run_panel.run_btn.isEnabled()
    window.preview_panel.previewed.emit(False)
    assert not window.run_panel.run_btn.isEnabled()


def test_calibration_panel_hidden_until_stereo_selected(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert not window.calibration_panel.isVisible()
    window.project_panel.stereo_radio.setChecked(True)
    assert window.calibration_panel.isVisible()


def test_calibration_panel_default_settings(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    settings = window.calibration_panel.get_settings()
    assert settings.world_scale_px_per_mm == 1.0
    assert set(settings.cam0_mapping.dx_coefs.keys()) == {
        "1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s"}
    # DaVis auto-load fields all off by default -- unchanged single-plane
    # manual-entry behavior for anyone who never loads from a .set
    assert settings.cam0_mapping.z_mm is None
    assert settings.cam0_mapping_plane2 is None
    assert settings.cam1_mapping_plane2 is None
    assert settings.sheet_z_mm is None


def test_main_window_is_resizable(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    # the window itself must NOT be width-locked...
    assert window.maximumWidth() > 10000  # Qt's "no maximum" sentinel is a huge int
    # ...only the left panel has a fixed, comfortable width with no
    # horizontal scroll bar, so growing the window benefits the right side
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea
    left_scroll = window.findChild(QScrollArea)
    assert left_scroll is not None
    assert left_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert left_scroll.minimumWidth() == left_scroll.maximumWidth()  # fixed width, just not the whole window


def test_loose_options_hidden_in_set_mode(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    pp = window.project_panel
    assert pp.mode_set.isChecked()
    assert not pp.loose_options.isVisible()
    pp.mode_loose.setChecked(True)
    assert pp.loose_options.isVisible()


def test_loose_options_show_correct_suffix_fields_per_mode(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    pp = window.project_panel
    pp.mode_loose.setChecked(True)

    pp.planar_radio.setChecked(True)
    assert pp.suffix_a_edit.isVisible()
    assert not pp.suffix_cam0_edit.isVisible()

    pp.stereo_radio.setChecked(True)
    assert not pp.suffix_a_edit.isVisible()
    assert pp.suffix_cam0_edit.isVisible()


def test_mode_and_backend_are_separate_group_boxes(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    pp = window.project_panel
    # planar_radio/stereo_radio and cpu_radio/gpu_radio must live in
    # different parent group boxes, not one combined box
    mode_parent = pp.planar_radio.parentWidget()
    backend_parent = pp.cpu_radio.parentWidget()
    assert mode_parent is not backend_parent


def test_validation_group_is_user_editable(qtbot):
    # ValidationSettings' remaining 5 fields are the internal per-pass
    # NaN-safety-fill mechanism only (see schema.py's ValidationSettings
    # docstring) -- sig2noise_*/validation_first_pass/replace_vectors were
    # removed when real vector validation moved entirely to
    # PostProcessSettings. Confirm the controls that ARE still user-facing
    # exist AND that get_validation_settings() actually reflects widget
    # state, not a hardcoded ValidationSettings().
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    assert hasattr(sp, "filter_method_combo")
    assert hasattr(sp, "smoothn_check")
    assert not hasattr(sp, "s2n_threshold_spin")
    assert not hasattr(sp, "s2n_validate_check")

    sp.filter_kernel_size_spin.setValue(5)
    sp.smoothn_check.setChecked(True)
    sp.smoothn_p_spin.setValue(0.2)
    settings = sp.get_validation_settings()
    assert settings.filter_kernel_size == 5
    assert settings.smoothn is True
    assert settings.smoothn_p == 0.2


def test_range_filter_has_only_residual_and_window_size(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    assert not hasattr(sp, "mag_min_spin")
    assert not hasattr(sp, "mag_max_spin")
    assert not hasattr(sp, "mag_enabled_check")
    assert hasattr(sp, "residual_max_spin")
    assert hasattr(sp, "window_size_spin")

    sp.residual_enabled_check.setChecked(True)
    sp.residual_max_spin.setValue(3.5)
    sp.window_size_spin.setValue(5)
    post = sp.get_postprocess_settings()
    assert post.range_filter.enabled is True
    assert post.range_filter.residual_max == 3.5
    assert post.range_filter.window_size == 5


def test_std_dev_filter_still_present(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    sp.std_filter_check.setChecked(True)
    sp.n_std_spin.setValue(3.0)
    post = sp.get_postprocess_settings()
    assert post.global_outlier_std == 3.0


def test_double_spinboxes_use_two_decimals_except_deliberate_high_precision_fields(qtbot):
    # style_spin()'s uniform 2-decimal default is right for most fields,
    # but was wrong for a few real-valued PIV inputs that legitimately
    # need finer precision (dt, calibration values, sig2noise threshold,
    # smoothn_p) -- a 2-decimal cap made sub-0.01 values impossible to
    # enter at all, not just imprecise. Those fields deliberately override
    # decimals; everything else still gets the uniform 2-decimal default.
    window = MainWindow()
    qtbot.addWidget(window)
    from PySide6.QtWidgets import QDoubleSpinBox
    high_precision = {
        window.settings_panel.dt_spin,
        window.settings_panel.pixel_pitch_spin,
        window.settings_panel.frame_dt_spin,
        window.settings_panel.smoothn_p_spin,
    }
    for panel in (window.project_panel, window.settings_panel, window.calibration_panel):
        for spin in panel.findChildren(QDoubleSpinBox):
            if spin in high_precision:
                assert spin.decimals() > 2, f"{spin} expected to override the 2-decimal default"
            else:
                assert spin.decimals() == 2, f"{spin} has {spin.decimals()} decimals, expected 2"


def test_calibration_settings_default_unset_but_settable(qtbot):
    # CalibrationSettings (pixel_pitch_mm, frame_dt_s) previously had no
    # GUI control anywhere, silently locking every GUI result to px/frame.
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    default = sp.get_calibration_settings()
    assert default.pixel_pitch_mm is None
    assert default.frame_dt_s is None

    sp.pixel_pitch_check.setChecked(True)
    sp.pixel_pitch_spin.setValue(0.012345)
    sp.frame_dt_check.setChecked(True)
    sp.frame_dt_spin.setValue(0.002)
    settings = sp.get_calibration_settings()
    assert settings.pixel_pitch_mm == pytest.approx(0.012345)
    assert settings.frame_dt_s == pytest.approx(0.002)


def test_multiset_index_is_editable(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    pp = window.project_panel
    assert pp.get_project_settings().multiset_index == 0
    pp.multiset_index_spin.setValue(3)
    assert pp.get_project_settings().multiset_index == 3


def test_correlation_settings_exposes_all_calculation_fields(qtbot):
    # correlation_method/deformation_method/interpolation_order (CPU) and
    # batch_size/tile_margin_px (GPU) previously had no GUI control and
    # were silently stuck at dataclass defaults.
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel

    sp.correlation_method_combo.setCurrentText("linear")
    sp.deformation_method_combo.setCurrentText("second image")
    sp.interpolation_order_spin.setValue(1)
    sp.batch_size_check.setChecked(True)
    sp.batch_size_spin.setValue(128)
    sp.tile_margin_check.setChecked(True)
    sp.tile_margin_spin.setValue(200)

    settings = sp.get_correlation_settings()
    assert settings.correlation_method == "linear"
    assert settings.deformation_method == "second image"
    assert settings.interpolation_order == 1
    assert settings.batch_size == 128
    assert settings.tile_margin_px == 200

    sp.batch_size_check.setChecked(False)
    sp.tile_margin_check.setChecked(False)
    settings = sp.get_correlation_settings()
    assert settings.batch_size is None
    assert settings.tile_margin_px is None


def test_run_panel_populates_calibration_in_config(qtbot):
    # run_panel._start_run must attach the settings panel's calibration
    # settings to the assembled ProjectConfig -- confirmed missing before
    # (ProjectConfig.calibration stayed at its default, unset, forever).
    window = MainWindow()
    qtbot.addWidget(window)
    window.settings_panel.pixel_pitch_check.setChecked(True)
    window.settings_panel.pixel_pitch_spin.setValue(0.05)

    project = window.project_panel.get_project_settings()
    correlation = window.settings_panel.get_correlation_settings()
    validation = window.settings_panel.get_validation_settings()
    post = window.settings_panel.get_postprocess_settings()
    calibration = window.settings_panel.get_calibration_settings()

    from piv_suite.config.schema import ProjectConfig
    config = ProjectConfig(project=project, correlation=correlation,
                            validation=validation, postprocess=post,
                            calibration=calibration)
    assert config.calibration.pixel_pitch_mm == 0.05


def test_passes_table_has_no_internal_scrollbar(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    from PySide6.QtCore import Qt
    table = window.settings_panel.passes_table.table
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    # the table must be tall enough to show every row without scrolling
    assert table.height() >= table.verticalHeader().length() + table.horizontalHeader().height()


def test_calibration_coef_table_has_no_internal_scrollbar(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    from PySide6.QtCore import Qt
    table = window.calibration_panel.cam0_form.coef_table
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert table.rowCount() == 10


def test_no_sign_flip_option(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert not hasattr(window.settings_panel, "sign_flip_check")
    post = window.settings_panel.get_postprocess_settings()
    assert not hasattr(post, "apply_v_sign_flip")


def test_spin_boxes_have_no_counter_buttons(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox
    # project_panel has no spin boxes (only text/combo/radio fields) --
    # settings_panel and calibration_panel are where every numeric field lives
    for panel in (window.settings_panel, window.calibration_panel):
        spins = panel.findChildren(QSpinBox) + panel.findChildren(QDoubleSpinBox)
        assert spins, f"{panel} has no spin boxes to check"
        for spin in spins:
            assert spin.buttonSymbols() == QAbstractSpinBox.NoButtons


def test_input_path_changed_emits_on_editing_finished(qtbot):
    from piv_suite_gui.widgets.project_panel import ProjectPanel

    panel = ProjectPanel()
    qtbot.addWidget(panel)
    received = []
    panel.input_path_changed.connect(received.append)
    panel.input_path_edit.setText("some/path.set")
    panel.input_path_edit.editingFinished.emit()
    assert received == ["some/path.set"]


def test_input_path_changed_emits_on_browse(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from piv_suite_gui.widgets.project_panel import ProjectPanel

    panel = ProjectPanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("chosen.set", ""))
    received = []
    panel.input_path_changed.connect(received.append)
    panel._browse_input()
    assert panel.input_path_edit.text() == "chosen.set"
    assert received == ["chosen.set"]


def test_main_window_extracts_calibration_when_set_path_selected(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw
    from piv_suite.config.schema import CalibrationSettings

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    monkeypatch.setattr(mw, "read_calibration_from_set",
                         lambda path, idx: CalibrationSettings(pixel_pitch_mm=0.05, frame_dt_s=0.0007))

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.project_panel.mode_set.isChecked()

    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    sp = window.settings_panel
    assert sp.pixel_pitch_check.isChecked()
    assert sp.pixel_pitch_spin.value() == pytest.approx(0.05)
    assert sp.frame_dt_check.isChecked()
    assert sp.frame_dt_spin.value() == pytest.approx(0.0007)


def test_calibration_extraction_overwrites_prior_manual_edit(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw
    from piv_suite.config.schema import CalibrationSettings

    set_path = tmp_path / "recording.set"
    set_path.write_text("")

    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    sp.pixel_pitch_check.setChecked(True)
    sp.pixel_pitch_spin.setValue(1.0)  # stale manual value

    monkeypatch.setattr(mw, "read_calibration_from_set",
                         lambda path, idx: CalibrationSettings(pixel_pitch_mm=0.0514883, frame_dt_s=0.0007))
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    # pixel_pitch_spin only has 6-decimal precision (style_spin decimals=6)
    assert sp.pixel_pitch_spin.value() == pytest.approx(0.0514883, abs=1e-6)


def test_calibration_extraction_clears_field_that_could_not_be_extracted(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw
    from piv_suite.config.schema import CalibrationSettings

    set_path = tmp_path / "recording.set"
    set_path.write_text("")

    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    sp.frame_dt_check.setChecked(True)
    sp.frame_dt_spin.setValue(0.002)  # stale manual value

    monkeypatch.setattr(mw, "read_calibration_from_set",
                         lambda path, idx: CalibrationSettings(pixel_pitch_mm=0.05, frame_dt_s=None))
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    assert sp.pixel_pitch_check.isChecked()
    assert not sp.frame_dt_check.isChecked()  # couldn't extract -> authoritative, cleared


def test_calibration_extraction_failure_shows_status_and_does_not_crash(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    monkeypatch.setattr(mw, "read_calibration_from_set",
                         lambda path, idx: (_ for _ in ()).throw(RuntimeError("boom")))

    window = MainWindow()
    qtbot.addWidget(window)
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()  # must not raise

    assert window.statusBar().currentMessage()


def test_calibration_extraction_skipped_in_loose_mode(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw

    called = []
    monkeypatch.setattr(mw, "read_calibration_from_set", lambda path, idx: called.append(True))

    window = MainWindow()
    qtbot.addWidget(window)
    window.project_panel.mode_loose.setChecked(True)
    window.project_panel.input_path_edit.setText(str(tmp_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    assert called == []


def test_settings_panel_set_calibration_clears_unsupplied_fields(qtbot):
    from piv_suite.config.schema import CalibrationSettings

    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    sp.frame_dt_check.setChecked(True)
    sp.frame_dt_spin.setValue(0.002)

    sp.set_calibration_settings(CalibrationSettings(pixel_pitch_mm=0.02, frame_dt_s=None))

    assert sp.pixel_pitch_check.isChecked()
    assert sp.pixel_pitch_spin.value() == pytest.approx(0.02)
    assert not sp.frame_dt_check.isChecked()


def _fake_stereo_settings(z0=1.0, z1=-2.0):
    from piv_suite.config.schema import CameraMappingSettings, StereoSettings

    def mapping(z_mm, name):
        return CameraMappingSettings(x0=1.0, x_span=2.0, y0=3.0, y_span=4.0, name=name, z_mm=z_mm)

    return StereoSettings(
        cam0_mapping=mapping(z0, "cam0 plane1"), cam0_mapping_plane2=mapping(z1, "cam0 plane2"),
        cam1_mapping=mapping(z0, "cam1 plane1"), cam1_mapping_plane2=mapping(z1, "cam1 plane2"),
        world_shape=(500, 600), world_scale_px_per_mm=17.9,
    )


def test_camera_mapping_form_set_settings_shows_davis_autoload_summary(qtbot):
    from piv_suite_gui.widgets.calibration_panel import _CameraMappingForm

    form = _CameraMappingForm("cam0")
    qtbot.addWidget(form)
    form.show()
    assert not form.davis_plane_label.isVisible()

    stereo = _fake_stereo_settings()
    form.set_settings(stereo.cam0_mapping, stereo.cam0_mapping_plane2)
    assert form.davis_plane_label.isVisible()
    assert form.plane2 is stereo.cam0_mapping_plane2
    assert form.get_settings().z_mm == pytest.approx(1.0)

    form._clear_davis_autoload()
    assert not form.davis_plane_label.isVisible()
    assert form.plane2 is None
    assert form.get_settings().z_mm is None
    # clearing doesn't wipe the numbers -- x0 etc. stay as they were
    assert form.x0_spin.value() == pytest.approx(1.0)


def test_calibration_panel_set_settings_round_trips_two_planes_and_world_grid(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    cp = window.calibration_panel
    stereo = _fake_stereo_settings()

    cp.set_settings(stereo)
    result = cp.get_settings()
    assert result.cam0_mapping.z_mm == pytest.approx(1.0)
    assert result.cam0_mapping_plane2.z_mm == pytest.approx(-2.0)
    assert result.cam1_mapping_plane2.z_mm == pytest.approx(-2.0)
    assert result.world_shape == (500, 600)
    assert result.world_scale_px_per_mm == pytest.approx(17.9)
    assert result.sheet_z_mm is None  # not auto-filled by set_settings alone

    cp.sheet_z_mm_check.setChecked(True)
    cp.sheet_z_mm_spin.setValue(-0.5)
    assert cp.get_settings().sheet_z_mm == pytest.approx(-0.5)


def test_stereo_calibration_extraction_triggers_on_switching_to_stereo_after_path_selected(
        qtbot, monkeypatch, tmp_path):
    # Regression test: real usage picks the input path FIRST (planar is the
    # default mode) and only switches to Stereo mode afterward.
    # input_path_changed only fires on a path change, so without also
    # reacting to the mode switch, stereo calibration silently never got
    # extracted at all -- reported from real GUI use ("Load from..." looked
    # greyed out and the dewarp polynomial never auto-populated).
    import piv_suite_gui.main_window as mw

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    stereo_settings = _fake_stereo_settings()
    monkeypatch.setattr(mw, "read_stereo_calibration_from_set", lambda path, idx: stereo_settings)

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.project_panel.planar_radio.isChecked()  # default mode
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()
    assert window.calibration_panel.cam0_form.plane2 is None  # not extracted yet -- still planar

    window.project_panel.stereo_radio.setChecked(True)  # switch mode AFTER path is already set

    assert window.calibration_panel.cam0_form.plane2 is stereo_settings.cam0_mapping_plane2


def test_stereo_calibration_extraction_wiring_when_stereo_mode(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    stereo_settings = _fake_stereo_settings()
    monkeypatch.setattr(mw, "read_stereo_calibration_from_set", lambda path, idx: stereo_settings)

    window = MainWindow()
    qtbot.addWidget(window)
    window.project_panel.stereo_radio.setChecked(True)
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    assert window.calibration_panel.cam0_form.plane2 is stereo_settings.cam0_mapping_plane2
    assert window.calibration_panel.get_settings().world_scale_px_per_mm == pytest.approx(17.9)
    assert "stereo" in window.statusBar().currentMessage().lower()


def test_stereo_calibration_extraction_skipped_in_planar_mode(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    called = []
    monkeypatch.setattr(mw, "read_stereo_calibration_from_set", lambda path, idx: called.append(True))

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.project_panel.planar_radio.isChecked()
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()

    assert called == []


def test_stereo_calibration_extraction_failure_shows_status_and_does_not_crash(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw
    from piv_suite.config.schema import CalibrationSettings

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    monkeypatch.setattr(mw, "read_calibration_from_set",
                         lambda path, idx: CalibrationSettings())
    monkeypatch.setattr(mw, "read_stereo_calibration_from_set",
                         lambda path, idx: (_ for _ in ()).throw(RuntimeError("boom")))

    window = MainWindow()
    qtbot.addWidget(window)
    window.project_panel.stereo_radio.setChecked(True)
    window.project_panel.input_path_edit.setText(str(set_path))
    window.project_panel.input_path_edit.editingFinished.emit()  # must not raise

    assert "boom" in window.statusBar().currentMessage()


def test_calibration_panel_load_from_set_button_triggers_main_window_extraction(qtbot, monkeypatch, tmp_path):
    import piv_suite_gui.main_window as mw

    set_path = tmp_path / "recording.set"
    set_path.write_text("")
    stereo_settings = _fake_stereo_settings()
    monkeypatch.setattr(mw, "read_stereo_calibration_from_set", lambda path, idx: stereo_settings)

    window = MainWindow()
    qtbot.addWidget(window)
    window.project_panel.stereo_radio.setChecked(True)
    window.project_panel.input_path_edit.setText(str(set_path))  # no editingFinished yet

    window.calibration_panel.load_from_set_requested.emit()

    assert window.calibration_panel.cam0_form.plane2 is stereo_settings.cam0_mapping_plane2


def test_calibration_labels_use_math_notation(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    cam0 = window.calibration_panel.cam0_form
    # itemAt(0) is the DaVis-auto-load summary row added later; the
    # x0/x_span/y0/y_span label grid is itemAt(1).
    labels = [cam0.layout().itemAt(1).layout().itemAt(i).widget().text()
              for i in range(0, 10, 2)]
    assert "x₀:" in labels
    assert "y₀:" in labels
    assert "Δx:" in labels
    assert "Δy:" in labels
    assert cam0.coef_table.horizontalHeaderItem(0).text() == "dx(s,t)"
    assert cam0.coef_table.verticalHeaderItem(2).text() == "s²"  # "s2" -> s²
    assert cam0.coef_table.verticalHeaderItem(8).text() == "s²t"  # "s2t" -> s²t

    cp = window.calibration_panel
    from PySide6.QtWidgets import QLabel
    angle_labels = {lbl.text() for lbl in cp.findChildren(QLabel)} & {"α₁:", "α₂:", "β₁:", "β₂:"}
    assert angle_labels == {"α₁:", "α₂:", "β₁:", "β₂:"}
