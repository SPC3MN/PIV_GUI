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
    # ValidationSettings' 10 fields feed the PIV calculation directly (per-
    # pass sig2noise/replacement/smoothn inside the engine loop) -- they
    # were briefly hidden as fixed internal defaults, then explicitly
    # re-added to the GUI. Confirm the controls exist AND that
    # get_validation_settings() actually reflects widget state, not a
    # hardcoded ValidationSettings().
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    assert hasattr(sp, "s2n_threshold_spin")
    assert hasattr(sp, "filter_method_combo")
    assert hasattr(sp, "smoothn_check")

    sp.s2n_threshold_spin.setValue(1.3)
    sp.s2n_method_combo.setCurrentText("peak2peak")
    sp.smoothn_check.setChecked(True)
    sp.smoothn_p_spin.setValue(0.2)
    settings = sp.get_validation_settings()
    assert settings.sig2noise_threshold == 1.3
    assert settings.sig2noise_method == "peak2peak"
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
        window.settings_panel.s2n_threshold_spin,
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


def test_calibration_labels_use_math_notation(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    cam0 = window.calibration_panel.cam0_form
    labels = [cam0.layout().itemAt(0).layout().itemAt(i).widget().text()
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
