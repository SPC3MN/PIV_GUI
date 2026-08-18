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


def test_no_validation_group_in_settings_panel(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    sp = window.settings_panel
    assert not hasattr(sp, "s2n_threshold_spin")
    assert not hasattr(sp, "filter_method_combo")
    assert not hasattr(sp, "smoothn_check")
    # engines still need SOME validation defaults to run internally, just
    # not user-facing anymore
    settings = sp.get_validation_settings()
    assert settings.sig2noise_threshold == 1.05


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


def test_double_spinboxes_use_two_decimals(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    from PySide6.QtWidgets import QDoubleSpinBox
    for panel in (window.project_panel, window.settings_panel, window.calibration_panel):
        for spin in panel.findChildren(QDoubleSpinBox):
            assert spin.decimals() == 2, f"{spin} has {spin.decimals()} decimals, expected 2"


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
