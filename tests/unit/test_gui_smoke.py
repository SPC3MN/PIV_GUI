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


def test_main_window_has_fixed_width(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.minimumWidth() == window.maximumWidth()


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
