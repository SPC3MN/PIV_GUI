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
