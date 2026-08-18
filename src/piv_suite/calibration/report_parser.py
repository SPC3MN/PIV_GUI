"""Stub: parse a DaVis calibration report file directly into
CameraMappingSettings, instead of hand-transcribing coefficients off the
report panel into a config file/GUI form.

Not implemented yet -- calibration coefficients currently have to be
extracted manually from DaVis's calibration report (see the README
warnings in the original Stereo_PIV_GPU/Stereo_PIV_CPU repos re: alpha/beta
viewing angles being placeholders). This module exists so the GUI's
calibration panel can wire a "Load from DaVis report..." button to a
defined interface now (disabled/greyed until implemented), without needing
UI rework later.
"""


def parse_davis_calibration_report(path):
    """Parse a DaVis calibration report file and return a dict suitable for
    CameraMapping(**result) -- i.e. x0, x_span, y0, y_span, dx_coefs,
    dy_coefs, name.

    Not implemented -- see module docstring."""
    raise NotImplementedError(
        "Automated DaVis calibration report parsing isn't implemented yet -- "
        "enter calibration coefficients manually (see the calibration "
        "panel's form fields), reading them off DaVis's own calibration "
        "report panel."
    )
