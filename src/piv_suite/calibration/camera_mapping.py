"""DaVis's polynomial world<->raw camera calibration mapping.

Migrated unchanged from stereo_common.CameraMapping (identical across
Stereo_PIV_GPU and Stereo_PIV_CPU).
"""

import numpy as np


class CameraMapping:
    """
    DaVis 3rd-order polynomial calibration, single Z plane:
        s(x') = 2*(x' - x0) / x_span
        t(y') = 2*(y' - y0) / y_span
        x = x' - dx(s, t)
        y = y' - dy(s, t)
    dx_coefs / dy_coefs are dicts with keys
    '1','s','s2','s3','t','t2','t3','st','s2t','t2s' -- read directly off
    DaVis's calibration report panel, in order.
    """
    def __init__(self, x0, x_span, y0, y_span, dx_coefs, dy_coefs, name=""):
        self.x0, self.x_span = x0, x_span
        self.y0, self.y_span = y0, y_span
        self.dx_coefs, self.dy_coefs = dx_coefs, dy_coefs
        self.name = name
        self._cached_shape = None
        self._x_raw = self._y_raw = None

    def s(self, xp):
        return 2 * (xp - self.x0) / self.x_span

    def t(self, yp):
        return 2 * (yp - self.y0) / self.y_span

    @staticmethod
    def _poly(s, t, c):
        return (c['1'] + c['s'] * s + c['s2'] * s**2 + c['s3'] * s**3
                 + c['t'] * t + c['t2'] * t**2 + c['t3'] * t**3
                 + c['st'] * s * t + c['s2t'] * s**2 * t + c['t2s'] * t**2 * s)

    def world_to_raw(self, xp, yp):
        s, t = self.s(xp), self.t(yp)
        x = xp - self._poly(s, t, self.dx_coefs)
        y = yp - self._poly(s, t, self.dy_coefs)
        return x, y

    def _ensure_grid(self, world_shape):
        if self._cached_shape != world_shape:
            ny, nx = world_shape
            yp, xp = np.mgrid[0:ny, 0:nx].astype(np.float32)
            self._x_raw, self._y_raw = self.world_to_raw(xp, yp)
            self._cached_shape = world_shape

    def dewarp_image(self, raw_image, world_shape, order=1):
        """Backward-map raw_image onto a (world_shape) grid. Coordinate
        grid is computed once per (camera, world_shape) and cached."""
        from scipy.ndimage import map_coordinates
        self._ensure_grid(world_shape)
        return map_coordinates(raw_image, [self._y_raw, self._x_raw],
                                order=order, mode="constant", cval=0.0)


COEF_KEYS = ("1", "s", "s2", "s3", "t", "t2", "t3", "st", "s2t", "t2s")


def interpolate_camera_mapping(plane_a, plane_b, sheet_z_mm, name=""):
    """Build a CameraMapping for a real Z position that sits BETWEEN two
    DaVis-calibrated planes, by linearly interpolating every one of their
    parameters (x0, x_span, y0, y_span, and each of the 20 polynomial
    coefficients independently) at sheet_z_mm -- the standard dual-plane
    stereo-calibration technique. plane_a/plane_b are config.schema.
    CameraMappingSettings with `z_mm` set; CameraMapping's own math is
    completely unaffected, this only interpolates BETWEEN two already-
    compatible coefficient sets before construction."""
    z_a, z_b = plane_a.z_mm, plane_b.z_mm
    w = 0.0 if z_b == z_a else (sheet_z_mm - z_a) / (z_b - z_a)
    if not (0.0 <= w <= 1.0):
        print(f"[warn] camera_mapping: sheet_z_mm={sheet_z_mm} is outside the "
              f"calibrated plane range [{min(z_a, z_b)}, {max(z_a, z_b)}] -- extrapolating")

    def lerp(a, b):
        return a + w * (b - a)

    dx_coefs = {k: lerp(plane_a.dx_coefs[k], plane_b.dx_coefs[k]) for k in COEF_KEYS}
    dy_coefs = {k: lerp(plane_a.dy_coefs[k], plane_b.dy_coefs[k]) for k in COEF_KEYS}
    return CameraMapping(
        lerp(plane_a.x0, plane_b.x0), lerp(plane_a.x_span, plane_b.x_span),
        lerp(plane_a.y0, plane_b.y0), lerp(plane_a.y_span, plane_b.y_span),
        dx_coefs, dy_coefs, name,
    )


def build_camera_mapping(settings, plane2_settings=None, sheet_z_mm=None):
    """The one place the app should build a CameraMapping from config,
    instead of the 6-arg constructor call being repeated at every real
    call site (pipeline_worker.py, cli/main.py, preview_panel.py) and
    risking them drifting out of sync. settings/plane2_settings: config.
    schema.CameraMappingSettings. plane2_settings=None (the common case)
    means a single-plane mapping -- sheet_z_mm is ignored."""
    if plane2_settings is None:
        return CameraMapping(settings.x0, settings.x_span, settings.y0, settings.y_span,
                              settings.dx_coefs, settings.dy_coefs, settings.name)
    if sheet_z_mm is None:
        raise ValueError("sheet_z_mm is required when a camera has two calibrated Z-planes")
    return interpolate_camera_mapping(settings, plane2_settings, sheet_z_mm, name=settings.name)
