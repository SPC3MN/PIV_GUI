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
