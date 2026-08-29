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
    def __init__(self, x0, x_span, y0, y_span, dx_coefs, dy_coefs, name="",
                 raw_width=None, raw_height=None):
        self.x0, self.x_span = x0, x_span
        self.y0, self.y_span = y0, y_span
        self.dx_coefs, self.dy_coefs = dx_coefs, dy_coefs
        self.name = name
        # This camera's real raw sensor size (DaVis Calibration.xml's
        # OriginalImageSize), used by raw_domain_valid to tell a genuinely
        # out-of-view correlation point (world_to_raw lands outside the
        # real sensor) from an in-view one. None/0 (the default -- every
        # existing caller that doesn't pass these, e.g. the marks-fit
        # calibration path or a hand-built CameraMapping in a test) means
        # "unknown, no masking possible" -- raw_domain_valid then always
        # returns True, leaving behavior for those callers unchanged.
        self.raw_width, self.raw_height = raw_width, raw_height
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

    def raw_domain_valid(self, x, y):
        """True where world-grid point (x, y) maps, via this camera's own
        world_to_raw, to a pixel actually INSIDE its real raw sensor --
        [0, raw_width) x [0, raw_height). False where it maps outside
        (this camera genuinely can't see that point at all).

        WHY THIS EXISTS, separately from dewarp_image's existing cval=0.0:
        dewarp_image already correctly zero-fills a dewarped pixel outside
        this camera's true view (map_coordinates' cval=0.0 for any
        (x_raw, y_raw) sample landing outside the real raw frame) -- the
        IMAGE is already right. But a correlation WINDOW straddling that
        zero-padded region can still correlate mostly-zero content against
        mostly-zero content and return a spurious, statistically
        plausible-looking displacement -- nothing about a flat black patch
        trips the range/std-dev outlier filters in processing.postprocess,
        since those look at whether a vector's VALUE is an outlier, not
        whether the pixels behind it were ever real image content. DaVis's
        own stereo output doesn't report vectors there at all (crops the
        overlapping-FOV region rather than emitting bad ones); this is the
        geometric check that lets this app's own `valid` mask do the same
        thing, at the correlation grid's own points -- confirmed directly
        against real dual-plane calibration data that grid points near a
        world_shape canvas's edges fall outside one camera's raw_width/
        raw_height while a same-region point stays inside the other's,
        exactly the "only one camera can see this" case a stereo pair
        needs both cameras' agreement to reject.

        Deliberately evaluated directly on the (typically few-hundred-
        point) CORRELATION grid handed in by the caller, NOT routed
        through _ensure_grid/dewarp_image's cached PER-PIXEL world_shape
        grid (which can be several million points, e.g. 3067x5874) --
        world_to_raw is a handful of polynomial evaluations, cheap enough
        to simply call fresh here every time rather than adding a second
        cache keyed by grid shape for what's already a tiny computation.

        raw_width/raw_height default to None (see __init__) for any
        caller that doesn't have OriginalImageSize data (the marks-fit
        calibration path, or a hand-built CameraMapping in a test) --
        this returns all-True (no masking) in that case, exactly like
        this schema's existing 0/None-means-not-available convention
        elsewhere, so those callers' behavior is unaffected."""
        if not self.raw_width or not self.raw_height:
            return np.ones(np.asarray(x).shape, dtype=bool)
        raw_x, raw_y = self.world_to_raw(x, y)
        return (raw_x >= 0) & (raw_x < self.raw_width) & (raw_y >= 0) & (raw_y < self.raw_height)

def stereo_fov_valid(cam0, cam1, x, y):
    """The one place stereo's "does either camera actually have usable
    real data at this world-grid point" mask is computed -- every
    processing entry point (preview_panel/cli/pipeline_worker/
    parallel_stereo) should call this instead of hand-rolling the AND of
    raw_domain_valid across both cameras, so the definition of "valid"
    can't drift out of sync between them the way the 6-arg CameraMapping()
    constructor used to before build_camera_mapping existed.

    Requires BOTH cameras to actually see this point on their real raw
    sensor (raw_domain_valid). world_to_raw's full polynomial transform
    (not a plain affine one) means this naturally produces a tapered,
    non-rectangular shape in world-grid space for an angled camera pair --
    confirmed directly against real DaVis output: DaVis's own valid region
    is a keystone/hexagon (an 11-cell-wide sliver at the very top row,
    widening to the full rectangular width by row ~18), and this function
    already reproduces that shape closely (90.5% cell-for-cell agreement,
    0% of DaVis's real valid cells excluded, once properly compared on a
    common physical grid rather than raw array indices, which don't line
    up between this app's and DaVis's differently-sized grids).

    An earlier version of this ALSO required |s|<=1/|t|<=1 (the
    calibration polynomial's own normalized fit domain) for both cameras,
    on the theory that a point outside a calibration target's real
    footprint is an untrustworthy extrapolation DaVis wouldn't report
    either. Measured against real data, that check made the match WORSE,
    not better -- excluding up to 25% of cells DaVis actually shows real
    data for, at the literal |s|,|t|<=1 bound. The threshold needed for it
    to stop cutting off real DaVis data (~1.4) turned out to make it a
    complete no-op for this rig's real calibration (raw_domain_valid was
    already the more restrictive, binding constraint everywhere), so it
    was removed rather than kept as dead weight -- see git history if a
    future rig's calibration ever needs it reconsidered."""
    return cam0.raw_domain_valid(x, y) & cam1.raw_domain_valid(x, y)


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
    # raw_width/raw_height are a fixed per-CAMERA constant (the physical
    # sensor doesn't change between calibrated Z-planes), so no lerp() --
    # plane_a/plane_b carry the identical value (both populated straight
    # off the same camera's OriginalImageSize by _exact_camera_mapping_
    # from_calibration_xml); plane_a's is used as-is.
    return CameraMapping(
        lerp(plane_a.x0, plane_b.x0), lerp(plane_a.x_span, plane_b.x_span),
        lerp(plane_a.y0, plane_b.y0), lerp(plane_a.y_span, plane_b.y_span),
        dx_coefs, dy_coefs, name,
        raw_width=plane_a.raw_width, raw_height=plane_a.raw_height,
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
                              settings.dx_coefs, settings.dy_coefs, settings.name,
                              raw_width=settings.raw_width, raw_height=settings.raw_height)
    if sheet_z_mm is None:
        raise ValueError("sheet_z_mm is required when a camera has two calibrated Z-planes")
    return interpolate_camera_mapping(settings, plane2_settings, sheet_z_mm, name=settings.name)
