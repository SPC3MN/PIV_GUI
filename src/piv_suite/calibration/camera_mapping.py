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
                 raw_width=None, raw_height=None, ray_planes=None, px_per_mm=None):
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
        # (plane_a, plane_b) CameraMapping pair at two different calibrated
        # Z-planes, plus the canvas scale, retained ONLY so view_angles can
        # recover this camera's real viewing ray. A single interpolated
        # mapping has thrown that information away by construction: it
        # describes one plane, and a viewing direction needs two.
        # None (a genuinely single-plane calibration) -> view_angles raises.
        self.ray_planes = ray_planes
        self.px_per_mm = px_per_mm
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

    def raw_to_world(self, raw_x, raw_y, iters=12):
        """Inverse of world_to_raw, by fixed-point iteration.

        world_to_raw is x = xp - poly(s(xp), t(yp)), i.e. identity minus a
        small correction, so xp <- raw_x + poly(...) converges geometrically
        and needs no Jacobian. Converges to well under 1e-6 canvas px in a
        handful of iterations on real DaVis coefficients; 12 is generous."""
        xp = np.array(raw_x, dtype=float, copy=True)
        yp = np.array(raw_y, dtype=float, copy=True)
        for _ in range(iters):
            s, t = self.s(xp), self.t(yp)
            xp = raw_x + self._poly(s, t, self.dx_coefs)
            yp = raw_y + self._poly(s, t, self.dy_coefs)
        return xp, yp

    def view_angles(self, xp, yp):
        """Per-point viewing angles (alpha, beta) in DEGREES, in the
        convention calibration.reconstruction.reconstruct_stereo solves:
        dx_dewarped = dX - dZ*tan(alpha). Counterpart to
        pinhole.PinholeCameraMapping.view_angles, so both DaVis calibration
        models expose the same thing to stereo_view_angles.

        A viewing ray is recovered from the TWO calibrated Z-planes: the
        canvas point (xp, yp) on plane A maps to some raw sensor pixel; the
        SAME sensor pixel corresponds to a different canvas point on plane
        B, and those two world points define the ray. Then, for world
        points X_A at z_A and X_B at z_B on one ray,
            tan(alpha) = (X_B - X_A) / (z_B - z_A)
        which is exactly (C_x - X)/(C_z - z) for a camera centre C on that
        same ray -- matching the pinhole model's closed form without
        needing to know where the camera actually is.

        This is deliberately NOT the raw-sensor parallax the deleted
        _estimate_stereo_angles used. That measured the shift in SENSOR
        pixels, which is foreshortened by cos(alpha) for an angled camera,
        and never inverted the mapping back to world coordinates -- hence
        its systematic tan(34.5)/tan(44.1) = 0.710 ~= cos(44.1) error. The
        raw_to_world inversion above is the step it was missing.
        """
        if self.ray_planes is None or self.px_per_mm is None:
            raise ValueError(
                "view_angles needs two calibrated Z-planes and a canvas scale: this "
                "CameraMapping was built from a single plane (or by the marks fit), so "
                "there is no parallax to recover a viewing direction from. Supply "
                "StereoSettings.alpha1_deg/alpha2_deg/beta1_deg/beta2_deg manually for "
                "this calibration.")
        plane_a, plane_b = self.ray_planes
        dz_mm = plane_b.z_mm - plane_a.z_mm
        if not dz_mm:
            raise ValueError("view_angles: the two calibrated Z-planes share a Z position")
        raw_x, raw_y = plane_a.world_to_raw(xp, yp)
        xp_b, yp_b = plane_b.raw_to_world(raw_x, raw_y)
        dx_mm = (np.asarray(xp_b, float) - np.asarray(xp, float)) / self.px_per_mm
        dy_mm = (np.asarray(yp_b, float) - np.asarray(yp, float)) / self.px_per_mm
        # arctan of the RATIO, not arctan2 of the pair: this is a slope
        # (dX/dZ along the ray), and the principal value in (-90, 90) is
        # exactly what tan(alpha) means in reconstruct_stereo's model.
        # arctan2 would additionally encode which way along the ray the
        # camera lies, and would wrap by 180 deg whenever plane B sits
        # BELOW plane A in Z (dz < 0) -- which is the normal case here,
        # since DaVis writes the +Z plane first (z=+1.0 then z=-2.0 on the
        # reference project).
        return np.degrees(np.arctan(dx_mm / dz_mm)), np.degrees(np.arctan(dy_mm / dz_mm))


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


def _plane_mapping(settings):
    """A single-plane CameraMapping straight from one CameraMappingSettings,
    with no interpolation -- the per-Z-plane endpoints view_angles needs."""
    m = CameraMapping(settings.x0, settings.x_span, settings.y0, settings.y_span,
                      settings.dx_coefs, settings.dy_coefs, settings.name,
                      raw_width=settings.raw_width, raw_height=settings.raw_height)
    m.z_mm = settings.z_mm
    return m


def interpolate_camera_mapping(plane_a, plane_b, sheet_z_mm, name="", px_per_mm=None):
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
        # Keep the two un-interpolated endpoints alive: interpolation
        # collapses them to one plane, and view_angles needs both to
        # recover a viewing direction (see its docstring).
        ray_planes=(_plane_mapping(plane_a), _plane_mapping(plane_b)),
        px_per_mm=px_per_mm,
    )


def build_camera_mapping(settings, plane2_settings=None, sheet_z_mm=None, pinhole_settings=None,
                         px_per_mm=None):
    """The one place the app should build a camera mapping from config,
    instead of the 6-arg constructor call being repeated at every real
    call site (pipeline_worker.py, cli/main.py, preview_panel.py) and
    risking them drifting out of sync. settings/plane2_settings: config.
    schema.CameraMappingSettings. plane2_settings=None (the common case)
    means a single-plane mapping -- sheet_z_mm is ignored.

    pinhole_settings (config.schema.PinholeMappingSettings), when given,
    selects DaVis's OTHER internal calibration model and takes precedence:
    a pinhole camera is valid at any Z by construction, so there are no
    two planes to interpolate and sheet_z_mm just picks the plane the
    mapping is evaluated on (0.0 when not given). The returned object
    implements the same world_to_raw/dewarp_image/raw_domain_valid/
    view_angles interface as CameraMapping, so nothing downstream needs
    to know which model it got."""
    if pinhole_settings is not None:
        from .pinhole import PinholeCameraMapping, euler_zyx
        p = pinhole_settings
        return PinholeCameraMapping(
            p.f_px, p.cx, p.cy, p.k1, p.k2, p.p1, p.p2,
            euler_zyx(p.rx, p.ry, p.rz), [p.tx, p.ty, p.tz],
            p.scale_x, p.scale_y, p.offset_x, p.offset_y,
            z_mm=0.0 if sheet_z_mm is None else sheet_z_mm,
            name=p.name, raw_width=p.raw_width, raw_height=p.raw_height,
            fit_rms=p.fit_rms)
    if plane2_settings is None:
        return CameraMapping(settings.x0, settings.x_span, settings.y0, settings.y_span,
                              settings.dx_coefs, settings.dy_coefs, settings.name,
                              raw_width=settings.raw_width, raw_height=settings.raw_height)
    if sheet_z_mm is None:
        raise ValueError("sheet_z_mm is required when a camera has two calibrated Z-planes")
    return interpolate_camera_mapping(settings, plane2_settings, sheet_z_mm,
                                      name=settings.name, px_per_mm=px_per_mm)


def build_stereo_cameras(stereo_settings):
    """(cam0, cam1) for a StereoSettings, whichever DaVis calibration model
    it carries. The one place the four processing entry points
    (preview_panel, cli/main, parallel_stereo, pipeline_worker) should build
    a stereo pair, for the same reason build_camera_mapping exists: four
    hand-rolled copies of a 5-argument call drift."""
    st = stereo_settings
    return (build_camera_mapping(st.cam0_mapping, st.cam0_mapping_plane2, st.sheet_z_mm,
                                 st.cam0_pinhole, st.world_scale_px_per_mm),
            build_camera_mapping(st.cam1_mapping, st.cam1_mapping_plane2, st.sheet_z_mm,
                                 st.cam1_pinhole, st.world_scale_px_per_mm))


def stereo_angles_for(stereo_settings, cam0, cam1, x, y):
    """The triangulation angles (radians) to hand combine_stereo_pair, at the
    correlation grid (x, y) -- derived per pixel from the calibration, with
    any explicitly-set StereoSettings.alpha*/beta* used as a global override.
    Companion to stereo_fov_valid; called from the same four entry points."""
    st = stereo_settings
    return stereo_view_angles(cam0, cam1, x, y,
                              (st.alpha1_deg, st.alpha2_deg, st.beta1_deg, st.beta2_deg))


def stereo_view_angles(cam0, cam1, x, y, overrides=None):
    """Per-correlation-point stereo triangulation angles, in RADIANS, as the
    4-tuple (alpha1, alpha2, beta1, beta2) that pipeline.combine_stereo_pair
    and calibration.reconstruction.reconstruct_stereo already accept.

    The counterpart to stereo_fov_valid: same two camera mappings, same
    correlation grid (x, y), called from the same four processing entry
    points, so the definition of "what angle applies here" can't drift
    between them either.

    Returns ARRAYS matching x's shape, not scalars. reconstruct_stereo
    already broadcasts its angle arguments correctly, so this needs no
    change to the reconstruction kernel -- only to what gets handed to it.
    The real viewing angle varies several degrees across a stereo FOV;
    collapsing that to one number per camera is what used to put ~13% of
    |W| into U at the field edges (see config.schema.StereoSettings'
    alpha1_deg comment for the measured breakdown and why W barely moved,
    which is what made the error so easy to miss).

    overrides: an (alpha1_deg, alpha2_deg, beta1_deg, beta2_deg) tuple in
    which any entry that is not None REPLACES the derived per-pixel field
    with that single global angle -- the manual escape hatch for a rig
    whose calibration can't supply the geometry. All-None (the normal
    case) derives everything.
    """
    alpha1_ov, alpha2_ov, beta1_ov, beta2_ov = overrides or (None, None, None, None)
    shape = np.asarray(x).shape

    # Only derive what isn't overridden. A camera whose calibration genuinely
    # can't supply a viewing ray (a single calibrated Z-plane, or a marks
    # fit) raises from view_angles -- but if the caller supplied BOTH of that
    # camera's angles explicitly, there is nothing to derive and that
    # calibration must still work. Deriving eagerly would break exactly the
    # manual-entry case the override exists to serve.
    def resolve(cam, alpha_ov, beta_ov):
        if alpha_ov is not None and beta_ov is not None:
            return np.full(shape, float(alpha_ov)), np.full(shape, float(beta_ov))
        alpha, beta = cam.view_angles(x, y)
        if alpha_ov is not None:
            alpha = np.full(shape, float(alpha_ov))
        if beta_ov is not None:
            beta = np.full(shape, float(beta_ov))
        return alpha, beta

    a1, b1 = resolve(cam0, alpha1_ov, beta1_ov)
    a2, b2 = resolve(cam1, alpha2_ov, beta2_ov)
    return tuple(np.deg2rad(a) for a in (a1, a2, b1, b2))
