"""DaVis's `PinholeOpenCV` calibration model, decoded exactly.

DaVis writes one of two internal calibration Types into Calibration.xml.
`Polynomial3rdOrder` is handled by camera_mapping.CameraMapping; this module
handles the other one, a standard OpenCV pinhole+distortion camera with
explicit extrinsics.

DECODED CONVENTIONS -- every one of these was verified against real ground
truth (`MarkPositionTable.xml`), not assumed:

    f_px      = FocalLengthPixel / SensorPixelSizeMm
                The stored value is in MILLIMETRES despite the element name
                (75.9515 mm / 0.00274 mm/px = 27719.53 px). Reading it as
                pixels is off by ~365x and was the prior decode attempt's
                first wrong turn.
    (cx, cy)  = PrincipalPoint, in RAW SENSOR pixels. It is BOTH the
                projection centre and the distortion centre.
    OriginPixelPosition is NOT a sensor quantity and plays no part in the
                projection. It is the corrected-image pixel of world (0,0):
                exactly -OffsetMm/FactorMmPerPixel (matches to all 16
                digits on real data, both axes, both cameras). It is
                identical for both cameras, which is the tell -- a
                per-camera sensor quantity could not be.
    R         = Rz @ Ry @ Rx   (Euler order 'zyx')
    Xc        = R @ Xw + TranslationMm
    distortion= textbook OpenCV Brown-Conrady on NORMALIZED camera
                coordinates (xn = Xc/Zc), with NO rescaling of r.

k2 values of -10.6 / +23.5 look implausibly large for standard OpenCV and
were previously read as evidence of a non-standard normalization. They are
not: with the correct f_px ~= 27720, r only reaches ~0.08 over the real
sensor, so the k2*r^4 term contributes well under a pixel. A free refit of
the distortion coefficients recovers the stored values (k1 0.17328 vs
0.1731660 stored, k2 -10.7297 vs -10.642952 stored), confirming the
convention rather than merely fitting something that works.

VALIDATION. On a self-consistent snapshot the stored numbers, used verbatim
with nothing fitted, reproduce DaVis's own declared <FitError RMS> to a
relative difference of 3.6e-14 (camera 1) and 6.2e-15 (camera 2) -- i.e.
exactly, to floating-point precision. <FitError RMS> is the 2-D residual
RMS sqrt(mean(dx^2+dy^2)) WEIGHTED by each mark's own `w` attribute; both
details are required to land on DaVis's number (unweighted, the same model
gives 0.4533/0.8641 against declared 0.4259/0.6614).

A NOTE ON MARK TABLES, since it cost the previous attempt the decode:
DaVis writes the SAME MarkPositionTable.xml into both camera1\\ and
camera2\\, and each copy contains BOTH <Camera CameraNumber=...> blocks.
Selecting marks by FOLDER pairs camera 2's marks with camera 1's
parameters and produces exactly the "y is sub-pixel but x is several px
off" residual that was previously read as a failed decode. Always select
by CameraNumber -- see read_marks below.
"""

import xml.etree.ElementTree as ET

import numpy as np


def euler_zyx(rx, ry, rz):
    """R = Rz @ Ry @ Rx. Uniquely determined against real data: the
    next-best Euler order reprojects 10.2x worse, and a Rodrigues/axis-angle
    reading 1225x worse."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class PinholeCameraMapping:
    """One camera's exact DaVis PinholeOpenCV mapping.

    Implements the same interface calibration.camera_mapping.CameraMapping
    exposes to the rest of the app -- world_to_raw / dewarp_image /
    raw_domain_valid / view_angles, all taking CORRECTED-CANVAS PIXEL
    indices (xp, yp), not millimetres -- so every downstream consumer
    (dewarping, the stereo FOV mask, the processing entry points) works
    with either calibration model without knowing which it has.

    scale_x/scale_y/offset_x/offset_y convert canvas pixel <-> world mm
    and come straight off this camera's own <Scales> block. z_mm is the
    laser-sheet world Z the mapping is evaluated at; unlike the polynomial
    model there is no per-plane coefficient set to interpolate between,
    because the pinhole model is valid at any Z by construction.
    """

    def __init__(self, f_px, cx, cy, k1, k2, p1, p2, R, T,
                 scale_x, scale_y, offset_x, offset_y,
                 z_mm=0.0, name="", raw_width=None, raw_height=None,
                 fit_rms=None, corrected_wh=None):
        self.f_px, self.cx, self.cy = f_px, cx, cy
        self.k1, self.k2, self.p1, self.p2 = k1, k2, p1, p2
        self.R, self.T = np.asarray(R, float), np.asarray(T, float)
        self.scale_x, self.scale_y = scale_x, scale_y
        self.offset_x, self.offset_y = offset_x, offset_y
        self.z_mm = z_mm
        self.name = name
        self.raw_width, self.raw_height = raw_width, raw_height
        self.fit_rms = fit_rms
        self.corrected_wh = corrected_wh
        self._cached_shape = None
        self._x_raw = self._y_raw = None

    # ------------------------------------------------------------ geometry --
    @property
    def centre(self):
        """Camera centre in world mm."""
        return -self.R.T @ self.T

    def project(self, Xw):
        """World mm (..., 3) -> raw sensor pixel (..., 2)."""
        Xw = np.asarray(Xw, float)
        Xc = Xw @ self.R.T + self.T
        z = Xc[..., 2]
        xn, yn = Xc[..., 0] / z, Xc[..., 1] / z
        r2 = xn * xn + yn * yn
        rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd = xn * rad + 2 * self.p1 * xn * yn + self.p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + self.p1 * (r2 + 2 * yn * yn) + 2 * self.p2 * xn * yn
        return np.stack([self.f_px * xd + self.cx, self.f_px * yd + self.cy], axis=-1)

    def canvas_to_world_mm(self, xp, yp):
        """Corrected-canvas pixel indices -> world mm on this mapping's Z plane."""
        return (self.offset_x + np.asarray(xp, float) * self.scale_x,
                self.offset_y + np.asarray(yp, float) * self.scale_y)

    # -------------------------------------------- CameraMapping's interface --
    def world_to_raw(self, xp, yp):
        """Corrected-canvas pixel -> raw sensor pixel. Same contract as
        CameraMapping.world_to_raw, so dewarping and the FOV mask are
        model-agnostic."""
        X, Y = self.canvas_to_world_mm(xp, yp)
        Xw = np.stack([X, Y, np.full_like(np.asarray(X, float), self.z_mm)], axis=-1)
        uv = self.project(Xw)
        return uv[..., 0], uv[..., 1]

    def _ensure_grid(self, world_shape):
        if self._cached_shape != world_shape:
            ny, nx = world_shape
            yp, xp = np.mgrid[0:ny, 0:nx].astype(np.float64)
            self._x_raw, self._y_raw = self.world_to_raw(xp, yp)
            self._cached_shape = world_shape

    def dewarp_image(self, raw_image, world_shape, order=1):
        from scipy.ndimage import map_coordinates
        self._ensure_grid(world_shape)
        return map_coordinates(raw_image, [self._y_raw, self._x_raw],
                               order=order, mode="constant", cval=0.0)

    def raw_domain_valid(self, x, y):
        """True where this canvas point maps inside the real raw sensor --
        see CameraMapping.raw_domain_valid for why this exists separately
        from dewarp_image's zero fill."""
        if not self.raw_width or not self.raw_height:
            return np.ones(np.asarray(x).shape, dtype=bool)
        raw_x, raw_y = self.world_to_raw(x, y)
        return ((raw_x >= 0) & (raw_x < self.raw_width)
                & (raw_y >= 0) & (raw_y < self.raw_height))

    def view_angles(self, xp, yp):
        """Per-point viewing angles (alpha, beta) in DEGREES at canvas pixel
        (xp, yp), in exactly the convention calibration.reconstruction.
        reconstruct_stereo solves:  dx_dewarped = dX - dZ*tan(alpha).

        Exact for this model: the ray from the world point to the camera
        centre is known in closed form, no inversion or finite differencing
        needed. alpha depends only on x and beta only on y for this rig,
        but that is a property of the geometry, not an assumption made here.
        """
        X, Y = self.canvas_to_world_mm(xp, yp)
        C = self.centre
        vx, vy, vz = C[0] - X, C[1] - Y, C[2] - self.z_mm
        return np.degrees(np.arctan2(vx, vz)), np.degrees(np.arctan2(vy, vz))

    def at_z(self, z_mm):
        """A copy of this mapping evaluated at a different laser-sheet Z."""
        return PinholeCameraMapping(
            self.f_px, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2,
            self.R, self.T, self.scale_x, self.scale_y, self.offset_x, self.offset_y,
            z_mm=z_mm, name=self.name, raw_width=self.raw_width,
            raw_height=self.raw_height, fit_rms=self.fit_rms,
            corrected_wh=self.corrected_wh)


def read_pinhole_camera(calibration_xml_path, camera_identifier, z_mm=0.0, name=""):
    """Parse one camera's PinholeOpenCV CoordinateMapper out of a DaVis
    Calibration.xml. Returns None if that camera's mapper is a different
    Type (or absent), matching davis_set._exact_camera_mapping_from_
    calibration_xml's own "None means not this model" convention."""
    root = ET.parse(calibration_xml_path).getroot()
    cm = root.find(".//CoordinateMapper[@CameraIdentifier='%s']" % camera_identifier)
    if cm is None or cm.attrib.get("Type") != "PinholeOpenCV":
        return None
    common = cm.find(".//CommonParameters")
    intern = cm.find(".//InternalCameraParameters")
    extern = cm.find(".//ExternalCameraParameters")
    sx = cm.find(".//Scales/LinearScaleX")
    sy = cm.find(".//Scales/LinearScaleY")
    if common is None or intern is None or extern is None or sx is None or sy is None:
        return None

    pixel_size_mm = float(intern.find("SensorPixelSizeMm").attrib["Value"])
    focal_mm = float(intern.find("FocalLengthPixel").attrib["x"])
    pp = intern.find("PrincipalPoint").attrib
    rad = intern.find("RadialDistortion").attrib
    tan = intern.find("TangentialDistortion").attrib
    tr = extern.find("TranslationMm").attrib
    ro = extern.find("RotationAngles").attrib
    raw = common.find("OriginalImageSize").attrib
    corr = common.find("CorrectedImageSize").attrib
    fit = common.find("FitError")

    return PinholeCameraMapping(
        f_px=focal_mm / pixel_size_mm,
        cx=float(pp["x"]), cy=float(pp["y"]),
        k1=float(rad["radialDistortionCoefficient1"]),
        k2=float(rad["radialDistortionCoefficient2"]),
        p1=float(tan["tangentialDistortionCoefficient1"]),
        p2=float(tan["tangentialDistortionCoefficient2"]),
        R=euler_zyx(float(ro["Rx"]), float(ro["Ry"]), float(ro["Rz"])),
        T=[float(tr["Tx"]), float(tr["Ty"]), float(tr["Tz"])],
        scale_x=float(sx.attrib["FactorMmPerPixel"]),
        scale_y=float(sy.attrib["FactorMmPerPixel"]),
        offset_x=float(sx.attrib["OffsetMm"]),
        offset_y=float(sy.attrib["OffsetMm"]),
        z_mm=z_mm, name=name,
        raw_width=int(raw["Width"]), raw_height=int(raw["Height"]),
        fit_rms=float(fit.attrib["RMS"]) if fit is not None else None,
        corrected_wh=(int(corr["Width"]), int(corr["Height"])),
    )


def read_marks(mark_table_xml_path, camera_number):
    """-> (world_mm (N,3), raw_px (N,2), weight (N)) for ONE camera.

    DaVis writes the SAME file into camera1\\ and camera2\\, and each copy
    contains BOTH <Camera> blocks -- always select by CameraNumber, never
    by which folder the file was read from. Getting this wrong silently
    pairs one camera's marks with the other's parameters (see this
    module's docstring)."""
    root = ET.parse(mark_table_xml_path).getroot()
    cam = root.find(".//Camera[@CameraNumber='%s']" % camera_number)
    if cam is None:
        return None
    world, raw, weight = [], [], []
    for view in cam.findall("View"):
        for mark in view.findall("Mark"):
            w = mark.find("WorldPos").attrib
            r = mark.find("RawPos").attrib
            world.append([float(w["x"]), float(w["y"]), float(w["z"])])
            raw.append([float(r["x"]), float(r["y"])])
            weight.append(float(mark.attrib.get("w", 1.0)))
    return np.array(world), np.array(raw), np.array(weight)


def weighted_reprojection_rms(camera, marks):
    """DaVis's own <FitError RMS> statistic: the 2-D residual RMS weighted
    by each mark's `w`. Both the 2-D-ness and the weighting are required to
    reproduce DaVis's stored number."""
    world, raw, weight = marks
    resid = camera.project(world) - raw
    return float(np.sqrt(((resid ** 2).sum(axis=1) * weight).sum() / weight.sum()))
