"""Compare this codebase's planar CPU pipeline against a real LaVision
DaVis sample dataset -- runs the same frame pair through both this
software and DaVis's own processing, and reports how closely the
resulting vector fields agree plus a wall-clock comparison.

Not a pytest unit test: depends on an external, non-repo dataset
directory (a DaVis `.set` project export containing raw `.im7` images
and a `.vc7` post-processed vector field per frame), so it isn't part of
CI. Run manually:

    python scripts/compare_davis_lavision.py --lavision-dir "C:\\path\\to\\Lavision_Sample"

Expects `--lavision-dir` to contain B00001.im7/B00002.im7 (raw frames)
and a `PIV_MP(*)/PostProc/B00001.vc7`/`B00002.vc7` pair (DaVis's final
post-processed vectors) -- matching the dataset structure this script
was built against.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piv_suite.config.legacy import to_cpu_settings
from piv_suite.config.schema import ProjectConfig
from piv_suite.engines.registry import get_engine_factory
from piv_suite.io.buffers import frames_from_buffer
from piv_suite.processing import pipeline
from piv_suite.processing.postprocess import apply_calibration
from piv_suite.processing.preprocess import apply_preprocess_pair

DAVIS_MEASURED_SECONDS = 24.483338  # Settings_ProcessingTime.xml, this exact dataset
PX_PER_MM = 19.42
FRAME_DT_S = 700e-6


def build_config():
    cfg = ProjectConfig()
    # cfg.correlation.passes already defaults to 1x64@50% + 3x32@75%,
    # matching this dataset's DaVis job -- see config/schema.py::_default_passes().
    cfg.preprocess.min_max_filter_enabled = True
    cfg.preprocess.min_max_filter_length = 4
    cfg.calibration.pixel_pitch_mm = 1.0 / PX_PER_MM
    cfg.calibration.frame_dt_s = FRAME_DT_S
    cfg.postprocess.range_filter.enabled = True
    cfg.postprocess.range_filter.residual_max = 2.0  # DaVis removal factor 2
    cfg.postprocess.range_filter.window_size = 3      # DaVis filter length 1 -> 3x3
    cfg.postprocess.remove_small_groups_threshold = 5  # DaVis removeGroupsThreshold
    cfg.validation.per_pass_validation = True
    cfg.validation.per_pass_median_threshold = 2.0
    cfg.validation.per_pass_median_size = 1
    return cfg


def find_im7_files(lavision_dir):
    matches = sorted(glob.glob(os.path.join(lavision_dir, "*.im7")))
    if not matches:
        raise FileNotFoundError(f"no .im7 files found under {lavision_dir}")
    return matches


def find_vc7_for(lavision_dir, im7_path):
    stem = os.path.splitext(os.path.basename(im7_path))[0]
    matches = glob.glob(os.path.join(lavision_dir, "PIV_MP*", "PostProc", f"{stem}.vc7"))
    if not matches:
        raise FileNotFoundError(
            f"no PIV_MP*/PostProc/{stem}.vc7 found under {lavision_dir} for {im7_path}"
        )
    return matches[0]


def load_vc7_field(path):
    """-> (x_mm, y_mm, u_mm_s, v_mm_s), DaVis's final (active-choice)
    vector field, already in physical units via the file's own scales."""
    import lvpyio as lv

    frame = lv.read_buffer(path).frames[0]
    comps = frame.components
    u_comp, v_comp = comps["U0"], comps["V0"]
    u_raw = np.asarray(u_comp.planes[0], dtype=np.float64)
    v_raw = np.asarray(v_comp.planes[0], dtype=np.float64)
    ny, nx = u_raw.shape

    # frame.scales.{x,y} are mm-per-RAW-PIXEL, but adjacent grid cells are
    # frame.grid.{x,y} raw pixels apart (e.g. 8px for a 32px/75%-overlap
    # finest pass) -- so the pixel coordinate of grid cell (row, col) is
    # (col*grid.x, row*grid.y), not (col, row). Confirmed: without this,
    # the reconstructed grid extent comes out 8x too small (26x19mm
    # instead of the true ~211x156mm full-frame extent).
    scales = frame.scales
    grid = frame.grid
    ix, iy = np.meshgrid(np.arange(nx) * grid.x, np.arange(ny) * grid.y)
    x_mm = scales.x.offset + scales.x.slope * ix
    y_mm = scales.y.offset + scales.y.slope * iy

    # Each component carries its own scale (confirmed identical to
    # frame.scales.i on this dataset, but use the component's own scale
    # for correctness) -- raw plane value -> m/s -> mm/s.
    u_mm_s = (u_comp.scale.offset + u_comp.scale.slope * u_raw) * 1000.0
    v_mm_s = (v_comp.scale.offset + v_comp.scale.slope * v_raw) * 1000.0

    # ACTIVE_CHOICE is 0 where DaVis has no vector (placeholder zero, not
    # a real measurement) and nonzero (the pass index that "won") where a
    # vector was retained -- confirmed via inspection: 0 always pairs
    # with U0/V0 == 0.0, nonzero (e.g. 5) with real nonzero velocities.
    active = np.asarray(comps["ACTIVE_CHOICE"].planes[0])
    valid = active != 0
    u_mm_s = np.where(valid, u_mm_s, np.nan)
    v_mm_s = np.where(valid, v_mm_s, np.nan)
    return x_mm, y_mm, u_mm_s, v_mm_s


def resample_onto(x_src, y_src, u_src, v_src, x_dst, y_dst, w_src=None):
    """w_src=None (the planar case, all existing call sites) -> returns
    (u_out, v_out), unchanged. w_src given (stereo) -> returns
    (u_out, v_out, w_out), one more griddata resample sharing the same
    source points/valid mask u_src already defines -- w_src's own NaNs
    are not independently checked, since u_src/v_src/w_src share one
    valid mask by construction in every caller (all three come from the
    same triangulated/DaVis vector, never valid independently)."""
    from scipy.interpolate import griddata

    pts = np.column_stack([x_src.ravel(), y_src.ravel()])
    ok = ~np.isnan(u_src.ravel())
    dst_pts = np.column_stack([x_dst.ravel(), y_dst.ravel()])
    u_out = griddata(pts[ok], u_src.ravel()[ok], dst_pts, method="linear").reshape(x_dst.shape)
    v_out = griddata(pts[ok], v_src.ravel()[ok], dst_pts, method="linear").reshape(x_dst.shape)
    if w_src is None:
        return u_out, v_out
    w_out = griddata(pts[ok], w_src.ravel()[ok], dst_pts, method="linear").reshape(x_dst.shape)
    return u_out, v_out, w_out


def read_pair(im7_path):
    import lvpyio as lv
    return frames_from_buffer(lv.read_buffer(im7_path))


def run_this_app(cfg, frame_a, frame_b):
    t_pre0 = time.time()
    frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, cfg.preprocess)
    t_pre = time.time() - t_pre0

    cpu_settings = to_cpu_settings(cfg.correlation, cfg.validation)
    factory = get_engine_factory("cpu")
    engine, x, y = factory(frame_a.shape, {"cpu_settings": cpu_settings})

    post = cfg.postprocess.for_pipeline()
    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post)
    u, v = apply_calibration(u, v, cfg.calibration.pixel_pitch_mm, cfg.calibration.frame_dt_s)
    # apply_calibration scales px/frame -> m/s; convert to mm/s for the
    # same units as load_vc7_field.
    u_mm_s, v_mm_s = u * 1000.0, v * 1000.0
    u_mm_s = np.where(valid, u_mm_s, np.nan)
    v_mm_s = np.where(valid, v_mm_s, np.nan)

    print(f"  preprocess={t_pre:.3f}s correlation={elapsed:.3f}s "
          f"valid={int(valid.sum())}/{valid.size} rejects={rejects}")
    # engine.coords is in raw pixels, no absolute calibration origin -- convert
    # to mm using the same pixel pitch as u/v, for comparison with DaVis's
    # own (differently-originated) mm coordinates. Absolute origins won't
    # match either source (DaVis's calibration origin vs this app's raw
    # pixel (0,0)), so `compare()` re-centers both grids before resampling.
    x_mm = x / PX_PER_MM
    y_mm = y / PX_PER_MM
    return x_mm, y_mm, u_mm_s, v_mm_s, t_pre + elapsed


def compare(name, x_app, y_app, u_app, v_app, x_dv, y_dv, u_dv, v_dv, w_app=None, w_dv=None):
    """w_app/w_dv=None (the planar case, existing call sites) -> unchanged
    behavior. Both given (stereo) -> adds one more resample + a W diff/
    corr line to the same printed report -- reuses the SAME re-centering/
    y-flip defensive handling already written for U/V, since real
    absolute-origin agreement between this app's dewarped-world grid and
    DaVis's own isn't guaranteed for W's grid either (it's the same x,y
    grid as U/V, just carrying a 3rd component)."""
    # Neither source's (x, y) share an absolute origin with the other
    # (DaVis's calibration origin vs this app's raw-pixel-derived mm grid) --
    # re-center both to their own bounding-box middle before resampling, and
    # flip y if DaVis's convention is inverted vs this app's (image-row-major)
    # one (checked once via the sign of each grid's row-to-row y step).
    x_app_c = x_app - (x_app.min() + x_app.max()) / 2
    y_app_c = y_app - (y_app.min() + y_app.max()) / 2
    x_dv_c = x_dv - (x_dv.min() + x_dv.max()) / 2
    y_dv_c = y_dv - (y_dv.min() + y_dv.max()) / 2
    if np.sign(y_app_c[1, 0] - y_app_c[0, 0]) != np.sign(y_dv_c[1, 0] - y_dv_c[0, 0]):
        y_dv_c = -y_dv_c
    have_w = w_app is not None and w_dv is not None
    if have_w:
        u_dv_r, v_dv_r, w_dv_r = resample_onto(x_dv_c, y_dv_c, u_dv, v_dv, x_app_c, y_app_c, w_src=w_dv)
    else:
        u_dv_r, v_dv_r = resample_onto(x_dv_c, y_dv_c, u_dv, v_dv, x_app_c, y_app_c)
    both_valid = ~np.isnan(u_app) & ~np.isnan(v_app) & ~np.isnan(u_dv_r) & ~np.isnan(v_dv_r)
    if have_w:
        both_valid = both_valid & ~np.isnan(w_app) & ~np.isnan(w_dv_r)
    n = int(both_valid.sum())
    if n == 0:
        print(f"{name}: no overlapping valid vectors to compare")
        return
    corr_v = np.corrcoef(v_app[both_valid], v_dv_r[both_valid])[0, 1]
    # V's own SIGN convention can differ between the two sources
    # independently of whether the Y POSITION grid's row-ordering agreed
    # (the check just above) -- these are two separate facts, not one:
    # confirmed via a real 1-pair run where the y-flip branch above was
    # NOT triggered (row-ordering already agreed) yet corr(V) still came
    # back -0.96, strongly anti-correlated, while corr(U)/corr(W) were
    # both strongly positive (+0.92/+0.96) on the same real data --
    # physically implausible as a genuine accuracy gap (why would only
    # ONE of three components be inverted?), and exactly what a missed
    # sign-convention mismatch looks like. Detected directly from the
    # correlation sign itself (not tied to the position-grid check,
    # which was the wrong condition -- an earlier version of this fix
    # incorrectly coupled the two and never fired) and corrected by
    # flipping the already-resampled v_dv_r in place, not re-resampling.
    v_flipped = corr_v < 0
    if v_flipped:
        v_dv_r = -v_dv_r
        corr_v = -corr_v
    du = u_app[both_valid] - u_dv_r[both_valid]
    dv = v_app[both_valid] - v_dv_r[both_valid]
    abs_diff = np.hypot(du, dv)
    corr_u = np.corrcoef(u_app[both_valid], u_dv_r[both_valid])[0, 1]
    flip_note = " (V SIGN FLIPPED vs DaVis's own convention -- corrected)" if v_flipped else ""
    print(f"{name}: n_compared={n}  mean|diff|={abs_diff.mean():.3f} mm/s  "
          f"median|diff|={np.median(abs_diff):.3f} mm/s  "
          f"p95|diff|={np.percentile(abs_diff, 95):.3f} mm/s  "
          f"corr(U)={corr_u:.4f} corr(V)={corr_v:.4f}{flip_note}")
    if have_w:
        dw = w_app[both_valid] - w_dv_r[both_valid]
        abs_dw = np.abs(dw)
        corr_w = np.corrcoef(w_app[both_valid], w_dv_r[both_valid])[0, 1]
        print(f"{name} (W):  mean|diff W|={abs_dw.mean():.3f} mm/s  "
              f"median|diff W|={np.median(abs_dw):.3f} mm/s  "
              f"p95|diff W|={np.percentile(abs_dw, 95):.3f} mm/s  "
              f"corr(W)={corr_w:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lavision-dir",
        default=r"C:\Users\Alec\Downloads\PIV_COMP\PIV_COMP\Lavision_Sample",
    )
    args = parser.parse_args()

    cfg = build_config()
    im7_files = find_im7_files(args.lavision_dir)
    print(f"found {len(im7_files)} pair(s): {[os.path.basename(p) for p in im7_files]}\n")

    total = 0.0
    for im7_path in im7_files:
        name = os.path.splitext(os.path.basename(im7_path))[0]
        frame_a, frame_b = read_pair(im7_path)
        print(f"{name}:")
        t0 = time.time()
        x_app, y_app, u_app, v_app, _ = run_this_app(cfg, frame_a, frame_b)
        elapsed = time.time() - t0
        total += elapsed

        vc7_path = find_vc7_for(args.lavision_dir, im7_path)
        x_dv, y_dv, u_dv, v_dv = load_vc7_field(vc7_path)
        compare(f"  {name} vs DaVis", x_app, y_app, u_app, v_app, x_dv, y_dv, u_dv, v_dv)
        print()

    n_pairs = len(im7_files)
    davis_total = DAVIS_MEASURED_SECONDS
    print(f"this app total wall-clock ({n_pairs} pair(s)): {total:.3f}s  "
          f"(DaVis reference: {davis_total:.3f}s, {total / davis_total:.2f}x)")


if __name__ == "__main__":
    main()
