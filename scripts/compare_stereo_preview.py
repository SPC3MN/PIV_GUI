"""Compare this app's REAL stereo preview pipeline (piv_suite_gui.widgets.
preview_panel.PreviewPanel._compute_stereo -- the exact code a user's
Preview button runs, not a from-scratch reimplementation) against real
LaVision DaVis stereo output, for a small number of pairs from a real
`.set` project.

Stereo counterpart to compare_velocity_fields.py/compare_davis_lavision.py
(both planar-only, hardcoded to a different sample dataset) -- reuses
their genuinely dataset-agnostic pieces (field_stats/normalized_local_
residual/print_stats_table/plot_comparison, resample_onto/compare) rather
than duplicating them. See those two modules' own docstrings/comments for
the reused pieces' own rationale.

Two real, unavoidable calibration gaps this script has to ask the caller
to fill in explicitly (see --alpha*/--beta*/--sheet-z-mm below), neither
derivable from Calibration.xml:

1. Stereo triangulation angles (alpha1/alpha2/beta1/beta2). DaVis
   triangulates from each camera's full per-pixel polynomial mapping
   (which already encodes true local viewing geometry everywhere) --
   this app's own calibration.reconstruction.reconstruct_stereo uses a
   deliberately simpler model instead, one single global angle per
   camera applied uniformly across the whole image. Calibration.xml has
   no reason to store a number DaVis's own algorithm never uses -- this
   is a real model difference between this app and DaVis, not a missing
   calibration field. A reasonable ESTIMATE for this app's own required
   scalar can be derived from how far the two calibrated Z-planes'
   CoefficientsA/B constant terms (a_o/b_o, evaluated at image center)
   shift over the real Z separation (z_mm = ZPosition / PixelPerMmFactor
   -- see io.davis_set._exact_camera_mapping_from_calibration_xml's own
   docstring) -- but it's a center-of-FOV approximation of a model DaVis
   doesn't use, not a measured rig value. This script requires the
   caller to supply it explicitly rather than silently defaulting to
   the GUI calibration panel's own generic +-45/0 placeholder.
2. sheet_z_mm: the real-world Z of the laser sheet for THIS recording --
   an acquisition-time quantity, never a calibration-file one (see
   config.schema.StereoSettings.sheet_z_mm's own comment). Required
   whenever the extracted calibration has two Z-planes
   (calibration.camera_mapping.build_camera_mapping raises otherwise);
   this script checks and fails fast with a clear message instead of
   letting that exception surface three frames deep.

Run (never processes more than --max-pairs pairs; this is a small,
explicitly-scoped validation tool, not a batch runner):

    python scripts/compare_stereo_preview.py \\
        --set-file "J:\\Final_Stereo\\Swirl\\On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0.set" \\
        --vc7-dir "J:\\Final_Stereo\\Swirl\\On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0\\StereoPIV_MPd(3x32x32_75%ov)" \\
        --start-index 0 --max-pairs 3 \\
        --sheet-z-mm -0.5

--alpha1-deg/--alpha2-deg/--beta1-deg/--beta2-deg are now OPTIONAL (added
5ac0563, after this script itself was first written) -- read_stereo_
calibration_from_set already auto-derives them from the calibration
mapping itself (io.davis_set._estimate_stereo_angles); pass one only to
override the auto-derived estimate with an explicit value.
"""

import argparse
import os
import sys

# QT_QPA_PLATFORM must be set BEFORE any PySide6 import -- this is not an
# existing convention elsewhere in this repo's own code (tests/unit/ uses
# pytest-qt's qtbot fixture instead), but it's the right, minimal way to
# construct a real PreviewPanel (a QWidget) headlessly here: we never
# .show() it or click anything, only call its pure compute methods.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from compare_davis_lavision import compare
from compare_velocity_fields import field_stats, plot_comparison, print_stats_table

from piv_suite.config.schema import (
    CorrelationSettings, PostProcessSettings, PreprocessSettings, ProjectSettings,
    ValidationSettings,
)
from piv_suite.io.davis_set import read_calibration_from_set, read_stereo_calibration_from_set


def load_vc7_stereo_field(path):
    """-> (x_mm, y_mm, u_mm_s, v_mm_s, w_mm_s_or_None), DaVis's final
    (post-ACTIVE_CHOICE, post-MASK/ENABLED) vector field, in physical
    units via the file's own scales -- see module docstring for why this
    uses lvpyio's own as_masked_array() rather than compare_davis_
    lavision.load_vc7_field's hardcoded U0/V0 + "ACTIVE_CHOICE != 0"
    extraction (confirmed incomplete: doesn't handle DaVis's "smoothed/
    filled" pseudo-choices 4/5, and has no W/stereo support at all)."""
    import lvpyio as lv

    frame = lv.read_buffer(path).frames[0]
    arr = frame.as_masked_array(plane=0)
    valid = ~np.ma.getmaskarray(arr)["u"]
    u_mm_s = np.where(valid, arr["u"].filled(np.nan) * 1000.0, np.nan)
    v_mm_s = np.where(valid, arr["v"].filled(np.nan) * 1000.0, np.nan)
    w_mm_s = None
    if frame.is_3c:
        w_mm_s = np.where(valid, arr["w"].filled(np.nan) * 1000.0, np.nan)

    # Same grid-position reconstruction as compare_davis_lavision.
    # load_vc7_field -- frame.scales are mm-per-RAW-pixel, but adjacent
    # grid cells are frame.grid raw pixels apart, not 1 apart.
    ny, nx = u_mm_s.shape
    scales, grid = frame.scales, frame.grid
    ix, iy = np.meshgrid(np.arange(nx) * grid.x, np.arange(ny) * grid.y)
    x_mm = scales.x.offset + scales.x.slope * ix
    y_mm = scales.y.offset + scales.y.slope * iy
    return x_mm, y_mm, u_mm_s, v_mm_s, w_mm_s


def build_stereo_settings_bundle(set_path, multiset_index, angles_deg, sheet_z_mm, dewarp_order,
                                  min_max_filter_length=0, sig2noise_threshold=None):
    """Returns (project, preprocess, correlation, validation, post,
    calibration, stereo_settings) -- exactly the 7 positional args
    PreviewPanel._compute_stereo expects, built from PLAIN, UNMODIFIED
    schema defaults (matching compare_velocity_fields.gui_default_
    config's own "what a real user gets without touching anything"
    philosophy -- no hand-tuned settings to make numbers look better)
    except for the calibration fields that must come from the real
    project itself.

    angles_deg: (alpha1, alpha2, beta1, beta2), each Optional[float] --
    None (the default, when --alpha*/--beta* aren't passed on the CLI)
    means "keep whatever read_stereo_calibration_from_set already
    auto-derived" (io.davis_set._estimate_stereo_angles, added in
    5ac0563 -- estimates alpha/beta straight from the two-Z-plane
    calibration mapping itself, validated to corr(W)=0.955-0.983 against
    real DaVis reference data). This function only OVERRIDES with an
    explicit CLI value when the caller actually supplied one -- e.g. to
    A/B a hand-measured rig angle against the auto-derived estimate."""
    alpha1, alpha2, beta1, beta2 = angles_deg

    project = ProjectSettings(input_mode="set", input_path=set_path, mode="stereo",
                               multiset_index=multiset_index)
    preprocess = PreprocessSettings()
    if min_max_filter_length > 0:
        preprocess.min_max_filter_enabled = True
        preprocess.min_max_filter_length = min_max_filter_length
    correlation = CorrelationSettings()
    validation = ValidationSettings()
    if sig2noise_threshold is not None:
        validation.per_pass_sig2noise_threshold = sig2noise_threshold
    post = PostProcessSettings()

    calibration = read_calibration_from_set(set_path, multiset_index)
    stereo_settings = read_stereo_calibration_from_set(set_path, multiset_index)
    if alpha1 is not None:
        stereo_settings.alpha1_deg = alpha1
    if alpha2 is not None:
        stereo_settings.alpha2_deg = alpha2
    if beta1 is not None:
        stereo_settings.beta1_deg = beta1
    if beta2 is not None:
        stereo_settings.beta2_deg = beta2
    stereo_settings.dewarp_order = dewarp_order

    if stereo_settings.cam0_mapping_plane2 is not None and sheet_z_mm is None:
        raise SystemExit(
            "This calibration has two Z-planes (a real dual-plane stereo calibration) -- "
            "--sheet-z-mm is required (the real-world Z of the laser sheet for this "
            "recording; never derivable from Calibration.xml, see this module's own "
            "docstring). Pass --sheet-z-mm explicitly."
        )
    stereo_settings.sheet_z_mm = sheet_z_mm

    print(f"calibration: pixel_pitch_mm={calibration.pixel_pitch_mm}  "
          f"frame_dt_s={calibration.frame_dt_s}  "
          f"({'m/s output' if calibration.frame_dt_s is not None else 'mm/frame output -- no frame_dt_s found'})")
    print(f"stereo calibration: {stereo_settings.cam0_mapping.name}")
    print(f"angles (CALIBRATION-DERIVED ESTIMATE, NOT A MEASURED RIG VALUE -- see module "
          f"docstring): alpha1={stereo_settings.alpha1_deg}deg alpha2={stereo_settings.alpha2_deg}deg "
          f"beta1={stereo_settings.beta1_deg}deg beta2={stereo_settings.beta2_deg}deg")
    print(f"sheet_z_mm: {sheet_z_mm} (APPROXIMATION unless --sheet-z-mm was a real measured value)")

    return project, preprocess, correlation, validation, post, calibration, stereo_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-file", required=True)
    parser.add_argument("--vc7-dir", required=True,
                         help="DaVis's own result vectors (B00001.vc7, B00002.vc7, ... 1-based).")
    parser.add_argument("--multiset-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, required=True,
                         help="Required -- this tool must never silently default to "
                              "processing an entire (possibly 1000+ pair) recording.")
    parser.add_argument("--alpha1-deg", type=float, default=None,
                         help="Override the auto-derived alpha1 (see io.davis_set."
                              "_estimate_stereo_angles, 5ac0563) with an explicit value. "
                              "Omit to use the auto-derived estimate.")
    parser.add_argument("--alpha2-deg", type=float, default=None)
    parser.add_argument("--beta1-deg", type=float, default=None)
    parser.add_argument("--beta2-deg", type=float, default=None)
    parser.add_argument("--sheet-z-mm", type=float, default=None)
    parser.add_argument("--dewarp-order", type=int, default=1)
    parser.add_argument("--min-max-filter-length", type=int, default=0,
                         help="0 (default) leaves preprocess.min_max_filter_enabled at this "
                              "app's plain default (False). >0 enables it at that length -- "
                              "this dataset's own real DaVis job (JobHistory.xml, "
                              "imagePreprocessingParameter) used useMinMaxFilter=true, "
                              "minMaxFilterLength=4.")
    parser.add_argument("--sig2noise-threshold", type=float, default=None,
                         help="Override ValidationSettings.per_pass_sig2noise_threshold "
                              "(default 1.0, openpiv's own default) -- lower is more "
                              "permissive (fewer mid-calculation replacements, closer to "
                              "the pre-sig2noise-validation behavior). Omit to use this "
                              "app's plain default.")
    parser.add_argument("--out-dir", default="piv_comparison_output")
    parser.add_argument("--outlier-threshold", type=float, default=3.0)
    args = parser.parse_args()

    if args.max_pairs > 10:
        parser.error("--max-pairs > 10 is almost certainly not what you want for this "
                      "small-scale validation tool -- pass a smaller number.")

    os.makedirs(args.out_dir, exist_ok=True)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from piv_suite_gui.widgets.preview_panel import PreviewPanel
    panel = PreviewPanel()

    (project, preprocess, correlation, validation, post, calibration,
     stereo_settings) = build_stereo_settings_bundle(
        args.set_file, args.multiset_index,
        (args.alpha1_deg, args.alpha2_deg, args.beta1_deg, args.beta2_deg),
        args.sheet_z_mm, args.dewarp_order, args.min_max_filter_length, args.sig2noise_threshold)

    if calibration.frame_dt_s is None:
        raise SystemExit(
            "No frame_dt_s was found for this .set -- this app's own U/V/W would come "
            "back in mm/frame, not a per-second physical unit, and couldn't be compared "
            "meaningfully against DaVis's own m/s output. Real timing is required for "
            "this comparison to mean anything.")

    # world_scale_px_per_mm is this app's own dewarped-WORLD-PIXEL grid
    # scale (px/mm) -- r["x"]/r["y"] from _compute_stereo come straight
    # off that pixel grid (engine.coords on stereo_settings.world_shape),
    # NOT mm, unlike DaVis's own x_dv/y_dv (real mm, via frame.scales).
    # Must convert before handing either to compare()/plot_comparison,
    # which assume both sides share a physical unit -- confirmed via a
    # real 1-pair run: skipping this made n_compared collapse to 704 out
    # of ~250k valid vectors (compare()'s recentering step only finds a
    # tiny overlap when one grid's "mm" extent is actually still in
    # pixels, tens of times larger than the other's real mm extent).
    px_per_mm = stereo_settings.world_scale_px_per_mm
    # Same story for velocity: r["u"]/r["v"]/r["w"] are m/s (confirmed by
    # r["units"] whenever frame_dt_s is set, guaranteed above), but
    # load_vc7_stereo_field always returns mm/s (matching compare_davis_
    # lavision.load_vc7_field's own established convention) -- a raw
    # 1000x mismatch if compared directly. Convert this app's output to
    # mm/s once, up front, rather than threading a "which unit is this"
    # flag through every downstream function.
    vel_to_mm_s = 1000.0

    # native_scale: divide a MM/S field by this to reach native px/frame
    # displacement units, which is what normalized_local_residual's
    # eps=0.1 floor is calibrated for (see field_stats' own docstring).
    # u[px/frame] = u[mm/s] * frame_dt_s[s] * px_per_mm[px/mm], so
    # dividing by native_scale = 1/(frame_dt_s * px_per_mm) reaches that.
    native_scale = 1.0 / (calibration.frame_dt_s * px_per_mm)

    for offset in range(args.max_pairs):
        index = args.start_index + offset
        print(f"\n=== pair index {index} ===")

        r = panel._compute_stereo(project, preprocess, correlation, validation, post,
                                   calibration, stereo_settings, index)
        pair_id = r["pair_id"]
        print(f"  this-app preview: {r['elapsed']:.3f}s  {r['n_valid']}/{r['n_total']} valid  "
              f"units={r['units']}")
        print(f"  reject breakdown: range/residual={r['n_range']}  std_dev={r['n_std']}  "
              f"small_groups={r['n_group']}  "
              f"(out of {r['n_total']} total grid points, {r['n_total'] - r['n_valid']} rejected total)")
        print(f"  fov_overlap={r['n_fov']}/{r['n_total']} ({100*r['n_fov']/r['n_total']:.2f}%)")

        x_app_mm, y_app_mm = r["x"] / px_per_mm, r["y"] / px_per_mm
        u_app_mm_s, v_app_mm_s = r["u"] * vel_to_mm_s, r["v"] * vel_to_mm_s
        w_app_mm_s = r["w"] * vel_to_mm_s

        vc7_path = os.path.join(args.vc7_dir, f"B{index + 1:05d}.vc7")
        if not os.path.exists(vc7_path):
            raise FileNotFoundError(f"no DaVis result at {vc7_path} for set index {index}")
        x_dv, y_dv, u_dv, v_dv, w_dv = load_vc7_stereo_field(vc7_path)

        fields = [("this app (preview)", u_app_mm_s, v_app_mm_s), ("LaVision DaVis", u_dv, v_dv)]
        all_stats, all_resids = [], []
        for label, u, v in fields:
            stats, resid = field_stats(label, u, v, native_scale=native_scale)
            all_stats.append(stats)
            all_resids.append((label, resid))

        print()
        print_stats_table(all_stats)
        print()
        compare(f"  {pair_id} vs DaVis", x_app_mm, y_app_mm, u_app_mm_s, v_app_mm_s, x_dv, y_dv, u_dv, v_dv,
                w_app=w_app_mm_s, w_dv=w_dv)

        png = os.path.join(args.out_dir, f"stereo_{pair_id}_field_comparison.png")
        plot_comparison(png, fields, all_resids, pair_id, args.outlier_threshold)
        print(f"\n  wrote {png}")


if __name__ == "__main__":
    main()
