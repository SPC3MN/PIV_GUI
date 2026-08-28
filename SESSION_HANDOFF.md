# Session handoff — PIV_Suite_testing stereo density/accuracy investigation

Paste this whole document as the first message of a new chat to continue. Repo:
`C:\Users\Germiel\Desktop\PIV_Suite_testing`, branch `perf/cpu-pipeline-overhaul`. All work
described below is **UNCOMMITTED** (14 files changed, 637 insertions / 225 deletions — see `git
diff --stat`). Nothing has been pushed, version-bumped, or released this session.

**Why this doc exists right now:** the real reference dataset lives on an external/network drive
(`J:\Final_Stereo\...`, referenced throughout as "the real Swirl dataset" — never write to it,
read-only reference data) that has been stalling badly on this machine (background scripts hang
at ~0% CPU indefinitely when touching real image data on that drive, even after a full reboot and
a Defender exclusion for `J:\`). The user is moving the drive to a different computer to continue.
**On the new machine, re-map/reconnect the drive as `J:\` (or update paths below) before resuming
real-data work.**

## Where things stand

### 1. DONE and verified: sig2noise/peak-ratio validation re-enabled in the CPU engine
Small, self-contained change, fully tested, not part of the density investigation below (an
earlier, separate ask). `engines/_openpiv_speedups.py`'s fast/chunked correlation path now
computes peak2mean sig2noise cheaply from data it already gathers per window (previously any
non-None `sig2noise_method` fell back to openpiv's slow path entirely). `engines/cpu_engine.py`
wires this into `per_pass_validation`'s real-validation branch (`ValidationSettings.
per_pass_sig2noise_threshold`, default 1.0). Full test suite green.

### 2. DONE and verified: stereo combined-field validation (density-recovery fix)
**Root cause found and fixed.** Stereo mode used to run full validation independently on each
camera's raw 2D field, then require both to pass (`valid1 & valid2`) before triangulating. Per-
camera density was already ~91-93% (comparable to DaVis's own 89%), but two independent
accept/reject decisions compound multiplicatively — confirmed on real data, `p_cam0 * p_cam1 *
p_fov` predicted the actual combined density almost exactly.

**Fix**: new `pipeline.process_stereo_pair()` (in `processing/pipeline.py`) validates the
**triangulated** field once instead. Required extending `postprocess.py`'s `range_filter`,
`normalized_median_residual`, `global_outlier_mask`, `replace_invalid_vectors` with an optional
`w` parameter (backward-compatible, `w=None` preserves existing 2-component behavior for planar/
dual-planar). Local UOD now includes W (U,V-only validation let real garbage through — max
velocity spiked to 1600+ mm/s). A global std-dev check on the combined field is also needed (a
local check can't catch a small, internally-self-consistent cluster of bad vectors) —
`PostProcessSettings.global_outlier_std` default was briefly changed to `None` then **reverted
back to `3.0`** once it was discovered the check is genuinely needed on the *combined* field even
though it was found to contribute 0 rejections on the old *per-camera* field (see schema.py's
docstring for the full story — this is a real, non-obvious finding, don't re-disable it).

Rewired in 4 call sites: `preview_panel.py::_compute_stereo`, `processing/parallel_stereo.py`
(batch worker), `cli/main.py::handle_pair_stereo` (non-tiled branch only — tiled/GPU stereo keeps
the old per-camera approach, since `range_filter` needs a regular grid tiled output doesn't have),
`pipeline_worker.py::_process_set_stereo` (serial branch). `tests/unit/test_parallel_stereo.py`'s
`_serial_reference` helper updated to match. Full test suite: **361 passed**, 3 pre-existing
failures in `test_davis_set_stereo_calibration.py` (see "Known unrelated failures" below).

**Real 3-pair validation (before → after, this fix alone, angle NOT yet corrected — see #3)**:
density 75-78% → 78-81%, outlier rate (local residual > 2.0) 3.5-4.8% → 1.2-1.5% (~65-70%
relative reduction), max velocity back to physically plausible (280-340 mm/s vs DaVis's own
320-380), correlation unchanged. Real, verified win. **Not committed yet.**

### 3. FOUND but NOT YET FIXED IN CODE: camera-angle-derivation bug (big deal, real accuracy win available)
The user provided real DaVis calibration screenshots for this exact project (verified: coefficients
match `Calibration.xml` to 4 decimal places). DaVis's own calibration UI reports **"Min/Max angle
1-2: 89.53°"** — the real, measured angle between the two cameras (user physically set each camera
to 45° from the plate, ~90° apart; DaVis's own bundle-adjustment measurement confirms ~89.53°).

This app's own `io.davis_set._estimate_stereo_angles` (derives `alpha1`/`alpha2` for
`calibration.reconstruction.reconstruct_stereo`'s `dx1 = dX - dZ*tan(alpha1)` model from the
two-Z-plane calibration polynomial's parallax) computes **alpha1=33.33°, alpha2=-35.90°** — an
implied 69.23° camera-to-camera angle. **~20° too small.**

**Empirically confirmed this matters a lot**: overriding with `alpha1=44.765°, alpha2=-44.765°`
(DaVis's measured value split symmetrically) on all 3 real pairs:

| Pair | W mean\|diff\| (old→new) | Density (old→new) |
|---|---|---|---|
| 0 | 62.5 → 14.9 mm/s (-76%) | 78.6% → 80.8% |
| 1 | 44.2 → 14.3 mm/s (-68%) | 79.1% → 81.2% |
| 2 | 52.8 → 16.4 mm/s (-69%) | 77.7% → 80.1% |

Correlation stayed flat throughout (confirms this is a **scale/bias** error, not a shape error —
exactly what a wrong triangulation angle causes; corr(W) is mathematically scale-invariant to a
single global angle change, which is itself a real, useful finding, not a bug in the diagnostic).

**Root cause investigation (this is the part that's genuinely unresolved)**:
- Ruled out: sign convention (already validated correct in an earlier session).
- Ruled out: per-plane coordinate-normalization drift between the two Z-planes' `x0`/`x_span`
  (tested directly with a scratch script — forcing shared normalization barely moves the estimate,
  just adds noise).
- Found and fixed 2 real bugs in the *diagnostic scripts themselves* while investigating (NOT in
  production code): (1) `griddata` doesn't skip NaN source values, corrupting resampled output
  near any invalid cell — always filter `~np.isnan(...)` before calling it, matching
  `compare_davis_lavision.resample_onto`'s already-correct pattern; (2) the two coordinate grids
  (this app's world-pixel grid vs. DaVis's own mm grid) do NOT share an absolute origin —
  `compare_davis_lavision.compare()`'s mandatory re-centering step (`x - (x.min()+x.max())/2`)
  must be replicated in any quick comparison script or you get near-zero, meaningless correlation
  (learned the hard way: corr(W)=0.02 constant across every angle tested until this was added).
- **Likely real cause, not yet proven with certainty**: `_estimate_stereo_angles`'s 2-plane
  finite-difference assumes a *linear* parallax relationship (`shift = tan(angle) * dz`), correct
  only for an orthographic/infinite-standoff camera. Real DaVis calibrations have large
  higher-order polynomial terms (`dx_coefs['s2']`, `['s3']`, etc. — genuine perspective/lens
  effects from a *finite* standoff distance) that this simple model can't correctly invert. Built
  a synthetic pinhole-camera test (real perspective projection → fit a 3rd-order polynomial the
  way DaVis's own calibration would → run the actual derivation formula on it) that reproduces a
  large, real, **standoff-distance-dependent** bias (not a small rounding error) — but the bias's
  sign and magnitude vary a lot with the assumed standoff distance L, which isn't stored anywhere
  in this app's readable calibration files (`CalibrationDialogModel.xml` only has bundle-
  adjustment *config flags*, not the resulting camera pose — checked directly). **A fully general
  closed-form correction isn't derivable from the calibration file alone.**
- Also found: the existing unit test
  (`test_estimate_stereo_angles_recovers_known_alpha_exactly` in
  `tests/unit/test_davis_set_stereo_calibration.py`) is **circular** — it builds synthetic
  calibration data using the exact same linear formula the function later inverts, with zero
  higher-order terms, so it could never have caught this. Worth fixing/extending eventually.
- Was mid-way through a rigorous empirical sweep (compute per-camera correlation once, real
  angle-vs-real-DaVis-ground-truth sweep with proper validation applied) to nail down whether the
  true optimum is exactly symmetric or has real camera-specific asymmetry, when the J: drive
  stalling made this impractical to finish. **Preliminary (unvalidated due to a since-fixed
  recentering bug) sweep suggested the true optimum clusters very tightly around 44.7-46° on both
  sides — i.e. close to symmetric, no strong evidence of meaningful asymmetry — but this was not
  confirmed with the fully-corrected (recentered + properly validated) sweep methodology before
  the drive became unusable.** Re-run this to be sure before finalizing.

**Recommended path forward** (was about to ask the user to confirm before the drive issue
interrupted): add a real "measured camera-to-camera angle" input (CLI flag already exists:
`--alpha1-deg`/`--alpha2-deg` on `scripts/compare_stereo_preview.py`; would need a GUI field too)
that overrides the auto-derived estimate, since deriving a general bias-free formula requires data
(standoff distance) DaVis doesn't persist. **Not implemented in production code yet** — only
tested via CLI overrides on the comparison script.

### 4. FOUND, real but secondary: `sheet_z_mm` sensitivity
DaVis doesn't store the laser sheet's real-world Z position (it's an acquisition-time quantity);
this app requires it whenever a camera has 2 calibrated Z-planes and currently defaults to the
midpoint of the calibrated range as an approximation (`-0.5mm` for this project, real range is
-2.0mm to +1.0mm, i.e. 3mm span). Tested the -2.0mm extreme vs the -0.5mm midpoint (both with the
corrected camera angle): corr(U) dropped 0.916→0.873, corr(W) 0.952→0.914, mean|diff| worsened
28-40% **across all components** (not just W — sheet_z_mm affects the whole camera-mapping
interpolation, not just triangulation). Confirms real sensitivity within the 3mm range; the
midpoint default remains a reasonable choice absent a real measured value, but this is a genuine,
secondary contributor to remaining error. No code changes proposed or needed here yet — just
documented.

## Known unrelated test failures (ignore these, not caused by this session's work)
3 failures in `tests/unit/test_davis_set_stereo_calibration.py`
(`test_read_stereo_calibration_from_real_swirl_set_raises_pinhole_not_exact`,
`test_exact_camera_mapping_matches_real_marks[...Calibration_260713_181401...]`,
`test_files_are_identical_matches_real_duplicated_mark_tables`) — all tied to
`J:\Final_Stereo\Properties\Calibration History\` being empty/incomplete on this machine, an
external data issue, not a code problem. Confirmed present and unchanged across every test run
this entire session.

## Real dataset reference (for when the drive is reconnected)
- Project: `J:\Final_Stereo\Swirl\On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0.set`
- DaVis reference output: `J:\Final_Stereo\Swirl\On Time=6.0_Burst On Time=0.0_Burst Off
  Time=0.0\StereoPIV_MPd(3x32x32_75%ov)\` (`B00001.vc7` etc., 1-based)
- Calibration snapshot in use: `260724_211326` (the project's *current* Calibration.xml — the
  `Properties\Calibration History\` folder is empty/incomplete on this machine, unrelated issue,
  see above)
- Real per-plane facts (cross-checked against the user's own DaVis screenshots): plane 1 z=1mm,
  plane 2 z=-2mm, scale factor 17.9210 px/mm, dewarped image 5874×3067 px / 327.8×171.1mm,
  frame_dt_s=0.0007s, pixel_pitch_mm=0.05580053482032
- Standard comparison invocation:
  ```
  python scripts/compare_stereo_preview.py \
    --set-file "J:\Final_Stereo\Swirl\On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0.set" \
    --vc7-dir "J:\Final_Stereo\Swirl\On Time=6.0_Burst On Time=0.0_Burst Off Time=0.0\StereoPIV_MPd(3x32x32_75%ov)" \
    --start-index 0 --max-pairs 3 --sheet-z-mm -0.5 \
    --alpha1-deg 44.765 --alpha2-deg -44.765 \
    --out-dir <some new dir>
  ```
  (drop the `--alpha1-deg`/`--alpha2-deg` override to see the current, still-buggy auto-derived
  default)

## Immediate next steps, in priority order
1. Re-confirm the corrected-angle-vs-real-DaVis empirical sweep with the fully fixed methodology
   (recentering + real postprocess validation applied) — was interrupted by the drive stall right
   before completion. Scratch script existed at
   `<scratchpad>\exp_angle_sweep.py` this session (session-specific temp dir, likely gone on a new
   machine/session — rebuild from this doc's description if needed: compute per-camera correlation
   once via direct `engine(...)` calls, then cheaply sweep `pipeline.combine_stereo_pair` +
   `postprocess.range_filter(w=...)`/`global_outlier_mask(w=...)` against real DaVis W ground
   truth via `compare_davis_lavision.resample_onto` with proper recentering).
2. Decide with the user how to implement the angle fix in production code (measured-angle
   override input, most likely) and implement it.
3. Run the full 3-pair comparison one more time with BOTH fixes (combined-field validation +
   corrected angle) together for a final, complete before/after report.
4. Run the full test suite once more (`QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m
   pytest tests/unit`) to confirm still green.
5. Ask the user whether to commit this work (nothing committed yet), and separately whether to
   version-bump/release (this session's `pyproject.toml`/`__init__.py`/`piv_suite.iss` still say
   whatever they said before this session — check before assuming).

## Constraints (unchanged, still apply)
Never write to, modify, or delete anything under `J:\Final_Stereo\` or `D:\Truck_PIV_Round4\` —
read-only reference data only, always.
