# Exact stereo calibration and per-pixel triangulation — design

Date: 2026-09-01
Objective: raise computed-vector accuracy and density to match LaVision DaVis, measured
against the real reference project `D:\messy_data\Full_Tank(2026)` (read-only).

## Baseline (measured, not estimated)

| | value |
|---|---|
| Test suite at `8b42c1f` | 395 passed, 25 skipped |
| DaVis density, `dt_opt` B00001 | 93.78% (250209/266816) |
| DaVis density, `Initial_Test\Self2_04` B00001 | 94.45% (249789/264480) |
| **App on `dt_opt` / `Bursting` / current calibration** | **`NotImplementedError` — zero vectors** |
| App on `Self2_04`, app's own estimated angle | 89.79% genuinely measured (233637/260192) |
| App on `Self2_04`, mean\|diff\| U,V vs DaVis | 29.56 mm/s (mean \|velocity\| 86.5) |
| App on `Self2_04`, mean\|diff\| W vs DaVis | 58.36 mm/s |
| App on `Self2_04`, corr(U) / corr(V) / corr(W) | 0.857 / 0.893 / 0.966 |
| App max \|velocity\| vs DaVis's | 259.7 vs 1127.0 mm/s |

The app's reported "97.16% valid" is the camera-overlap FOV count, not measured density:
rejected vectors are interpolated back in by `replace_invalid`. DaVis runs with
`useFillUp=false`, so its 94.45% is genuinely measured. Like-for-like the app is ~4.7
points *below* DaVis while appearing above it.

## Root causes

Ordered by evidence strength. (1) and (2) are blocking/corrupting; (3)-(4) are accuracy.

1. **`PinholeOpenCV` calibration is unsupported.** `io/davis_set.py::read_stereo_calibration_from_set`
   raises `NotImplementedError` for any snapshot that is not `Polynomial3rdOrder`. Six of this
   project's eight snapshots — including every recent one and the currently-active one — are
   `PinholeOpenCV`. The app cannot produce a single vector for `dt_opt` or `Bursting`.

   The prior session concluded pinhole had "no exact decode". **That conclusion is wrong.** The
   model decodes exactly; verified reprojection against `MarkPositionTable.xml` reproduces DaVis's
   own stored `<FitError RMS>` to relative difference 3.6e-14 (cam1) and 6.2e-15 (cam2) on a clean
   snapshot with nothing fitted. Their residual came from `MarkPositionTable.xml` being
   byte-identical in `camera1\` and `camera2\` with *both* `<Camera>` blocks inside each copy, so
   selecting marks by folder paired cam2's marks with cam1's parameters.

   Decoded conventions: `FocalLengthPixel` is **millimetres** (f_px = f / SensorPixelSizeMm);
   `PrincipalPoint` is the projection *and* distortion centre in raw sensor px;
   `OriginPixelPosition` is not a sensor quantity at all but the corrected-image pixel of world
   (0,0), equal to `-OffsetMm/FactorMmPerPixel`, and plays no part in projection; distortion is
   textbook OpenCV Brown-Conrady on normalized camera coords; `R = Rz @ Ry @ Rx`; `Xc = R@Xw + T`.
   No Scheimpflug term is needed. `<FitError RMS>` is the 2-D residual weighted by each mark's `w`.

2. **Self-calibration correction fields are silently ignored.** Both `Polynomial3rdOrder`
   snapshots (`Calibration_260309_161814`, `Calibration_260310_140344`) contain a
   `Correction field\` folder: the stored polynomial is only a base layer. The app decodes the
   base layer alone and raises nothing, giving a mapping measured 16.1 / 32.0 px wrong. This
   silently corrupts every result on `Initial_Test\*` — including this document's own Self2_04
   baseline, which is why that baseline's U/V numbers must not be read as engine quality.
   `Calibration_260310_140344` also declares an implausible `FitError RMS` of 0.0035/0.0021 px,
   which is a usable staleness signal.

3. **Triangulation uses one global viewing angle per camera.** `reconstruct_stereo` solves
   `dx1 = dX - dZ*tan(alpha1)` with scalar angles. The true per-pixel angle varies **8.4°
   (cam1) / 8.8° (cam2)** across this FOV. The two cameras' errors largely cancel in the
   *difference*, so W is barely affected (~0.6%), but the in-plane components are not:
   **U error ~13% of |W| at the FOV edges (~0.051 m/s, ~20% of |U|max)** and
   **V error ~7% of |W| at top/bottom (~0.028 m/s), entirely from beta**, whose near-zero mean
   (-0.07°) hides a ±4.3° variation. `reconstruct_stereo` already broadcasts array angles
   correctly, so the kernel itself needs no change.

4. **`_estimate_stereo_angles` is wrong by a known factor.** It returns -34.52°/+34.97°
   (69.5° camera-to-camera) where the truth is ~-44.1°/+45.0° (~90.4°; DaVis's own UI reports
   89.53°). Root cause: it measures parallax **on the raw sensor**, foreshortened by cos(alpha),
   and never inverts the mapping — `tan(34.53°)/tan(44.08°) = 0.710 ≈ cos(44.1°)`. Because it is
   untrustworthy, `alpha1_deg`/`alpha2_deg` are currently `None` and every processing entry point
   *refuses to run* until the user hand-enters a measured angle.

5. **Postprocess defaults were tuned around the above defects.** `global_outlier_std = 3.0` runs a
   field-wide 3-sigma cut that DaVis does not do at all (`useAllowedVectorRange=false`). It was
   added to suppress wild velocities produced by the wrong triangulation angle. With mean
   |velocity| 82.6 mm/s it clamps near 260 mm/s, while DaVis's own real max on the same pair is
   1127 mm/s (~13 px, matching its `maxExpectedDisplacement=12`). It is destroying real dynamic
   range to hide a symptom whose cause is being fixed here.

## Design

### A. A camera-model abstraction with two exact implementations

`CameraMapping` already has exactly the interface the rest of the app consumes —
`world_to_raw(xp, yp)`, `dewarp_image(raw, world_shape, order)`, `raw_domain_valid(x, y)` —
where `xp, yp` are *corrected-canvas pixel indices*. The pinhole model fits it unchanged: canvas
pixel -> world mm via `OffsetMm + xp*FactorMmPerPixel`, then `project()` -> raw px.

New `calibration/pinhole.py` provides `PinholeCameraMapping` implementing that same interface,
plus the parameter parsing. Nothing downstream of the interface changes.

### B. Per-pixel triangulation angles

Both models gain `view_angles(xp, yp) -> (alpha_deg, beta_deg)`:
- Pinhole: exact, from the camera centre `C = -R.T @ T`.
- Polynomial: by correctly inverting the two-Z-plane mapping (not raw-sensor differencing).

A new `stereo_view_angles(cam0, cam1, x, y)` helper sits beside the existing `stereo_fov_valid`
and is called at the same four processing entry points, on the same correlation grid `x, y`.
Per-pixel angle *arrays* are then passed to `combine_stereo_pair`.

This is deliberately **not** stored in `StereoSettings`: the fields are correlation-grid-shaped
and derived, so carrying the camera models and deriving on demand keeps them from going stale and
avoids serializing large arrays. `alpha1_deg`/`alpha2_deg` remain as an optional manual
*override* for a rig whose calibration cannot supply them.

### C. Delete the broken estimator and the manual-angle requirement

`_estimate_stereo_angles` is removed rather than repaired: it exists only to approximate a
quantity now derived exactly, and its unit test is circular (it builds synthetic data with the
same linear formula the function inverts, so it could never have caught the defect). The
"Angles measured" GUI gate is removed with it — the angles now come from the calibration.

### D. Refuse to silently mis-dewarp

`read_stereo_calibration_from_set` gains a correction-field / staleness check: a snapshot with a
`Correction field\` folder, or whose reprojection RMS disagrees with its declared `<FitError RMS>`,
is reported rather than silently used. Silent 16-32 px error is worse than a clear refusal.

### E. Re-derive postprocess defaults against DaVis's documented parameters

Once A-D land, re-measure on clean pinhole data and set `global_outlier_std`, the range-filter
threshold and the correlation method from evidence rather than from compensating for A-D.
DaVis's own documented chain is: no scalar-field thresholds at all, one median UOD (removal
factor 2, insertion factor 3, 3x3, min 3 neighbours), remove-groups < 5, no fill-up.

## Testing

- Unit: pinhole parse/project against real `MarkPositionTable.xml`, asserting agreement with the
  snapshot's own declared `<FitError RMS>` — real ground truth, not a self-consistent synthetic.
- Unit: `view_angles` agreement between the pinhole and polynomial models on the same rig.
- Unit: correction-field/staleness detection.
- Replace the circular `test_estimate_stereo_angles_recovers_known_alpha_exactly`.
- End-to-end: app vs DaVis on `dt_opt` and `Bursting` pairs, reporting density and per-component
  error, versus the baseline table above.

## Explicitly out of scope

Reading and applying the DaVis self-calibration correction field itself. It would make
`Initial_Test\*` usable again, but the clean pinhole snapshots cover the datasets that matter and
detection is enough to stop silent corruption.
