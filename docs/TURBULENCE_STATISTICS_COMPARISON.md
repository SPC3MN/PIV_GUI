# Turbulence statistics: this app vs LaVision DaVis

How the two programs' vector fields compare *after* turbulence post-processing —
Reynolds decomposition, RMS, TKE, Reynolds stresses, structure functions,
autocorrelations, dissipation rate, integral length and Lumley anisotropy
invariants — rather than vector-by-vector. Companion to
[DAVIS_COMPARISON.md](DAVIS_COMPARISON.md), which covers per-vector accuracy,
density and throughput.

The question this answers: *if you ran a real turbulence study on this app's
vectors instead of DaVis's, would you reach different conclusions?*

**Not a changelog.** This describes current measured behaviour, not what
changed to get here.

## Method

Both programs' finished fields for the same two swirl recordings — 1000
snapshots each — were pushed through one identical post-processing pipeline
([SPC3MN/PIV-PostProcessing](https://github.com/SPC3MN/PIV-PostProcessing),
`main` at `f0b287c`). Four cases in total: this app and DaVis, planar and
stereo. Because the statistics code is byte-identical across all four, every
difference reported here comes from the upstream vector fields.

**One shared grid, no interpolation.** Both programs lay vectors on an
8-raw-pixel grid derived from the same images, so the two grids have an
identical spacing — 0.41191 mm planar, 0.44640 mm stereo — and differ only by
an integer offset. That offset was found by brute-force correlation over 25
snapshots rather than assumed: planar (3 rows, 2 cols); stereo (1, 1). Both
fields were then cropped to the same physical points. Resampling one side onto
the other would have smoothed the fluctuating field and biased RMS, structure
functions and dissipation downward — corrupting precisely the quantities under
comparison.

The edge trim follows PIV-PostProcessing's own convention (7 points/edge
planar, 8 stereo, ~6.3% of FOV area, dropping the outermost interrogation
windows), applied in DaVis's indexing so both sides land on that project's
canonical grids: **365 x 500** planar, **368 x 719** stereo. Coordinates come
from DaVis's calibration for both, since this app's own x/y are raw pixels with
no absolute origin, and the swirl axis x = 0 only exists in DaVis's frame.

**Two corrections were required and verified.** This app's V component is
sign-inverted relative to DaVis's (its y runs downward in image space; DaVis's
calibrated y runs upward) — measured as `cmp_v_sign_flipped` on all 1000 pairs
of both datasets. Left uncorrected it flips `V_mean` and the `uv`/`vw` Reynolds
stresses. After negation, corr(V) is +0.986 with matching means. Velocities are
carried in m/s, the unit both producers are natively in and the one the
analysis stage assumes.

Both sides are **measured-only**: each masked by its own validity (this app's
`valid` array, DaVis's `ENABLED` mask). DaVis's job ran `useFillUp=false`, so
its output is unfilled; masking this app's to match keeps the comparison
like-for-like rather than scoring interpolated fill against real measurements.
The decomposition is NaN-aware throughout. Mean vector density over the trimmed
window: planar 99.28% (app) vs 100.00% (DaVis); stereo 93.85% vs 94.14%.

## Mean and fluctuating fields

Spatial means over the domain. `corr` is the correlation of the two ensemble
maps; `|d|/CI` expresses the difference as a fraction of the mean bootstrap
confidence half-width each side already carries over 1000 snapshots.

### Planar

| quantity | this app | DaVis | diff | diff % | corr | \|d\|/CI |
|---|---|---|---|---|---|---|
| U_mean (cm/s) | -0.3342 | -0.3362 | +0.0020 | +0.59% | 0.9729 | 0.00 |
| V_mean (cm/s) | 0.0563 | 0.0590 | -0.0027 | -4.62% | 0.9954 | 0.01 |
| U_rms (cm/s) | 10.0258 | 10.1188 | -0.0930 | -0.92% | 0.9577 | 0.24 |
| V_rms (cm/s) | 7.2417 | 7.3460 | -0.1043 | -1.42% | 0.9129 | 0.37 |
| TKE (cm2/s2) | 126.860 | 129.529 | -2.668 | -2.06% | 0.9377 | — |
| isotropy U/V | 1.3844 | 1.3774 | +0.0070 | +0.51% | — | — |
| TKE spatial dev | 4.90% | 5.69% | — | -14.0% | — | — |

### Stereo

| quantity | this app | DaVis | diff | diff % | corr | \|d\|/CI |
|---|---|---|---|---|---|---|
| U_mean (cm/s) | 0.3852 | 0.3993 | -0.0142 | -3.55% | 0.9665 | 0.03 |
| V_mean (cm/s) | 0.1380 | 0.1390 | -0.0010 | -0.71% | 0.9841 | 0.00 |
| W_mean (cm/s) | -1.0540 | -1.0542 | +0.0002 | +0.02% | 0.9885 | 0.00 |
| U_rms (cm/s) | 10.2378 | 10.3315 | -0.0937 | -0.91% | 0.9347 | 0.22 |
| V_rms (cm/s) | 6.9851 | 7.0852 | -0.1000 | -1.41% | 0.9505 | 0.35 |
| W_rms (cm/s) | 9.3663 | 9.4990 | -0.1326 | -1.40% | 0.9743 | 0.34 |
| TKE (cm2/s2) | 121.010 | 123.990 | -2.980 | -2.40% | 0.8725 | — |
| isotropy U/V | 1.4657 | 1.4582 | +0.0075 | +0.51% | — | — |
| isotropy U/W | 1.0930 | 1.0876 | +0.0054 | +0.50% | — | — |
| TKE spatial dev | 5.14% | 5.29% | — | -2.7% | — | — |

Every RMS difference is well inside the per-point bootstrap band (|d|/CI <=
0.37), but the *sign* is the same on all five components across both datasets —
so it is a small systematic offset, not sampling scatter.

### Reynolds stresses (stereo)

The cross-stresses sit near zero, so a percentage difference divides by a
near-zero denominator and exaggerates. Against DaVis's TKE of 124.0 cm2/s2:

| quantity | this app | DaVis | corr | difference |
|---|---|---|---|---|
| uv | -0.8182% of TKE | -0.7393% | 0.9574 | 0.079 pts |
| uw | +0.1930% of TKE | +0.1072% | 0.9747 | 0.086 pts |
| vw | -0.9319% of TKE | -0.9682% | 0.9357 | 0.036 pts |

All three agree to under a tenth of a percentage point of TKE, with map
correlations of 0.94-0.97.

### Lumley anisotropy invariants (stereo)

| quantity | this app | DaVis | diff % | corr |
|---|---|---|---|---|
| eta^2 | 0.03324 | 0.03315 | +0.27% | 0.9502 |
| xi | -0.04834 | -0.04898 | +1.33% | 0.9649 |

The turbulence anisotropy state — where the flow sits on the Lumley triangle —
is effectively identical.

## Structure functions and the noise floor

This is where the two pipelines genuinely differ, and it is the most
informative result here.

D(r) as a ratio of this app's to DaVis's:

| r (mm) | planar D11 | planar D33 | stereo D11 | stereo D33 |
|---|---|---|---|---|
| ~0.42 (1 lag) | 0.796 | 0.797 | 0.651 | 0.542 |
| ~0.85 (2 lags) | 0.756 | 0.754 | 0.720 | 0.637 |
| ~2.1 (5 lags) | 0.861 | 0.857 | 0.859 | 0.791 |
| 10 | 0.961 | 0.954 | 0.968 | 0.910 |
| 20 | 0.969 | 0.965 | 0.970 | 0.936 |
| 40 | 0.970 | 0.967 | 0.970 | 0.950 |
| 80 | 0.970 | 0.964 | 0.969 | 0.955 |

The shape is the signature of a **lower random-noise floor**. For independent
per-vector noise of variance sigma^2, D(r) = D_turbulence(r) + 2*sigma^2. As
r goes to 0 the turbulent term vanishes, so the intercept isolates the noise.
Fitting D = D0 + a*r^2 over the first four lags (D_turbulence ~ r^2 in the
dissipation range) gives:

| implied noise sigma | this app | DaVis | ratio | noise **variance** reduction |
|---|---|---|---|---|
| planar, U | 0.6043 cm/s | 0.7184 cm/s | 0.841 | 29% |
| planar, V | 0.6137 cm/s | 0.7282 cm/s | 0.843 | 29% |
| stereo, U | 0.5396 cm/s | 0.7023 cm/s | 0.768 | 41% |
| stereo, V | 0.4132 cm/s | 0.5917 cm/s | 0.698 | 51% |

**This app's vectors carry 29% less random noise variance than DaVis's on the
planar dataset and 41-51% less on the stereo one.** The corroborating evidence
is the TKE spatial deviation above: this app's planar TKE map is 14% smoother.

Read the **ratio**, not the absolute sigma. Those four lags span 0.25 W to
1.0 W, which is inside the region where the noise is still partly correlated
between overlapping windows (see the next section), so the noise term has not
yet reached its full 2*sigma^2 and the fitted intercept is a lower bound on the
true per-vector noise. The ratio survives that: both programs ran the identical
window size and overlap, so both are biased by the same factor and it divides
out.

### But the noise floor does not explain the RMS gap

The obvious hypothesis — that this app reports lower RMS *because* it carries
less noise (measured variance = true variance + sigma^2) — was tested by
removing each side's own noise variance, and it does not hold:

| quantity | corrected app | corrected DaVis | corrected diff | raw diff |
|---|---|---|---|---|
| planar U_rms | 10.0076 | 10.0933 | -0.85% | -0.92% |
| planar V_rms | 7.2157 | 7.3098 | -1.29% | -1.42% |
| stereo U_rms | 10.2236 | 10.3076 | -0.81% | -0.91% |
| stereo V_rms | 6.9729 | 7.0604 | -1.24% | -1.41% |

The noise-floor difference accounts for only about 8% of the RMS gap. The rest
is real: at large separations, where noise is negligible, this app resolves a
consistent **~3% less** velocity-difference energy (the D ratio flattens at
0.970 planar, 0.969/0.955 stereo). A 3% deficit in variance corresponds to
about -1.5% in RMS, which is the order of what is measured.

So the two effects are separate and point in opposite directions on quality:
this app is measurably **less noisy at small scales** and measurably **slightly
more attenuated at large scales**, consistent with marginally stronger
effective spatial smoothing in its correlation/deformation path.

## The shoulder at r = W, and what it does and does not contaminate

Both programs' D(r) carries a distinct shoulder: the local log-slope falls from
1.37 at the smallest lag to a minimum of 0.735, then **recovers** to a plateau
of 0.877. Compensated as D/r^0.85 it is a visible bump peaking at exactly
r = W. It appears in both programs because both ran the same 3x32x32 at 75%
overlap schedule.

**It is an artefact of the interrogation scale, not a feature of the flow.**
Four independent pieces of evidence:

| evidence | result |
|---|---|
| Position across all four production cases | slope minimum at exactly 40 raw px = 1.25 W in every one, i.e. 2.060 mm planar but 2.232 mm stereo — a fixed *pixel* count, not a fixed length |
| Non-monotonic shape | a real dissipation-to-inertial transition falls monotonically from 2 toward 2/3; it never dips below the eventual inertial slope and returns |
| Reprocessing the same images at W = 64 px | the shoulder disappears entirely — no local minimum anywhere in 40 lags, and at the physical separation where the baseline dips the 64 px run reads a smooth 1.68/1.57 |
| Depth vs noise | ranks monotonically with each case's measured noise: app 0.739 / DaVis 0.570 planar, app 0.885 / DaVis 0.723 stereo. Excess above the plateau is 5% for this app and 24% for DaVis |

**Mechanism.** At 75% overlap, adjacent windows share 75% of their particles,
so their random errors are *correlated*. The noise contribution to a structure
function is 2*sigma^2*(1 - rho_noise(r)), which ramps up as the windows slide
apart and **saturates once they stop overlapping, at r = W**. Below W that ramp
adds slope on top of the turbulence; at W it stops contributing; beyond it only
turbulence remains. The kink is the saturation point.

Window size and vector spacing are locked together at a fixed overlap
(s = W(1 - overlap), so W = 4s at 75%), so separating them takes a run at a
different overlap:

| config | W | spacing | shoulder | /W | /spacing |
|---|---|---|---|---|---|
| 32 px @ 75% (production) | 32 px | 8 px | 40 px | 1.25 | 5.0 |
| 32 px @ 50% | 32 px | 16 px | 48 px | 1.50 | 3.0 |
| 64 px @ 75% | 64 px | 16 px | none | — | — |

Doubling the spacing at a fixed window leaves the shoulder where it was (40 to
48 px is one lag on that run's coarser grid); doubling the window removes it.
**The shoulder is set by the interrogation window.**

### Does it contaminate the analysis-section statistics?

Mostly no, and this is testable rather than arguable: changing the window from
32 to 64 px takes the shoulder from deep to absent, so any statistic carrying
it has to move. Running the full analysis chain on each configuration:

| statistic | W=32 s=8 | W=64 s=16 | W=32 s=16 | spread | verdict |
|---|---|---|---|---|---|
| eps_11 (cm2/s3) | 42.39 | 40.81 | 41.73 | 3.8% | **not contaminated** |
| L_11 (cm) | 9.851 | 9.812 | 9.832 | 0.4% | **not contaminated** |
| lambda_1 (cm) | 0.169 | 0.339 | 0.336 | 60.2% | **not a measurement** |

- **eps_11, eps_33, L_11, L_33 are clear of it.** `Planar_Analysis.py`'s
  inertial-range mask lands at r = 55-88 mm, which is 33-53 W — some 30 times
  further out than the shoulder. `fit_integral_length` is handed that same
  window, so L is fitted there too. Both are essentially unchanged by a
  doubling of the window.
- **lambda_1 and lambda_3 are not measurements of anything.** They are computed
  from lags 1 and 2 only, i.e. entirely inside r < W, and `taylor_microscale`
  fits a degree-2 polynomial through **two points**, which is rank-deficient.
  The result is simply 4.1 x the grid spacing in every configuration tested
  (lambda/spacing = 4.11, 4.11, 4.07, 4.09 across windows of 32, 64 and 32 px),
  so it tracks the grid rather than the flow or even the window. Neither
  lambda nor any Re_lambda derived from it should be reported.
- **RMS, TKE, isotropy and the Lumley invariants are single-point statistics**
  and cannot inherit a two-point artefact. They are inflated by the same
  underlying noise that produces the shoulder, but only slightly:
  sigma^2/u_rms^2 is about 0.36%, i.e. roughly 0.2% on RMS.

### Removing it, if you want to

The shoulder sits at r = 1.25-1.5 W regardless of spacing, so to clear it from
a given analysis range at a fixed vector spacing, raise W and raise the overlap
to compensate: s = W(1 - overlap), so W = 64 px at 87.5% overlap keeps the
present 8 px grid. It works because the bump's height is set by the
noise-to-signal ratio at r = W, and a larger window both holds more particles
(lower sigma) and sits where there is more turbulent energy.

**It is not a free win.** The 64 px run resolves genuinely less turbulence: its
compensated plateau is 2.48 against 2.65 for 32 px, and its dissipation
estimate over 8-16 mm is 17% lower. The trade is a visible artefact for
invisible attenuation of real structure below W. The shoulder is honest signage
that the data below W is unreliable; a larger window removes the sign rather
than the problem. Restricting any fit to r > 2W costs nothing and achieves the
same thing.

## Autocorrelation, integral length and dissipation

The autocorrelations are nearly indistinguishable — every rho(r) difference is
<= 0.008 in magnitude, and the sign flips with separation (this app marginally
lower below ~20 mm, marginally higher above), so there is no systematic offset.
Neither program's rho_33 crosses zero inside the window; rho_11 crosses only in
the stereo dataset, at 308 mm for both.

The autocorrelation carries the same window shoulder, necessarily: for a
homogeneous field 1 - rho(r) is proportional to D(r), so it is the same
statistic rescaled. Measured on the compensated form, the shoulder falls at
**exactly** the same lags as in D — 40 px for the production runs of both
programs, 48 px at 50% overlap, absent at W = 64 px. Its practical consequence
is confined to quantities read from the curvature of rho near the origin, which
is the Taylor microscale and nothing else in this analysis; the integral length
is fitted far out and is unaffected.

Derived scalars from the analysis stage (`eps` in cm2/s3, `L` and `lambda` in
cm):

| quantity | planar app | planar DaVis | diff % | stereo app | stereo DaVis | diff % |
|---|---|---|---|---|---|---|
| eps_11 | 40.698 | 42.457 | -4.14% | 38.468 | 40.313 | -4.57% |
| eps_33 | 31.478 | 33.081 | -4.85% | 25.950 | 27.989 | -7.29% |
| L_11 | 9.2522 | 9.0620 | +2.10% | 10.2330 | 10.0292 | +2.03% |
| L_33 | 5.2132 | 5.1638 | +0.96% | 5.6129 | 5.4909 | +2.22% |
| lambda_1 | 0.1692 | 0.1687 | +0.30% | 0.1842 | 0.1843 | -0.05% |
| lambda_3 | 0.1674 | 0.1663 | +0.63% | 0.1840 | 0.1840 | -0.01% |
| M1 (homogeneity) | 0.0462 | 0.0475 | -2.76% | — | — | — |

The dissipation offset is exactly what the structure functions predict, which
is a useful internal consistency check: eps goes as D^(3/2), so a D ratio of
0.970 implies an eps ratio of 0.970^1.5 = 0.955, i.e. -4.5%. Measured: -4.14%
and -4.57% for eps_11. The dissipation difference is therefore not independent
evidence — it is the same ~3% structure-function offset re-expressed.

**These eps values are not inertial-range dissipation rates.** Compensating by
the inertial scaling never produces a plateau on this data: eps(r) climbs
monotonically, +35% from the 8-16 mm band to the 32-64 mm band, because the
measured log-slope is 0.82-0.97 and never 2/3. `Planar_Analysis.py` locates its
"inertial range" by looking for slope in [0.6, 0.73], and on this flow that
condition is first met at r = 55-88 mm — comparable to the integral length
L_11 = 92 mm — where the slope is passing through 2/3 on its way down to
large-scale saturation. So the number is measured in the energy-containing
range. It is reproducible and window-robust (40.8 to 42.5 across window sizes
of 32 and 64 px), and the app-vs-DaVis comparison above is sound because both
sides were measured identically, but the absolute value should not be quoted as
an inertial-range eps.

## What this means in practice

- **Flow-field conclusions are unchanged.** Mean velocities, isotropy ratios,
  anisotropy invariants, autocorrelation shape and integral length scales agree
  to 0.3-2%. A study run on this app's vectors would report the same flow.
- **Turbulence intensity comes out ~1% lower, TKE ~2% lower, dissipation ~4-7%
  lower.** These are systematic, not scatter. If absolute dissipation is the
  headline number, that offset matters and should be quoted.
- **Small-scale noise is materially better on this app** — 29% less noise
  variance planar, 41-51% stereo. For anything resolving near the grid scale,
  that is the more consequential difference of the two.
- **The reported dissipation and integral lengths are safe from the r = W
  shoulder**, verified by doubling the window; the Taylor microscale is not,
  and should not be used at all.
- Which program is closer to truth on the ~3% large-scale attenuation cannot be
  settled from these two datasets alone; it needs an independent reference
  (a synthetic image set with known displacement would do it).

## Caveats

- **Energy spectra were deliberately not compared.** PIV-PostProcessing's
  `spatial_spectrum_1d` drops any row containing a NaN, unlike its
  autocorrelation and structure-function estimators, which normalise by a
  per-lag valid count. On these real gappy fields it keeps 100% of DaVis's
  planar rows but only 53% of this app's, and 0% of either stereo field — so a
  spectrum comparison would contrast two different row subsets rather than two
  pipelines. (The repo's own stereo path computes no spectrum either.)
- **The Taylor microscale is not a measurement here.** lambda lands at 4.1 grid
  spacings in every case and every window configuration tested — 4.11, 4.11,
  4.07 and 4.09 across windows of 32, 64 and 32 px. PIV-PostProcessing's own
  HANDOFF.md records the pinning; the cause is that `taylor_microscale` fits a
  degree-2 polynomial through two points, which is rank-deficient, evaluated at
  lags 1-2 where the field is window-dominated anyway. It therefore returns a
  fixed multiple of the grid spacing and carries no flow information. That the
  two programs agree on it to 0.6% reflects the shared grid, not shared
  physics. No Re_lambda should be derived from it.
- **The dissipation rates are energy-containing-range values, not
  inertial-range ones** — there is no r^(2/3) plateau on this flow. See the
  dissipation section; they remain valid for comparing the two programs, which
  were measured identically.
- **The shoulder at r = W does not reach eps or L.** Both are fitted at
  33-53 W, and both move by under 4% when the window is doubled. Only the
  Taylor microscale is computed inside the affected region.
- The stereo row alignment is ambiguous between offsets of 1 and 2 grid points
  (correlation scores differ by 3e-5) — a genuine half-grid-point, 0.22 mm
  registration ambiguity. It shifts the spatial maps slightly but not the
  domain-aggregate statistics reported here.
- Difference maps show the largest deviations at the left and right FOV edges,
  where this app's density is lowest; the interior is flat.
- One recording each. These are single-case results on a swirl flow, not a
  characterisation across flow regimes.

## Reproducing

Four stages, ~25 minutes total on a 48-thread machine:

1. Export both programs' fields onto the shared grid as `snap_*.npz`
   (m/s, measured-only, V sign corrected) — 2000 files, ~4 min.
2. Run the decomposition over all four cases — 12.8 min
   (planar 1.6 min/case, stereo 4.6 min/case).
3. Run PIV-PostProcessing's analysis stage once per mode, into separate
   workbooks. Planar and stereo need separate roots: `discover_case_dirs`
   treats every subfolder holding an `Ensemble_Averages/Averages.npz` as a
   case, so a shared root would feed stereo cases into the planar analysis.
   `Stereo_Analysis.py` has no `plotting` flag and calls `plt.show()`
   unconditionally, so it needs `MPLBACKEND=Agg` to run headless.
4. Compare and plot.

Outputs land in `PIV_Compare_PROC/`: per-case `Ensemble_Averages/` and
`Lumley_Statistics/`, the two `Analysis_Results/*.xlsx` workbooks, `Figures/`
(structure-function and autocorrelation curves plus ensemble field maps per
mode, the shoulder and shape comparisons, and the dissipation plateau test),
and `comparison_report.txt` with the full numeric tables.

The window-size attribution needs two extra runs of this app over the same
planar images — 150 pairs is ample, the estimator reproduces the 1000-snapshot
result to three decimals — changing only the three fine passes:

| run | passes | spacing | cost |
|---|---|---|---|
| 64 px @ 75% | `[64@0.5] + [64@0.75] x 3` | 16 px | ~28 min |
| 32 px @ 50% | `[64@0.5] + [32@0.5] x 3` | 16 px | ~7 min |

Everything else — preprocessing, validation, post-processing, calibration —
must stay at the production configuration, or the comparison confounds the
window with something else.
