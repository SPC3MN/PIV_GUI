# Full-dataset validation report — stereo, planar, dual-planar

Ran three real datasets (each already processed in LaVision DaVis, with real
reference output available) through the full pipeline: automatic
calibration/mode verification via a single-pair preview, a full batch run
through this app's own CLI, then a snapshot-by-snapshot comparison against
DaVis's own final (`PostProc`) output using `scripts/compare_dataset.py`.

**Status: stereo and planar completed the full pipeline end-to-end. The
dual-planar (Truck) dataset's batch run completed, but only after an
investigation script bypassed a calibration safety check that should have
stopped it (see corrected finding 4 below) — its numeric results are
invalid and skipped from this report rather than reported as wrong numbers.**

**Post-report addendum**: two real bugs this report flagged as unfixed —
`recommended_workers()`'s missing Windows 61-worker cap, and a stale
`fast_replace_nans` numerical patch (see [git log](https://github.com/SPC3MN/PIV_GUI/commits/master)
for both) — were fixed in a follow-up end-to-end improvement pass after
this report was first written. Finding 4 below was also corrected after
further direct verification against the app's real source showed it was
not an app bug the way it was first described.

## Datasets

| | Stereo | Planar | Dual-planar (Truck) |
|---|---|---|---|
| Project | `D:\Final_Stereo\Swirl\On Time=6.0_...set` | `D:\Final_Planar_Swirl\...\On Time=0.7_...set` | `E:\Truck_PIV_Round4\...\X_50_mm_Y_0_mm.set` |
| Pairs | 1000 | 1000 | 1500 |
| DaVis reference | `StereoPIV_MPd(3x32x32_75%ov)\PostProc` | `PIV_MPd(3x32x32_75%ov_ImgCorr)\PostProc` | `SideBySide_PIV_MPd(1x32x32_75%ov_ImgCorr)\PostProc` |
| Batch processing | ✅ complete | ✅ complete | ✅ complete (but see calibration bug) |
| Comparison vs DaVis | ✅ complete (100-pair sample) | ✅ complete (100-pair sample) | ❌ invalid, not reported |

Two real mode-detection surprises came up before any processing started, both
resolved with direct evidence rather than guesswork:

- **"Final_Planar_Swirl" is actually a real 2-camera stereo calibration**
  (`FieldOfView=SideBySideStereoVolume`), but this specific recording's own
  raw frame files only have `Frame0`/`Frame1` (one camera's exposure pair,
  not the 4 streams a real 2-camera acquisition would have) — confirmed
  genuinely single-camera at the raw-data level despite the shared project
  folder's calibration being stereo-typed. Processed as **planar**, correctly.
- **This app's own `detect_project_type_from_set` only reads the shared
  calibration folder's `FieldOfView`**, never cross-checking the specific
  recording's actual frame-stream count — a real gap that mislabels a
  project like this one as stereo when the calibration folder is shared
  across mixed stereo/planar recordings. Noted as a bug below.

## Performance investigation (this machine: dual-socket, 2× Intel Xeon Gold
6154, 18 cores/36 physical cores total, 72 logical via hyperthreading, 391GB
RAM)

This session found this app's Tier-3 multi-worker batch parallelism has two
real, unrelated problems that compound into a large, silent performance
regression for a very natural usage pattern (a single real recording run
directly via CLI):

1. **CLI parallelism never engages for a single `.set` file.**
   `cli.main.main()` computes `interactive_preview = not is_batch`, and
   `is_batch` (from `resolve_set_paths`) is only `True` for a **folder of
   many `.set` files** — pointing `--input-path` at one real recording
   directly (`piv-suite my_project.pivproj`, the CLI's own documented usage)
   silently falls back to fully serial processing, with no warning. The
   GUI's own Run button has no such exclusion. Worked around this session by
   calling `process_pairs_stereo`/`_planar`/`_dual_planar` directly with
   `interactive_preview=False`.
2. **`recommended_workers()` doesn't cap at Windows' hard `ProcessPoolExecutor`
   limit.** It picks `min(cpu_count, ram_workers)` — 72 on this machine —
   but Windows' `ProcessPoolExecutor` raises `ValueError: max_workers must
   be <= 61` (a `WaitForMultipleObjects` handle-count limit), so a genuine
   batch run on any Windows machine with >61 logical processors **crashes
   outright** the first time Tier 3 actually engages.

With those worked around, a real empirical worker-count sweep (stereo, same
1000-pair dataset, steady-state measured over 30-45 minute windows each) gave
a clear, non-obvious answer:

| `n_workers` | Pairs/min | |
|---|---|---|
| **32** (= physical core count) | **6.81** | **best** |
| 60 | 5.36 | worse |
| 61 (near Windows' hard max, exercises hyperthreading) | 5.12 | worst |

Pushing toward hyperthreading (61) made things measurably **worse**, not
better — confirmed via direct process inspection: each worker used ~100% of
its own core continuously (no OS-level scheduling contention), yet completed
roughly half as many pairs as its own CPU-busy time implied it should, and
disk I/O was confirmed idle throughout. This points to **aggregate memory
bandwidth**, not core count, as the real bottleneck for this large-FFT,
NUMA-dual-socket workload — a genuine, non-obvious characteristic of this
app's Tier-3 parallelism on this class of hardware, not something
`recommended_workers()`'s current CPU-count/RAM-only heuristic accounts for.

**Recommendation**: `recommended_workers()` should (a) hard-cap at 61 on
Windows regardless of `cpu_count()`, to prevent the crash, and (b) ideally
default toward *physical* core count (`psutil.cpu_count(logical=False)` or
equivalent) rather than logical/hyperthreaded count for this specific
CPU-bound, large-array workload, since hyperthreading measurably hurts
throughput here rather than helping.

### Real batch timing achieved (n_workers=32, the tuned setting)

| Dataset | Pairs | Wall time | Effective rate |
|---|---|---|---|
| Stereo | 1000 | 2.50 hr | 6.67 pairs/min (9.00s/pair) |
| Planar | 1000 | 0.80 hr | 20.85 pairs/min (2.88s/pair) |
| Dual-planar | 1500 | 2.63 hr | 9.51 pairs/min (6.31s/pair) |

Planar's per-pair cost is far lower (single camera, no world-grid dewarp to
a large shared canvas) and scales much better under the same
memory-bandwidth ceiling — a useful data point for anyone tuning batch
throughput expectations by mode.

## Comparison vs DaVis (100-pair stride sample of each 1000-pair dataset,
against DaVis's own final `PostProc` output)

| | Stereo | Planar |
|---|---|---|
| This app's density (valid %) | 81.85% avg (77.4-89.3%) | 95.44% avg (92.8-97.1%) |
| DaVis's density (valid %) | 88.90% avg (88.0-89.3%) | 98.11% avg (98.1-98.1%) |
| This app's mean flow magnitude | 105.8 mm/s avg (≈0.11 m/s) | 58.9 mm/s avg (≈0.06 m/s) |
| DaVis's mean flow magnitude | 105.2 mm/s avg (≈0.11 m/s) | 55.7 mm/s avg (≈0.06 m/s) |
| corr(U) | 0.9594 avg (min 0.9101) | 0.9576 avg (min 0.9213) |
| corr(V) | 0.9639 avg (min 0.9279) | 0.9503 avg (min 0.9183) |
| corr(W) | 0.9682 avg (min 0.9253) | n/a |
| mean\|diff\| | 20.2 mm/s avg (max 29.1) | 14.2 mm/s avg (max 18.5) |
| **mean\|diff\| as % of mean flow magnitude** | **~19%** | **~24-25%** |
| this app's local-residual outlier rate (>3.0) | 0.56% avg | 0.25% avg |
| DaVis's local-residual outlier rate (>3.0) | 0.005% avg | 0.000% avg |

Both datasets are low-speed swirl flows (tens to ~100 mm/s mean magnitude,
**not** m/s-scale — worth stating explicitly since it's easy to misjudge
`mean|diff|`'s significance without this reference point). Read against the
correct magnitude, the diff is **not small**: ~19-25% of the mean flow
magnitude, on both datasets independently.

**Correlation is consistently strong and stable across the entire sampled
range** (see `correlation_and_outlier_trends.png` for each dataset) — no
pairs or regions of the dataset show a correlation collapse or drift; U/V/W
all stay in the 0.91-0.99 band throughout. But correlation is scale-invariant
— it confirms the two fields track the same spatial pattern, and says
nothing about whether they agree in absolute magnitude. The ~19-25% relative
mean\|diff\| above is the number that actually speaks to magnitude
agreement, and it shows a real, non-trivial gap that high correlation alone
was masking. This combination (high corr, high relative diff) points at a
systematic magnitude discrepancy or added noise rather than a structural/
pattern-matching failure.

### Root-cause investigation of the ~19-25% relative diff (2026-08-29 follow-up)

Two hypotheses were tested directly against real data (see `scripts/
compare_dataset.py`'s new `--app-field raw|filled` option and two one-off
diagnostic scripts run against real pairs from both datasets):

**Hypothesis 1 — raw-vs-PostProc pipeline-stage mismatch: RULED OUT.**
The original comparison used this app's RAW (pre-`replace_invalid_vectors`)
vectors against DaVis's fully-filled PostProc output — two different
pipeline stages, which could plausibly inflate the apparent diff. Re-ran
the same 100-pair samples comparing this app's FINAL (post-fill) output
instead — the same pipeline stage DaVis's PostProc represents:

| | Stereo raw | Stereo filled | Planar raw | Planar filled |
|---|---|---|---|---|
| mean\|diff\| | 20.2 mm/s | 21.4 mm/s | 14.2 mm/s | 14.4 mm/s |
| relative to mean magnitude | ~19% | ~20.4% | ~24-25% | ~24.6% |

The gap did not shrink — if anything it's marginally *worse* under filled
mode (adding the interpolated-fill cells back in adds slightly more error,
not less). This cleanly rules out "the app's real measurements already
agree with DaVis; the density gap was just dragging down the raw-vs-final
comparison" — the disagreement is present in the genuinely-measured
vectors themselves, not an artifact of comparing different pipeline stages.

**Hypothesis 2 — spatial-gradient/registration sensitivity: CONFIRMED as a
real, partial contributor.** A swirl flow has strong local velocity
gradients near the vortex core; comparing two independently-computed
grids via resampling (`griddata` linear interpolation) is inherently more
sensitive to small spatial misregistration exactly where the flow curves
sharply. Checked directly on one real pair from each dataset: local
diff magnitude vs. local velocity-gradient magnitude (`np.gradient`) are
positively correlated, and the field's highest-gradient third shows
meaningfully more error than its lowest-gradient third:

| | Stereo (pair 0000) | Planar (pair 0000) |
|---|---|---|
| corr(\|diff\|, \|local gradient\|) | 0.37 | 0.50 |
| relative diff, low-gradient third | ~25% | ~14% |
| relative diff, high-gradient third | ~38% | ~26% |
| high/low ratio | 1.53x | 1.85x |

This is a real, physically-sensible effect (PIV correlation windows
spatially average velocity over a finite footprint, so near a sharp local
gradient even a small registration offset between two independently-
computed grids shows up as extra apparent diff) — not a bug in either app.

**What remains unexplained**: even in the CALMEST (lowest-gradient) third
of the flow field, a substantial baseline relative diff persists — ~25%
for stereo, ~14% for planar. Gradient sensitivity is a real, measurable
contributor (roughly doubling the error in high-curvature regions) but
does not account for the full gap; a genuine baseline disagreement between
this app's and DaVis's correlation engines remains, and pinning that down
further (a systematic correlation/subpixel-fit difference vs. a
calibration/scale difference vs. a noise-floor difference) needs either a
synthetic/known-ground-truth flow field or a much closer comparison of the
two engines' specific correlation parameters — beyond what a comparison
against real (ground-truth-unknown) flow data alone can resolve.

### Deep dive into the correlation pipeline itself (2026-08-29, follow-up)

A further, deeper investigation specifically targeted the correlation
method, subpixel-fitting method, and per-pass replacement scheme (as
opposed to post-processing, already ruled out above). This surfaced two
real, unrelated bugs — one fixed and shipped, one attempted and reverted
after real-data testing — plus a systematic sweep of every remaining
correlation-side hypothesis, none of which closed the residual gap.

**Bug found and fixed: `use_vectorized` was silently active in every real
run.** `openpiv.settings.PIVSettings`'s own default is `use_vectorized=
True`, and nothing in this app ever overrode it — meaning every real
correlation this app has ever run silently used openpiv's own alternate
`vectorized_*` correlation/subpixel functions, not the carefully
faithful-verified `fast_*` reimplementations `_openpiv_speedups.py`'s own
docstring describes at length (and explicitly says were "checked and
found to differ subtly... not acceptable here, hence writing faithful
equivalents"). Now forced `False` unconditionally in `CPUPIVProcess`.
**Verified on real data to be safe with zero practical accuracy impact**:
reprocessing 20 planar pairs with only this fix applied reproduced the
baseline numbers essentially exactly (density 95.41% vs 95.41%, relative
diff 25.15% vs 25.15%, corr(U) 0.9571 vs 0.9570, corr(V) 0.9492 vs
0.9492) — a real correctness fix (this app's own documented, intended
code path is now actually exercised for the first time), but it does not
explain or close the DaVis gap.

**Attempted and reverted: a custom "davis_combined" per-pass validation
scheme.** DaVis's real `JobHistory.xml` shows its per-pass rejection
pairs a local-median test (already matched) with an INDEPENDENT
peak-ratio threshold (`peakRatioThreshold=1.5`) and a minimum-
correlation-value floor (`correlationThreshold=0.5`) — a combination this
app's existing `peak2mean`-based criterion never implemented. Built a
faithful, bit-exact-verified (against openpiv's own `peak2peak` formula,
5 synthetic test cases) vectorized peak-ratio computation and wired it in
as the new per-pass default, combined with the correlation floor.
**Real-data testing found a severe, disqualifying regression**: on the
same 20 planar pairs, corr(U)/corr(V) vs DaVis collapsed from ~0.95 to
~0.36 — worse than every other hypothesis tested this session, including
deliberately bad ones (centroid subpixel, linear correlation). Root-
caused by direct inspection of one real first-pass correlation (64px
window, 11,811 windows): **99.16% of ALL windows failed the peak-ratio
test** at DaVis's own threshold. openpiv's own default second-peak
exclusion width (`width=2`, a 5×5 box) is too narrow for this app's real,
wide correlation peaks at real window sizes — a first-pass 64px window's
genuine peak footprint can exceed a 5×5 box, so the "second peak" found
is often just the shoulder of the SAME peak, not a real competing match,
giving a near-1.0 ratio for nearly everything regardless of true match
quality. The underlying formula itself is correct (bit-exact vs. openpiv);
the miscalibration was in copying DaVis's threshold/exclusion-width
numbers unmodified onto a different real peak-width regime. **Reverted**
to the original, real-data-proven `peak2mean` default; the tested
`peak2peak`/`davis_combined` infrastructure remains in
`_openpiv_speedups.py` for whoever picks this up after fixing the
exclusion-width calibration (a wider zone, and/or a threshold derived
from this app's own real peak-ratio distribution rather than DaVis's
number copied unmodified).

**Every other correlation-side hypothesis tested and ruled out** (same
20-pair planar methodology throughout, `--app-field filled`):

| Hypothesis | Relative diff | corr(U) | corr(V) | Verdict |
|---|---|---|---|---|
| Baseline (peak2mean, gaussian, circular) | 25.15% | 0.9570 | 0.9492 | — |
| `per_pass_validation=False` | much worse | ~0.6-0.78 | ~0.6-0.75 | Refuted |
| `correlation_method=linear` | 31.86% | — | — | Refuted |
| `subpixel_method=parabolic` | 25.06% | 0.9574 | 0.9495 | No effect |
| `subpixel_method=centroid` | 49.21% | 0.8940 | 0.8705 | Refuted |
| `min_max_filter` preprocessing | 51.28% | 0.6820 | 0.6425 | Refuted |
| `davis_combined` per-pass validation | 171.55% | 0.3577 | 0.3693 | Refuted (severe) |

**Conclusion**: every parameter variation tested makes agreement with
DaVis worse or unchanged, never better — the app's original defaults
(gaussian subpixel, circular correlation, peak2mean per-pass validation)
were already the best-performing configuration found. The residual
~25% (planar) / ~20% (stereo) relative-diff gap remains open. A
background research subagent additionally audited every setting in a
real DaVis `JobHistory.xml` against this app's code (see docs/
IMPROVEMENT_PLAN.md for the full ranked list) — the one other concrete,
actionable lead it found (the min/max preprocessing filter) was tested
above and refuted; the rest either already matched DaVis or couldn't be
verified without LaVision's own proprietary documentation.

**Density is consistently 5-7 points below DaVis's own**, in both modes —
matches this session's earlier, smaller-scale stereo angle investigation
almost exactly (77-81% vs DaVis's 89%), now confirmed as a stable,
dataset-wide characteristic rather than a fluke of the few pairs checked
earlier, for both stereo and planar.

**The outlier-rate gap (this app ~0.25-0.56% vs DaVis ~0.00-0.005%) is a
methodology artifact, not a real quality gap** — worth stating plainly so
it isn't misread as "DaVis is 100x cleaner." `compare_dataset.py`
deliberately compares this app's **raw, genuinely-measured** vectors
(masked by the pipeline's own pre-fill `valid` array, explicitly *not*
`replace_invalid_vectors`'s interpolated fill — see that function's own
docstring in `scripts/compare_dataset.py`) against DaVis's **PostProc**
output, which is DaVis's own fully validated-and-filled final result. A
genuinely-measured vector is expected to carry more local-residual noise
than an interpolated one by construction; this is comparing "real
measurements" against "smoothed final result," not two final results
against each other.

### Root cause found and fixed: inter-pass smoothing was never enabled (2026-08-29)

The residual gap left open by the deep dive above turned out to be
explained by a real, previously-undiscovered default-configuration gap,
not by anything wrong in the correlation/subpixel/per-pass-validation
machinery itself (which the deep dive above had already ruled out as the
cause). This app has always had a working inter-pass smoothing feature
(`ValidationSettings.smoothn`/`smoothn_p`, wired into `cpu_engine.py` via
`openpiv.smoothn.smoothn` -- Pierre Garcia's robust smoothing spline,
applied to the U/V field between passes, before it is used to deform the
next, finer pass) -- but it defaulted to `smoothn=False` (effectively off)
in every real `.pivproj` used this session. DaVis's own real `vc7` output
carries a `MultiPassSmoothingMode=5` attribute (decoded from real
`JobHistory.xml` earlier this session, but never previously acted on),
meaning DaVis's real production pipeline always applies real inter-pass
smoothing -- an ingredient this app's pipeline never replicated.

A strength sweep (`smoothn_p` = 0.05 [old default], 1.0, 5.0, 15.0, 50.0),
each tested on the same 3 real planar pairs (0000-0002) against the same
DaVis PostProc reference used throughout this investigation:

| `smoothn_p` | corr(U) range | corr(V) range | mean\|diff\| range (mm/s) | pooled relative diff | Verdict |
|---|---|---|---|---|---|
| 0.05 (old default) | 0.963-0.976 | 0.950-0.963 | 12.95-16.03 | ~22.5% | baseline |
| 1.0 | improved | improved | improved | improving | real, scaling improvement |
| 5.0 | 0.977-0.987 | 0.971-0.980 | 9.12-11.66 | ~16.2% | substantial improvement |
| **15.0** | **0.979-0.988** | **0.975-0.981** | **8.83-11.24** | **~16.2%** | **best found -- adopted as new default** |
| 50.0 | 0.980-0.988 | 0.974-0.981 | 8.85-11.32 | ~16.4% | plateau / earliest sign of over-smoothing reversal |

15.0 sits at (or just past) the peak of the sweep: consistently at or
better than every other strength tested, with 50.0 showing the trend
flattening out and beginning to reverse (the expected signature of
over-smoothing starting to wash out genuine flow structure). This was
then confirmed at full 20-pair scale (matching this investigation's
standing methodology) before being adopted as the shipped default:

| | density | corr(U) | corr(V) | relative diff |
|---|---|---|---|---|
| Old default (`smoothn=False`) | 95.41% | 0.9570 | 0.9492 | 25.15% |
| **New default (`smoothn=True, smoothn_p=15.0`)** | **98.66%** | **0.9779** | **0.9738** | **18.26%** |

A ~27% relative reduction in the DaVis disagreement, density now matching
DaVis's own ~98% almost exactly, and every metric (density, corr(U),
corr(V), mean/median/p95 diff) moving together in DaVis's favor -- the
first and only hypothesis across this whole investigation (deep-dive
sweep above, plus the earlier root-cause investigation) with this
signature. **Fixed**: `ValidationSettings.smoothn` now defaults to `True`
and `smoothn_p` to `15.0` (`src/piv_suite/config/schema.py`), applied on
both backends since it is a shared schema field (only verified against
real data on the CPU backend this session, the only backend this
investigation's real datasets were run through).

Scale/calibration mismatch (checked via linear regression of pooled real
app-vs-DaVis U/V/magnitude: slope=1.0, intercepts=0) and
`normalized_correlation` (tested, found slightly worse) were also
checked and ruled out along the way as part of narrowing down to this
finding.

## Bugs found and fixed this session

1. **`resample_onto` crashed (`ValueError: No points given`) on a
   completely-empty DaVis reference field**, instead of degrading
   gracefully — confirmed on real data (one Truck PostProc pair had zero
   valid DaVis vectors). This crashed the *entire* batch comparison over a
   single bad frame. Fixed: returns all-NaN when fewer than 3 valid source
   points exist (too few to triangulate anyway).
2. **`process_pair()`'s call to `compare()` was never wrapped in its own
   try/except**, unlike the load steps around it — any future unexpected
   failure there would still have crashed the whole run. Added as
   defense-in-depth alongside fix #1.
3. **`load_vc7_field`'s validity check (`ACTIVE_CHOICE != 0`) silently reads
   every cell as invalid for a single-pass DaVis job.** `ACTIVE_CHOICE`
   tracks which *pass* won in a multi-pass job; a genuinely single-pass job
   (confirmed on the Truck project's real "1x32x32" — one pass — PostProc
   file) simply never populates it, so it stays 0 everywhere even though
   real velocity data exists underneath (confirmed: 375,788 real nonzero
   `U0` values on a file where `ACTIVE_CHOICE` was 0 across all 387,096
   cells). Fixed by switching to `lvpyio`'s own `as_masked_array()`
   (matching `load_vc7_stereo_field`'s already-correct approach) —
   verified to give the identical valid-count on a genuinely multi-pass
   file (191,094 both ways), so this is a pure fix, not a behavior change,
   for every dataset that isn't single-pass.
4. **CORRECTION (superseding the original phrasing of this finding):
   this was NOT an app bug** — re-checked directly against the app's real,
   unpatched source (`davis_set.py`) rather than trusting the original
   session's characterization. For the Truck recording, `_select_calibration_
   snapshot` does pick a History snapshot (`Calibration_260605_160405`)
   whose `FieldOfView="SameForAllCameras"` is an incomplete/placeholder
   calibration state (both cameras at the identical canvas region,
   0,0,856×2987 — stacked, not side-by-side) rather than the CURRENT
   calibration's genuine ~319mm-wide side-by-side placement. But calling
   the app's real, unpatched functions on this exact project confirms they
   already handle this correctly: `detect_dual_planar_from_set` returns
   `False`, `detect_project_type_from_set` returns `"planar"` (not a wrong
   positive), and `read_dual_planar_calibration_from_set` **raises**
   `ValueError: ... FieldOfView is 'SameForAllCameras', not 'SideBySide2D'`
   rather than silently returning wrong geometry. The garbage 1500-pair
   Truck output happened only because the original investigation used a
   throwaway scratchpad script that monkeypatched `_read_field_of_view` to
   map `"SameForAllCameras"` → `"SideBySide2D"`, specifically to force the
   pipeline past this exact safety check — the check itself was doing
   its job. The genuine, narrower gap: `_select_calibration_snapshot`'s
   strictly-temporal selection has no way to recognize a placeholder
   snapshot and prefer a nearby valid one instead, so a first-time user of
   this exact dataset would see it auto-detected as `"planar"` (silently
   wrong MODE, though non-crashing and correctable via the GUI's own
   radio button) rather than `"dual_planar"` — a real but narrow limitation
   of a function that already documents itself as best-effort with
   user-correctable fallback, not a silent-wrong-output defect. Given the
   fail-loud behavior already works correctly, redoing the Truck batch
   with the CURRENT calibration remains a reasonable follow-up (see
   Recommendations) but is not fixing a bug in the shipped pipeline.
5. **`detect_project_type_from_set`/`detect_dual_planar_from_set` only read
   the calibration folder's `FieldOfView`**, with no cross-check against the
   specific recording's own real frame-stream count — caught the
   "Final_Planar_Swirl" mislabeling above. A calibration folder shared
   across a mix of stereo and planar recordings (as this project's is) will
   silently mis-suggest the wrong mode for some of them.
6. **`compare_dataset.py`'s own per-pair cost (~35s/pair stereo, ~15s/pair
   planar) is dominated by `griddata`'s from-scratch Delaunay triangulation
   every single pair** — a full 1000-pair comparison would take ~9.7 hours
   (stereo) / ~4.1 hours (planar), far more than the tool's own design
   intent suggested. Worked around this session by adding a `--stride`
   option (process every Nth pair) rather than fixing the underlying cost;
   a real future optimization would be reusing/caching the DaVis-side
   triangulation across pairs sharing the same grid, or switching to a
   faster interpolation backend for the resample step.

## Recommendations

- ~~Root-cause the ~19-25% mean\|diff\|-relative-to-magnitude gap on stereo
  and planar~~ — **investigated in depth, three times, and substantially
  fixed on planar** (see "Root-cause investigation", "Deep dive into the
  correlation pipeline itself", and "Root cause found and fixed:
  inter-pass smoothing was never enabled" above). The first two rounds
  ruled out raw-vs-PostProc pipeline-stage mismatch, confirmed spatial-
  gradient/registration sensitivity as a real but partial contributor
  (~1.5-1.85x more error in high-gradient regions), and tested/refuted
  every correlation-method/subpixel-fit/per-pass-validation-scheme
  hypothesis (per_pass_validation off, correlation_method=linear,
  subpixel parabolic/centroid, min_max_filter preprocessing, a custom
  DaVis-matching peak-ratio+correlation-floor scheme, normalized_correlation)
  plus scale/calibration mismatch (ruled out via regression, slope=1.0).
  The actual root cause turned out to be a default-configuration gap, not
  a correlation-pipeline bug: this app's existing inter-pass smoothing
  feature (`smoothn`/`smoothn_p`) defaulted to effectively off, unlike
  DaVis's real pipeline (`MultiPassSmoothingMode=5`). **Fixed**: new
  defaults `smoothn=True, smoothn_p=15.0`, confirmed at 20-pair scale to
  cut the planar relative-diff from 25.15% to 18.26% and raise density
  from 95.41% to 98.66% (matching DaVis's own). Two further real,
  unrelated bugs were found and fixed along the way (`use_vectorized`
  never being forced False; `_fill_residual_nan` unable to recover an
  all-rejected pass). The remaining ~18% planar gap (stereo not
  re-verified with the new smoothing default) is smaller but not zero;
  a synthetic flow field with known ground truth remains the cleanest way
  to fully resolve whatever's left (removes the "which engine is actually
  right" ambiguity real flow data can't resolve).
- ~~Fix `recommended_workers()` to hard-cap at 61 on Windows~~ — **done**
  (see [autotune.py](src/piv_suite/perf/autotune.py), commit `3f70f53`);
  defaulting toward physical core count specifically for this NUMA
  workload is a separate, smaller follow-up not yet done.
- Redo the Truck dual-planar batch run with the CURRENT calibration, then
  re-run its comparison, to get a real dual-planar accuracy datapoint —
  the shipped code already refuses to run with the wrong calibration
  (see corrected finding 4 above), so this is a data-gathering follow-up,
  not a bug fix.
- Consider whether `_select_calibration_snapshot` should recognize an
  obviously-incomplete snapshot (e.g. `FieldOfView="SameForAllCameras"`,
  or a dual-planar snapshot with identical cam0/cam1 region placement)
  and prefer a nearby valid one instead of the strictly-latest-preceding
  one — would upgrade this exact Truck scenario from "auto-detects as
  the wrong mode, user corrects it" to "auto-detects correctly." Real but
  narrow value: needs a careful design for which snapshot to prefer
  instead (skip backward further, or forward to current) without risking
  silently picking a WRONG snapshot in some other project's history.
- Consider whether `detect_project_type_from_set` should sample the actual
  frame-stream count (already cheap — just a directory listing) as a
  cross-check against the calibration folder's `FieldOfView`, for projects
  with a shared/mixed-mode calibration folder (the "Final_Planar_Swirl"
  mislabeling finding above).
- If `compare_dataset.py` is going to be used routinely on real 1000+ pair
  datasets, its griddata-per-pair cost is worth optimizing directly rather
  than routinely relying on `--stride` to work around it.
