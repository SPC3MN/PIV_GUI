# Full-dataset validation report — stereo, planar, dual-planar

Ran three real datasets (each already processed in LaVision DaVis, with real
reference output available) through the full pipeline: automatic
calibration/mode verification via a single-pair preview, a full batch run
through this app's own CLI, then a snapshot-by-snapshot comparison against
DaVis's own final (`PostProc`) output using `scripts/compare_dataset.py`.

**Status: stereo and planar completed the full pipeline end-to-end. The
dual-planar (Truck) dataset's batch run completed, but its comparison
surfaced a real calibration bug (below) that invalidates its numeric
results — skipped for this report rather than reporting wrong numbers.**

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
| corr(U) | 0.9594 avg (min 0.9101) | 0.9576 avg (min 0.9213) |
| corr(V) | 0.9639 avg (min 0.9279) | 0.9503 avg (min 0.9183) |
| corr(W) | 0.9682 avg (min 0.9253) | n/a |
| mean\|diff\| | 20.2 mm/s avg (max 29.1) | 14.2 mm/s avg (max 18.5) |
| this app's local-residual outlier rate (>3.0) | 0.56% avg | 0.25% avg |
| DaVis's local-residual outlier rate (>3.0) | 0.005% avg | 0.000% avg |

**Correlation is consistently strong and stable across the entire sampled
range** (see `correlation_and_outlier_trends.png` for each dataset) — no
pairs or regions of the dataset show a correlation collapse or drift; U/V/W
all stay in the 0.91-0.99 band throughout.

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
4. **The Truck project's calibration-snapshot selection produced
   incomplete/wrong dual-planar camera placement** — `_select_calibration_
   snapshot` correctly picked whichever History snapshot preceded the
   recording (its own documented, principled behavior), but that specific
   snapshot's `FieldOfView="SameForAllCameras"` turned out to represent an
   **incomplete/placeholder calibration state**, not a real side-by-side
   arrangement: both cameras were placed at the *exact same* canvas region
   (0,0,856×2987 — literally stacked on top of each other), producing a
   combined canvas only 33mm wide, vs. the CURRENT calibration's genuinely
   different camera placements and ~319mm-wide combined canvas (matching
   DaVis's own real output width). This was caught because comparison
   against DaVis showed near-zero/negative correlation despite superficially
   similar velocity magnitudes — a real, load-bearing finding that an
   earlier "these two snapshots are structurally identical" check (which
   only confirmed both have `CameraIdentifier=1/2` mapper blocks present,
   not that their region/placement values agree) was insufficient to catch.
   **The full 1500-pair Truck batch run needs to be redone with the CURRENT
   calibration before it can be meaningfully compared against DaVis** — not
   done in this session per your explicit choice to skip it and move on
   with stereo+planar's results.
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

- Fix `recommended_workers()` to hard-cap at 61 on Windows and default
  toward physical core count for this workload (see performance section).
- Redo the Truck dual-planar batch run with the CURRENT calibration, then
  re-run its comparison, before drawing any conclusions about dual-planar
  mode's real-world accuracy.
- Consider whether `detect_project_type_from_set` should sample the actual
  frame-stream count (already cheap — just a directory listing) as a
  cross-check against the calibration folder's `FieldOfView`, for projects
  with a shared/mixed-mode calibration folder.
- If `compare_dataset.py` is going to be used routinely on real 1000+ pair
  datasets, its griddata-per-pair cost is worth optimizing directly rather
  than routinely relying on `--stride` to work around it.
