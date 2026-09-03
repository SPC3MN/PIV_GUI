# Improvement backlog

Tracks repository-wide correctness/robustness/testability/performance work,
separate from feature requests. Update this file as items complete or new
ones are found — keep it reflecting actual repo condition, not aspirational
work.

## Completed (2026-08-29 end-to-end improvement pass)

- **Fixed `fast_replace_nans`'s stale numerical patch** — openpiv silently
  moved 0.25.4→0.25.5 with no version pin, and 0.25.5 fixed a real
  convergence-check aliasing bug this app's patch deliberately replicated.
  Every CPU-backend run was filling invalid vectors with logic that no
  longer matched real, unpatched openpiv. Rewrote to match 0.25.5 exactly
  (verified bit-exact across every method/kernel_size/max_iter/tol
  combination), pinned `openpiv>=0.25.5,<0.26`, added a version-tripwire
  test. `cd819ff`
- **Capped `recommended_workers()` at Windows' 61-worker limit** — a real
  crash on any Windows machine with >61 logical processors, hit for real
  this session. `3f70f53`
- **Corrected a mischaracterized finding in `DATASET_VALIDATION_REPORT.md`**
  — the Truck dataset's bad calibration was not an app bug; the shipped
  code already raises a clear error for it. `14e472e`
- **Fixed CLI parallelism silently disabled for single-`.set` runs** —
  `interactive_preview` (True for any non-batch run, regardless of pair
  count) also gated Tier-3 parallelism, so a real single-`.set` batch of
  hundreds/thousands of pairs ran fully serial with no warning. `2ac911f`
- **Declared `tifffile`/`imageio` as direct dependencies** — previously
  relied on openpiv's own transitive dependency graph for a first-class,
  documented feature (loose TIFF/PNG/BMP/JPG pairs). Added direct test
  coverage for `io/readers.py` (RGB/RGBA→grayscale collapse, unsupported
  extensions). `b82f548`
- **Made `save_project` atomic; clear error on malformed `.pivproj` JSON**
  — a process killed mid-write (Ctrl+C, disk full, power loss) could
  truncate the config file with no recovery path; `save_project` runs on
  every CLI invocation. `aad9971`
- **Added `--app-field raw|filled` to `compare_dataset.py` and root-caused
  the mean\|diff\| gap** (see below — moved from HIGH PRIORITY to
  completed, with a narrower follow-up item in its place). `55677dd`
- **Fixed `use_vectorized` never being forced False** — `PIVSettings`'s own
  default (`True`) meant every real CPU-backend correlation silently used
  openpiv's own alternate `vectorized_*` functions, never the carefully
  faithful-verified `fast_*` path this app's whole `_openpiv_speedups.py`
  module is built around. Verified safe on real data (identical results to
  baseline, zero practical accuracy impact) — a real correctness fix,
  restoring this app's own intended code path for the first time. `d9f5b19`
- **Fixed a latent `_fill_residual_nan` crash risk** — `np.nanmedian` of an
  entirely-NaN pass (a real, reachable case: a genuinely-uncorrelated or
  fully-rejected pass) is itself NaN, silently leaving the field NaN right
  where this function's job is to prevent that, reaching the next pass's
  spline deformation and crashing with a real, uncatchable Windows access
  violation. Falls back to 0.0 when the median itself is NaN. `d9f5b19`
- **Attempted and reverted a custom DaVis-matching per-pass validation
  scheme** (`sig2noise_method="davis_combined"`, peak-ratio + correlation-
  floor) — built from real DaVis settings, bit-exact-verified on synthetic
  data, but real-data testing found it rejects 99.16% of real first-pass
  windows (openpiv's own default peak-ratio exclusion width is too narrow
  for this app's real, wide correlation peaks), collapsing DaVis-agreement
  from corr~0.95 to corr~0.36. Reverted to the proven-safe `peak2mean`
  default; the tested infrastructure remains available for a properly
  recalibrated future attempt. `9e475af`
- **Root-caused and fixed most of the residual DaVis accuracy gap on
  planar data: inter-pass smoothing was never enabled.** This app's
  existing `smoothn`/`smoothn_p` feature (openpiv's own Garcia robust
  smoothing, applied between passes) defaulted to effectively off, unlike
  DaVis's real pipeline (`MultiPassSmoothingMode=5`, decoded from real
  `JobHistory.xml` earlier this session but not previously acted on). A
  strength sweep (0.05/1.0/5.0/15.0/50.0) on real planar data found a
  clear, consistently-scaling improvement peaking around `smoothn_p=15.0`
  (50.0 showed the trend plateauing/reversing — early over-smoothing).
  Confirmed at full 20-pair scale: relative diff 25.15%→18.26%, density
  95.41%→98.66% (now matching DaVis's own), corr(U) 0.9570→0.9779, corr(V)
  0.9492→0.9738. New defaults: `smoothn=True, smoothn_p=15.0` (`src/
  piv_suite/config/schema.py`). See `DATASET_VALIDATION_REPORT.md`'s "Root
  cause found and fixed: inter-pass smoothing was never enabled" section
  for the full sweep table.

## HIGH PRIORITY

- ~~Root-cause the ~19-25% mean\|diff\|-relative-to-magnitude gap~~ —
  **investigated in depth, and substantially fixed on planar** (2026-08-29;
  see `DATASET_VALIDATION_REPORT.md`'s "Root-cause investigation", "Deep
  dive into the correlation pipeline itself", and "Root cause found and
  fixed: inter-pass smoothing was never enabled" sections for full
  numbers). Findings:
  - Raw-vs-PostProc pipeline-stage mismatch: **ruled out** — re-comparing
    this app's post-fill (final) output instead of raw measurements
    (`--app-field filled`) did not shrink the gap (stereo 19%→20.4%,
    planar ~24-25%→24.6%, both slightly worse, not better).
  - Spatial-gradient/registration sensitivity: **confirmed as a real,
    partial contributor** — local diff correlates with local velocity-
    gradient magnitude (r=0.37 stereo, r=0.50 planar); the highest-
    gradient third of each field shows 1.5-1.85x more relative error than
    the lowest-gradient third. Physically sensible for a swirl flow
    (strong local curvature near the vortex core makes spatial
    registration between two independently-computed grids more
    sensitive), not a bug.
  - Every correlation-pipeline knob tested (`per_pass_validation=False`,
    `correlation_method=linear` (since removed from the GUI and coerced
    away on config load -- see config.io.from_dict),
    `subpixel_method=parabolic/centroid`,
    `min_max_filter` preprocessing, the custom `davis_combined` per-pass
    scheme above): **all made agreement with DaVis worse or unchanged,
    never better** — the app's original defaults are already the best-
    performing configuration found across every real-data test run.
  - **Root cause found and fixed**: this app's inter-pass smoothing
    feature (`smoothn`/`smoothn_p`) defaulted to effectively off, unlike
    DaVis's real pipeline (`MultiPassSmoothingMode=5`). New defaults
    (`smoothn=True, smoothn_p=15.0`, found via a real-data strength sweep)
    cut the planar relative diff from 25.15% to 18.26% (20-pair confirmed)
    and raised density from 95.41% to 98.66%, matching DaVis's own. See
    `DATASET_VALIDATION_REPORT.md`'s "Root cause found and fixed" section.
  - **Remaining open item**: the fix and its confirmation were planar-only
    (scoped this way deliberately, to save time); stereo has not been
    re-verified with the new smoothing default and may see a similar or
    different improvement. The ~18% residual planar gap is smaller but not
    zero — a synthetic flow field with a known ground truth remains the
    cleanest way to fully resolve whatever's left (real swirl data can't
    establish which engine is "right" for the remainder).

## MEDIUM PRIORITY

- **`_select_calibration_snapshot` can't recognize a placeholder/incomplete
  calibration snapshot** (e.g. `FieldOfView="SameForAllCameras"`, or a
  dual-planar snapshot with identical cam0/cam1 region placement) and
  picks it anyway when it's the latest one preceding a recording. The
  shipped code already fails safely when this happens (raises rather than
  producing wrong geometry — confirmed directly against real data), but
  auto-detection silently falls back to the wrong MODE (e.g. "planar"
  instead of "dual_planar") for a project with this exact history shape.
  Needs careful design for which snapshot to prefer instead (skip further
  back, or forward to "current") without risking a wrong pick in some
  other project's history — do not implement without validating against
  more than one real project's calibration history.
- **`detect_project_type_from_set`/`detect_dual_planar_from_set` only read
  the calibration folder's `FieldOfView`**, never cross-checking the
  specific recording's own raw frame-stream count. A calibration folder
  shared across mixed stereo/planar recordings (confirmed on a real
  project this session) can mislabel a genuinely single-camera recording
  as stereo. The frame-stream count is already cheap to check (a
  directory listing).
- **`compare_dataset.py`'s per-pair cost (~15-35s/pair) is dominated by
  `scipy.griddata`'s from-scratch Delaunay triangulation every pair.** A
  full 1000-pair comparison takes hours; `--stride` works around this but
  doesn't fix it. Worth caching/reusing the DaVis-side triangulation
  across pairs sharing the same grid, or a faster interpolation backend.
- **GPU backend code paths (`engines/gpu_engine.py`, `processing/
  parallel_*` GPU branches) got no real-hardware exercise in this
  improvement pass** — no CUDA hardware was available in this environment
  (`test_gpu_tiling.py`'s CUDA-dependent tests are skipped here). Not
  touched, since changes there couldn't be genuinely verified. Worth a
  dedicated pass on real GPU hardware.

## LOW PRIORITY

- No linter/type-checker (ruff/mypy/flake8) is configured for this
  project. A quick manual sweep (bare excepts, mutable default args,
  TODOs) found nothing — this codebase is unusually disciplined already
  — but a configured linter would catch future drift automatically.
  Introducing one is a tooling decision worth raising with the maintainer
  rather than doing unprompted (choice of tool, rule set, and whether to
  enforce in CI are all real decisions, not a bug fix).
