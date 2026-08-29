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

## HIGH PRIORITY

- ~~Root-cause the ~19-25% mean\|diff\|-relative-to-magnitude gap~~ —
  **investigated 2026-08-29** (see `DATASET_VALIDATION_REPORT.md`'s
  "Root-cause investigation" section for full numbers). Findings:
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
  - **Remaining open item**: even the calmest (lowest-gradient) regions
    still show a substantial baseline relative diff (~14-25%), unexplained
    by either hypothesis above. Next step needs either a synthetic flow
    field with a known ground truth (real swirl data can't establish which
    engine is "right"), or a closer parameter-by-parameter comparison of
    this app's vs. DaVis's specific correlation/subpixel-fit settings for
    this job — a genuinely new investigation, not a quick follow-up.

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
