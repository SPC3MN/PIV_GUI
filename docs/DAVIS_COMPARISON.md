# This app vs LaVision DaVis

How this PIV suite's output compares to LaVision DaVis's, and where the two
programs genuinely work differently. Two purposes: a record of current
measured accuracy against a real reference, and a map of the structural
differences worth knowing before trusting either program's output on a new
kind of recording.

**Not a changelog.** This describes current behavior and current measured
agreement, not what changed to get here — see git history for that.

## Dataset and methodology

Two complete DaVis 10.2 projects, each 1000 double-frame snapshots at
4096×3008 px, 700 µs inter-frame separation:

| | Planar | Stereo |
|---|---|---|
| Cameras | 1 | 2 |
| Calibration model | Polynomial 3rd order, 2 Z-planes | Polynomial 3rd order, 2 Z-planes |
| Calibrated Z-planes | +1.0 / −2.0 mm | +1.0 / −2.0 mm |
| Scale | 19.422 px/mm | 17.921 px/mm |
| DaVis's own output grid | 379 × 514 | 384 × 735 |
| This app's grid | 373 × 509 | 380 × 731 |

Both ship DaVis's own finished vectors (1000 `.vc7` files each) and the
`JobHistory.xml` recording DaVis's own processing parameters. This app was
configured to match those parameters directly rather than tuned against the
comparison: one 64×64 pass at 50% overlap followed by three 32×32 passes at
75% overlap, DaVis's own local-median universal-outlier-detection removal
factor (2), insertion factor (3), and minimum-neighbour rule (3), on a
3×3 neighbourhood.

Every number below comes from a 100-pair sample of each recording (pairs
0–99), CPU backend, on a 16-logical-core / 7.7 GB machine. Comparison
re-centres both fields on their own bounding box, resamples DaVis's field
onto this app's grid (`RegularGridInterpolator` — DaVis's field is already a
regular grid), corrects for DaVis's V sign convention, and scores only cells
both sides measured. Density is always measured on each side's own native
grid, never after resampling — resampling onto this app's grid, which is
inset half an interrogation window from the canvas edge, would flatter
DaVis by simply never asking about the border cells where its own field is
invalid.

## Performance

### Per-pair agreement

| | Planar | Stereo |
|---|---|---|
| Density, this app | 99.33% | 90.45% |
| Density, DaVis | 98.10% | 88.90% |
| corr(U) | 0.9914 | 0.9881 |
| corr(V) | 0.9892 | 0.9879 |
| corr(W) | — | 0.9861 |
| mean\|diff\| | 9.51 mm/s | 11.55 mm/s (in-plane); 8.38 mm/s (W) |

### Ensemble-mean agreement

The time-averaged field over all 100 pairs — what most downstream analysis
actually consumes, and where per-pair noise averages out:

| Component | corr | this app (mm/s) | DaVis (mm/s) | mean\|diff\| |
|---|---|---|---|---|
| Planar U | 0.9694 | −6.09 | −5.64 | 1.16 |
| Planar V | 0.9722 | 4.72 | 5.14 | 1.21 |
| Stereo U | 0.9817 | −6.08 | −4.90 | 2.35 |
| Stereo V | 0.9827 | 1.41 | 2.22 | 1.63 |
| Stereo W | 0.9598 | −15.47 | −15.21 | 1.70 |

The per-pair residual (9.5–11.5 mm/s) is almost entirely uncorrelated
frame-to-frame noise rather than bias: it collapses to 1.2–2.4 mm/s once
averaged over 100 pairs.

### Internal field consistency

Each side's own field judged against its own local neighbourhood — no
cross-comparison, no resampling. This asks whether a field agrees with
*itself*, independent of whether it agrees with the other program:

| | Valid % | Local residual, p50 | Local residual, p99 | >3 local MAD |
|---|---|---|---|---|
| Planar, this app | 99.47 | 0.358 | 1.298 | 0.004% |
| Planar, DaVis | 98.10 | 0.396 | 1.346 | 0.015% |
| Stereo, this app | 90.35 | 0.329 | 1.155 | 0.003% |
| Stereo, DaVis | 88.92 | 0.331 | 1.140 | 0.020% |

(30-pair sample of each recording; "local residual" is deviation from the
local median normalized by local MAD, Westerweel & Scarano; >3 MAD is a
conventional spurious-vector threshold.)

### Throughput

DaVis's own `Settings_ProcessingTime.xml` reports 3 h 53 m for the full
1000-pair planar job (14.0 s/pair) and 7 h 58 m for the full 1000-pair
stereo job (28.7 s/pair). This app's CPU backend, on the 16-core/7.7 GB
machine above, processed the 100-pair planar sample in 21.1 s/pair wall
clock at 3 worker processes, and stereo samples in roughly 80–110 s/pair
wall clock at 2 worker processes (stereo's four 3067×5874 dewarp canvases
per pair make memory, not core count, the binding constraint on this
machine).

**This is not a controlled hardware comparison.** DaVis's own hardware for
this dataset is not recorded in the project (its result-set metadata
references GPU-accelerated preprocessing, suggesting GPU involvement of
some kind), while the numbers above are this app's CPU backend only. This
app also has a separate GPU backend (`piv_suite.engines.gpu_engine`,
built on `openpiv-python-gpu`) not exercised in this comparison.

## Functional differences

Structural choices where the two programs genuinely work differently, not
just parameter values that happen to differ.

**Image correction is asymmetric between modes.** DaVis's planar job for
this dataset ran "using image correction" — it dewarps each raw frame onto
a rectified canvas with the full polynomial calibration before correlating,
and its planar vectors live on that canvas. This app's planar path
correlates raw sensor pixels directly and converts displacement to physical
units with a single scalar (`CalibrationSettings.pixel_pitch_mm`) — it does
not dewarp planar images. Stereo is the opposite: both apps dewarp each
camera's frame onto a shared world grid before correlating, because stereo
triangulation requires a common coordinate system between the two cameras
(`calibration.camera_mapping.CameraMapping.dewarp_image`, called from
`processing.parallel_stereo`). For this rig's calibration the planar
polynomial's own raw-vs-corrected magnification varies only ~0.3% across
the field, so the practical effect on this dataset is small — a project
with a more strongly distorting lens would see a bigger gap between the two
approaches.

**The correlation grids don't share an origin.** DaVis starts its first
interrogation window at the corrected canvas's corner. This app insets the
first window by half a window from the frame edge (the standard
openpiv/windef convention), which yields a grid a few percent smaller in
each dimension that isn't cell-for-cell aligned with DaVis's own — hence
every comparison tool in `scripts/` resamples one field onto the other's
grid rather than differencing arrays directly.

**Outlier detection layers a global filter DaVis's own job didn't use.**
Both programs run the same kind of local-median universal outlier
detection (DaVis's `medianUniversalOutlierRemovalFactor`/`InsertionFactor`/
`FilterLength`/`MinNoNeighbours`, mirrored by
`processing.postprocess.range_filter`'s `residual_max`/`insertion_max`/
`window_size`/`min_neighbours`). This app additionally applies an
always-on field-wide standard-deviation cutoff
(`PostProcessSettings.global_outlier_std`, default 3.0) after correlation;
DaVis's own job for both of these recordings has
`useAllowedVectorRange=false` — no global range filter at all. On these
particular (now quite clean) fields the global filter is measurably
conservative: disabling it on a planar sample raised density from 99.39%
to 99.99% while correlation and mean\|diff\| moved by less than 0.001 and
0.03 mm/s respectively. DaVis does not implement a comparable field-wide
statistical gate for these jobs; this app's local-only vs local-plus-global
philosophies genuinely diverge here, and the right global threshold likely
depends on how clean a given field already is.

**Stereo triangulation angles are derived differently.** This app computes
the per-pixel viewing angle for each camera directly from the calibration's
own two Z-planes (`calibration.camera_mapping.stereo_view_angles`), rather
than using one fixed angle per camera — on this rig the true angle varies
several degrees across the field of view. DaVis's own internal
triangulation approach isn't inspectable from the project files, so this is
a stated design choice on this app's side rather than a directly verified
difference, but it is a departure from simpler stereo PIV implementations
that do use a single scalar angle per camera.

**Stereo needs one manual, acquisition-time input DaVis's calibration
doesn't expose.** `StereoSettings.sheet_z_mm` — the real Z position of the
laser sheet for a specific recording, between the calibration's two
Z-planes — has no source in any DaVis calibration file; it's an
acquisition-time fact, not a calibration-time one. This app requires it as
an explicit parameter (or a manual GUI entry). It matters: on a sample
pair, corr(W) ranged from 0.981 to 0.993 depending on the chosen value
across a physically plausible 1 mm range, with mean\|diff\| swinging from
6.4 to 10.5 mm/s. Whoever ran the acquisition needs to supply this
correctly per recording; there is no way to derive it after the fact.

**Preprocessing matches DaVis's own reported step.** Both of DaVis's jobs
for this dataset report a sliding min/max local-contrast-normalization
filter (`useMinMaxFilter=true`, `minMaxFilterLength=4`). This app implements
the equivalent filter (`processing.preprocess.min_max_filter`) with a
matching default window length, on by default.

**Inter-pass smoothing exists on both sides but isn't the same
algorithm.** DaVis's own `.vc7` output carries a `MultiPassSmoothingMode`
attribute, confirming it smooths the displacement field between multi-pass
iterations; this app applies Garcia robust smoothing
(`openpiv.smoothn`, `ValidationSettings.smoothn_p`) for the same purpose.
The two are not the same implementation, so their strengths aren't
directly comparable parameter-for-parameter — only the presence of some
inter-pass smoothing is common ground.

**The underlying correlation implementations differ.** DaVis's own
`JobHistory.xml` records `correlationFct=0` (its own FFT cross-correlation);
this app's CPU backend uses `openpiv-python`'s circular (FFT)
cross-correlation with a Gaussian sub-pixel fit. The two were matched on
directly comparable parameters (window sizes, overlap, pass count,
sub-pixel method where applicable) but are independent numerical
implementations, not the same code.

**Batch parallelism is this app's own addition.** This app parallelizes
across independent pairs within one recording via a process pool sized to
CPU core count (`processing.parallel_planar`/`parallel_stereo`); DaVis's
own per-pair throughput and hardware for this dataset aren't recorded in
the project, so — as noted above — the two aren't on comparable hardware
footing for a throughput claim, only for the accuracy numbers.

## Tooling

Four scripts in `scripts/` do the comparison work above and are reusable
against any DaVis `.set` project with its own real `.vc7` output, not just
this dataset:

- **`piv_batch_sample.py`** — runs a bounded slice (N pairs starting at
  index M) of a `.set` through this app's own CLI pipeline, for a fast
  experiment against a recording that's otherwise hours long to process in
  full.

      python scripts/piv_batch_sample.py --mode planar \
          --set "C:\data\MyRecording.set" --out piv_sample_output --n 25

- **`piv_parameter_sweep.py`** — A/B one or more configuration variants
  against DaVis's own vectors in a single command, printing a comparison
  table. Add entries to its `VARIANTS` dict to try something new.

      python scripts/piv_parameter_sweep.py --mode planar \
          --set-file "C:\data\MyRecording.set" \
          --vc7-dir "C:\data\MyRecording\PIV_MPd(...)" \
          --variants base,smoothn_off,interp_5 --n 3

- **`piv_field_quality.py`** — the internal-consistency table above, for
  any finished batch output, in well under a minute (no cross-field
  resampling, unlike `compare_dataset.py`'s own outlier statistics).

- **`make_comparison_plots.py`** — the density/correlation trend,
  ensemble-mean field, parity, and centreline-profile figures used above,
  from a finished batch output plus DaVis's own `.vc7` files.

These sit alongside the pre-existing `compare_dataset.py`,
`compare_davis_lavision.py`, `compare_stereo_preview.py`, and
`compare_velocity_fields.py`, which remain the tools for a full-dataset
resumable comparison run and the shared field-statistics/plotting code the
new scripts import.

## Caveats

- All numbers above are from one planar and one stereo recording off one
  rig. They characterize agreement on this dataset, not a general accuracy
  guarantee for arbitrary lens distortion, seeding density, or flow
  character.
- The throughput comparison is not hardware-controlled — see above.
- The global-outlier-filter observation is a live open question, not a
  settled recommendation: the right threshold plausibly depends on how
  clean a given field already is, and this app's default is unchanged
  pending a broader sweep across more datasets.
- The sheet-Z sensitivity measurement is from a 2-pair sample; treat the
  specific correlation numbers as illustrative of the sensitivity's
  existence and rough size, not as a precise calibration curve.
