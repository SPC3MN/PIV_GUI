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
| DaVis's own output grid | 379 × 514 (194,806) | 384 × 735 (282,240) |
| This app's grid | 373 × 509 (189,857) | 380 × 731 (277,780) |

Both ship DaVis's own finished vectors (1000 `.vc7` files each) and the
`JobHistory.xml` recording DaVis's own processing parameters. This app was
configured to match those parameters directly rather than tuned against the
comparison: one 64×64 pass at 50% overlap followed by three 32×32 passes at
75% overlap, DaVis's own local-median universal-outlier-detection removal
factor (2), insertion factor (3), and minimum-neighbour rule (3), on a
3×3 neighbourhood.

**Every number below is the full 1000 pairs of each recording** — no
sampling. CPU backend, 48 worker processes, on a dual Xeon Gold 6146
workstation (24 physical / 48 logical cores, 191.7 GB RAM). Comparison
re-centres both fields on their own bounding box, resamples DaVis's field
onto this app's grid, corrects for DaVis's V sign convention (which differed
on all 1000 pairs of both recordings — a consistent convention difference,
not a per-pair anomaly), and scores only cells both sides measured
(~188k of 189,857 planar, ~248k of 277,780 stereo). Density is always
measured on each side's own native grid, never after resampling —
resampling onto this app's grid, which is inset half an interrogation
window from the canvas edge, would flatter DaVis by simply never asking
about the border cells where its own field is invalid.

## Accuracy

### Per-pair agreement

Mean across all 1000 pairs, with spread:

| | Planar | Stereo |
|---|---|---|
| Density, this app | **99.23%** (sd 0.66) | **90.48%** (sd 0.81) |
| Density, DaVis | 98.10% (sd 0.01) | 89.01% (sd 0.23) |
| corr(U) | 0.9888 (sd 0.0054) | 0.9855 (sd 0.0059) |
| corr(V) | 0.9840 (sd 0.0067) | 0.9855 (sd 0.0057) |
| corr(W) | — | 0.9849 (sd 0.0081) |
| mean\|diff\| | 11.62 mm/s | 11.90 mm/s (in-plane); 8.68 mm/s (W) |
| median\|diff\| | 9.70 mm/s | 9.83 mm/s (in-plane); 6.49 mm/s (W) |
| p95\|diff\| | 27.18 mm/s | 26.68 mm/s |

Against a mean flow speed of ~107 mm/s, the per-pair residual is ~11% of
signal. The stereo out-of-plane component agrees as well as the in-plane
ones (0.985 vs 0.986) despite W being the component most sensitive to
calibration and triangulation geometry — and its absolute residual is
smaller than in-plane (8.68 vs 11.90 mm/s).

Density is remarkably stable pair-to-pair on DaVis's side (sd 0.01% planar)
and more variable on this app's (sd 0.66%, min 95.13%), reflecting that
DaVis's invalid cells here are dominated by a fixed geometric border while
this app's vary with each pair's own seeding and correlation quality.

### Ensemble-mean agreement

The time-averaged field over all 1000 pairs — what most downstream analysis
actually consumes, and where per-pair noise averages out:

| Component | corr | this app (mm/s) | DaVis (mm/s) | mean\|diff\| |
|---|---|---|---|---|
| Planar U | 0.9720 | −3.41 | −3.44 | 0.63 |
| Planar V | 0.9956 | −0.55 | −0.59 | 0.42 |
| Stereo U | 0.9711 | 3.82 | 4.05 | 0.81 |
| Stereo V | 0.9837 | −1.41 | −1.41 | 0.46 |
| Stereo W | 0.9899 | −10.54 | −10.54 | 0.62 |

The ~12 mm/s per-pair residual collapses to 0.4–0.8 mm/s once averaged —
it is almost entirely uncorrelated frame-to-frame noise, not bias. The
ensemble means themselves agree to within 0.04 mm/s on every component, and
stereo V and W agree to the reported precision.

### Internal field consistency

Each side's own field judged against its own local neighbourhood — no
cross-comparison, no resampling. This asks whether a field agrees with
*itself*, independent of whether it agrees with the other program:

| | Valid % | Local residual, p50 | Local residual, p99 | >3 local MAD |
|---|---|---|---|---|
| Planar, this app | 99.23 | 0.410 | 1.538 | **0.0067%** |
| Planar, DaVis | 98.10 | 0.439 | 1.526 | 0.0323% |
| Stereo, this app | 90.48 | 0.335 | 1.186 | **0.0030%** |
| Stereo, DaVis | 89.01 | 0.337 | 1.197 | 0.1341% |

("Local residual" is deviation from the local median normalized by local
MAD, Westerweel & Scarano; >3 MAD is a conventional spurious-vector
threshold.)

Both programs produce fields of very similar internal smoothness (p50 and
p99 residuals within a few percent of each other). Where they differ is the
tail: this app leaves roughly 5× fewer >3-MAD vectors than DaVis on the
planar recording and ~45× fewer on the stereo one, while also carrying
slightly *more* vectors overall. The same pattern shows in peak speeds —
DaVis's per-pair maximum averages 353.7 mm/s planar / 442.9 mm/s stereo
against this app's 312.3 / 320.2, on fields whose *mean* speeds agree to
within 1% (107.5 vs 108.4, and 105.9 vs 106.8). Those extra DaVis extremes
are concentrated in the spurious-vector tail rather than distributed
through the field.

## Throughput

DaVis's own `Settings_ProcessingTime.xml` reports 3 h 53 m for the full
1000-pair planar job (14.0 s/pair) and 7 h 58 m for the full 1000-pair
stereo job (28.7 s/pair).

This app's CPU backend, processing the same 1000 pairs on the machine
described above with 48 worker processes:

| | Wall clock | Per pair (wall) | Per pair (single-core serial) | DaVis's own |
|---|---|---|---|---|
| Planar | 1 h 34 m | 5.6 s | 210.3 s | 14.0 s |
| Stereo | 7 h 19 m | 26.3 s | 657.3 s | 28.7 s |

The gap between the serial and wall-clock columns is the cross-pair process
pool: pairs are independent, so throughput scales with cores until memory
bandwidth binds. Measured speedup was ~29× on 24 physical cores.

Within a stereo pair the cost is overwhelmingly correlation, not geometry:
~96% correlation, 3–4% preprocessing plus the four 3067×5874 dewarp
canvases, and a negligible remainder for post-processing. Stereo costs
~3.1× planar per pair because it correlates two cameras over a 1.46× larger
grid.

**This is not a controlled hardware comparison.** DaVis's own hardware for
this dataset is not recorded in the project (its result-set metadata
references GPU-accelerated preprocessing, suggesting GPU involvement of some
kind), while the numbers above are this app's CPU backend only on a
24-core workstation. This app also has a separate GPU backend
(`piv_suite.engines.gpu_engine`, built on `openpiv-python-gpu`) which was
not used here — the machine's GPU is a 2014-era Quadro K2200 (640 cores,
4 GB), which is unlikely to beat 24 modern Xeon cores on this workload and
whose 4 GB would force the tiled path for stereo's dewarp canvases. Treat
the throughput table as "what each program actually took on the hardware it
ran on," not as a like-for-like efficiency ranking.

## Pass-count sensitivity

The production schedule — one 64×64 pass at 50% overlap then three 32×32 at
75% — mirrors DaVis's own job rather than being tuned. Dropping to **two**
32×32 passes was measured over the first 100 pairs of each recording, against
the same schedule otherwise unchanged. The 3-pass rerun reproduces
`batch_planar_full` bit for bit, so the two runs differ in pass count and
nothing else.

**Vector density is unaffected.** The output grid is identical by construction
— it comes from the final pass alone, and that pass is unchanged — so
`n_total` is fixed (189,857 planar, 277,780 stereo). The engine rejects
nothing either: `cpu_engine` returns `val_locations` all-False, and validity is
decided afterwards by `processing.postprocess`. Pass count therefore reaches
density only through how converged the field is when the UOD sees it, and it
barely moves:

| | 3 fine passes | 2 fine passes | change | retained |
|---|---|---|---|---|
| Planar | 99.2257% | 99.2225% | −0.0032 pp (t = −2.15) | 99.954% |
| Stereo | 90.620% | 90.625% | +0.0044 pp (t = +4.77) | 99.969% |

Both changes are a few thousandths of a percentage point, and the **sign
differs between the two recordings** — the direction is not even consistent,
which is what no real effect looks like. Vectors lost and gained roughly
cancel (planar 0.046% against 0.043% of the grid).

**What the third pass does change** is vector values, by ~1.0 mm/s planar
(both components) and 0.6–0.9 mm/s stereo, concentrated at small separations.
The second-order structure function of the 2-pass field is lower, in a dip
centred on the interrogation window:

| r (mm), planar | 0.41 | 0.82 | 1.65 (= W) | 3.30 | 6.59 | 13.18 |
|---|---|---|---|---|---|---|
| D11 ratio, 2-pass / 3-pass | 0.940 | 0.904 | **0.900** | 0.931 | 0.959 | 0.978 |

So the third pass adds content at and below the interrogation window, and the
fields converge at larger separations. Whether that content is real is
testable, because DaVis ran the same three-32px schedule: if the third pass
were converging on the true small-scale field it should move this app *toward*
the reference. It does the opposite.

| | 3 fine passes | 2 fine passes | closer to DaVis |
|---|---|---|---|
| Planar, mean\|diff\| U | 8.119 mm/s | 8.072 mm/s | 2 passes (t = −39.2) |
| Planar, mean\|diff\| V | 8.266 mm/s | 8.210 mm/s | 2 passes (t = −51.9) |
| Stereo, mean\|diff\| U | 8.555 mm/s | 8.399 mm/s | 2 passes (t = −99.5) |
| Stereo, mean\|diff\| V | 6.575 mm/s | 6.466 mm/s | 2 passes (t = −154.7) |
| Stereo, mean\|diff\| W | 8.544 mm/s | 8.397 mm/s | 2 passes (t = −110.0) |

Two passes agree better on all five components across both recordings, with
correlations marginally higher as well (planar U 0.9866 against 0.9865, stereo
W 0.9861 against 0.9857). The improvement is small — 0.6-0.7% planar, 1.7%
stereo — but consistent on essentially every pair.

**Throughput.** Back to back on the same 100 planar pairs and the same
machine: 7.57 s/pair at three fine passes against 4.85 s/pair at two, a
**35.9% saving**, close to what removing one of three fine passes predicts from
correlation work alone. The 2-pass stereo run measured 14.6 s/pair, but against
a 1000-pair production run rather than a matched control, so treat that one as
indicative.

The finding is therefore that **diverging from DaVis's pass schedule makes this
app's output agree slightly better with DaVis's output**, for about a third
less compute, at no cost in density. Two things argue against acting on it as a
default: these are two recordings of one flow from one facility, and DaVis is a
reference rather than ground truth — both programs could be smoothing away the
same real structure. The third pass may also earn its keep on higher-shear or
more sparsely seeded data, where window deformation needs more iterations to
converge. Synthetic images with a known displacement field would settle whether
the sub-window content the third pass adds is signal or noise. The one-sentence
summary in the meantime: the third fine pass is not buying accuracy on this
data, and it is not what sets vector density.

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
that do use a single scalar angle per camera. The stereo W agreement above
(per-pair corr 0.985, ensemble corr 0.990, ensemble means matching to the
reported precision) is the evidence that whatever the two approaches do
internally, they land in the same place on this rig.

**Stereo needs one manual, acquisition-time input DaVis's calibration
doesn't expose.** `StereoSettings.sheet_z_mm` — the real Z position of the
laser sheet for a specific recording, between the calibration's two
Z-planes — has no source in any DaVis calibration file; it's an
acquisition-time fact, not a calibration-time one. This app requires it as
an explicit parameter (or a manual GUI entry). It matters: on a sample
pair, corr(W) ranged from 0.981 to 0.993 depending on the chosen value
across a physically plausible 1 mm range, with mean\|diff\| swinging from
6.4 to 10.5 mm/s. Whoever ran the acquisition needs to supply this
correctly per recording; there is no way to derive it after the fact. (The
stereo numbers in this document use the midpoint of the two calibrated
planes, −0.5 mm.)

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

`compare_dataset.py` writes one checkpoint per pair and skips already-done
pairs on a rerun, so a full-dataset run survives interruption. It also
accepts `--start-index`/`--stride`, which shards a run across several
concurrent processes writing into one output directory — on this machine
12 shards cut the 1000-pair planar comparison from ~6.5 hours to ~50
minutes, since the per-pair cost is dominated by resampling one field onto
the other's grid. Run it once more with `--summarize-only` afterward to
rebuild the combined `summary.csv` from every checkpoint.

## Caveats

- All numbers above are from one planar and one stereo recording off one
  rig. They characterize agreement on this dataset, not a general accuracy
  guarantee for arbitrary lens distortion, seeding density, or flow
  character.
- The throughput comparison is not hardware-controlled — see above.
- The global-outlier-filter observation is a live open question, not a
  settled recommendation: the right threshold plausibly depends on how
  clean a given field already is, and this app's default is unchanged
  pending a broader sweep across more datasets. That measurement is from a
  bounded sample, unlike the 1000-pair figures elsewhere in this document.
- The sheet-Z sensitivity measurement is from a 2-pair sample; treat the
  specific correlation numbers as illustrative of the sensitivity's
  existence and rough size, not as a precise calibration curve.
- Per-pair correlations here come from `compare_dataset.py` (scattered-point
  `griddata` resampling); `make_comparison_plots.py` uses
  `RegularGridInterpolator` on the same data and lands within ~0.001 on
  correlation and ~0.15 mm/s on mean\|diff\|. Either is a fair reading; they
  are not identical pipelines.
