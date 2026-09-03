# piv_suite

A unified PIV (Particle Image Velocimetry) processing suite -- planar and
stereo, CPU and GPU, driven by a desktop GUI or a single CLI -- built on
top of [OpenPIV](https://github.com/OpenPIV/openpiv-python) and
[openpiv-python-gpu](https://github.com/OpenPIV/openpiv-python-gpu).

This consolidates four previously-separate, script-based repos
(`Stereo_PIV_GPU`, `Planar_PIV_GPU`, `Planar_PIV_CPU`, `Stereo_PIV_CPU`)
into one package with pluggable GPU/CPU and planar/stereo backends behind
a shared interface, plus a new GUI layer. See `docs/` (or the project
history) for the full design rationale.

## Install

```bash
pip install -e .            # core (CPU backend, CLI only)
pip install -e ".[gui]"     # + desktop GUI (PySide6)
pip install -e ".[gpu]"     # + GPU backend deps (see note below)
```

GPU support additionally requires a CUDA-capable NVIDIA GPU, the matching
CUDA Toolkit, a `cupy-cuda*` wheel for your CUDA version, and
`openpiv-python-gpu` cloned separately (not on PyPI) and added to your
`PYTHONPATH` -- see that project's own install instructions. The CPU
backend and GUI work fully without any of this.

## Quickstart

**CLI** (reproduces what the four original scripts did, unified behind
one command):

```bash
piv-suite my_project.pivproj --backend cpu --mode planar --input-path ./data
piv-suite my_project.pivproj --backend gpu --mode stereo
```

If `my_project.pivproj` doesn't exist yet, it's created with defaults on
first run (same UX as the original scripts' JSON config files) -- edit it
and re-run to tune settings.

**GUI**:

```bash
piv-suite-gui
```

Set the input (a `.set` DaVis project, a folder of `.set` files to batch,
or a folder of labeled image pairs -- `.im7`, TIFF, or PNG), pick
planar/stereo and CPU/GPU, tune the multi-pass window schedule and
post-processing filters, click **Preview first pair** to sanity-check
before committing, then **Run batch**.

## Migrating an old config

If you have a tuned `stereo_piv_config.json` / `planar_piv_config.json` /
`planar_cpu_piv_config.json` / `stereo_cpu_piv_config.json` from one of
the four original repos:

```bash
python scripts/migrate_legacy_config.py old_config.json new_config.pivproj
```

Backend and mode are auto-detected from which keys are present.

## Package layout

- `piv_suite/io/` -- `.set` DaVis project ingestion (`davis_set.py`,
  lvpyio-backed) and loose-folder ingestion (`loose_files.py`, supporting
  `.im7` plus generic TIFF/PNG labeled image pairs via `readers.py`).
- `piv_suite/calibration/` -- both of DaVis's internal calibration models,
  decoded exactly from a project's own `Calibration.xml`: the 3rd-order
  polynomial mapping (`camera_mapping.py`) and the OpenCV pinhole one
  (`pinhole.py`). Also two-camera 3-component reconstruction
  (`reconstruction.py`).

  Stereo triangulation angles are DERIVED PER CORRELATION POINT from the
  calibration (`stereo_view_angles`), not entered by hand and not a single
  scalar per camera: the real viewing angle varies several degrees across a
  stereo field of view, and collapsing it to one number measurably corrupts
  the in-plane components. `StereoSettings.alpha1_deg`/`alpha2_deg`/
  `beta1_deg`/`beta2_deg` remain as an override for a calibration that
  genuinely cannot supply the geometry (a single calibrated Z-plane, or a
  hand-entered mapping).
- `piv_suite/engines/` -- the CPU (`cpu_engine.py`, plain openpiv-python)
  and GPU (`gpu_engine.py`, openpiv-python-gpu + tiling for large frames)
  backends behind a shared `PIVEngine` Protocol (`base.py`), selected via
  `registry.py`.
- `piv_suite/processing/` -- shared post-processing (`postprocess.py`:
  standard-deviation spurious-vector filter, range/local-residual filter,
  invalid-vector interpolation, smoothing, calibration) and the per-pair
  pipeline (`pipeline.py`).
- `piv_suite/config/` -- the canonical settings schema (`schema.py`),
  JSON round-trip (`io.py`), and adapters translating the canonical
  schema to/from each backend's native settings vocabulary
  (`legacy.py`).
- `piv_suite/plotting/` -- quiver plotting (planar/stereo) and in-memory
  preview figures for the GUI.
- `piv_suite/cli/` -- the unified command-line entry point.
- `piv_suite_gui/` -- the PySide6 desktop GUI. The left rail carries the
  everyday flow top to bottom (Source, Physical units, Output, Camera
  calibration, then the processing settings); the right side is Preview and
  Run. Settings that are real but rarely touched -- hand-entered calibration
  coefficients, algorithm method pickers, GPU tiling, per-pass internals,
  worker count -- live in ONE consolidated "Advanced" disclosure at the
  bottom of the left rail rather than several scattered ones competing with
  the flow. Batch runs happen on a `QThread` worker.

## Known limitations

- Stereo loose-folder ingestion (`io/loose_files.py`'s
  `iter_stereo_from_loose_files`) is still `.im7`/lvpyio-only, inherited
  unchanged from the original repos -- unlike planar loose ingestion, it
  hasn't been generalized to generic image formats yet.
- Per-vector correlation-plane uncertainty quantification is explicitly
  out of scope for this program -- post-processing here means
  displacement-range/residual and standard-deviation-based spurious-
  vector rejection, not uncertainty estimates.
- No installer/packaged binary yet -- run from a Python environment.

## Testing

```bash
pip install -e ".[dev,gui]"
pytest tests/unit
```

Unit tests cover the pure post-processing/calibration/config-adapter
functions plus GUI panel construction and wiring (`pytest-qt`, run
headlessly with `QT_QPA_PLATFORM=offscreen`). There's no CI-tracked golden
fixture set yet from real experimental data -- see the project history's
verification notes for the synthetic end-to-end checks (planar CPU on a
known-shift synthetic image pair; full stereo dewarp -> correlate ->
reconstruct chain against a known synthetic 3-component displacement) that
validated this consolidation against the original four scripts' behavior.
