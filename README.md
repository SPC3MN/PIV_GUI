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
- `piv_suite/calibration/` -- DaVis polynomial camera mapping/dewarping
  (`camera_mapping.py`), two-camera 3-component reconstruction
  (`reconstruction.py`), and a stub for automated DaVis-calibration-report
  parsing (`report_parser.py`, not yet implemented -- calibration is
  entered manually via form fields for now).
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
- `piv_suite_gui/` -- the PySide6 desktop GUI (project/settings/
  calibration/preview/run panels, a `QThread` batch worker).

## Known limitations

- Stereo loose-folder ingestion (`io/loose_files.py`'s
  `iter_stereo_from_loose_files`) is still `.im7`/lvpyio-only, inherited
  unchanged from the original repos -- unlike planar loose ingestion, it
  hasn't been generalized to generic image formats yet.
- `calibration/report_parser.py` (parsing a DaVis calibration report
  directly instead of manual coefficient entry) is a stub.
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
