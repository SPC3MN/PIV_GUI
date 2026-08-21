# Windows installer

Builds a standalone Windows installer for the PIV Suite GUI -- no Python
install required on the target machine. The wizard lets the user opt
into GPU (CUDA) support at install time.

## One-time setup

```powershell
.venv\Scripts\python.exe -m pip install pyinstaller
winget install --id JRSoftware.InnoSetup -e
```

## Build

```powershell
installer\build_installer.ps1
```

Produces `installer\Output\PIV_Suite_Setup_<version>.exe`. Three stages
(the script runs all of them):

1. **`prepare_gpu_assets.ps1`** downloads and preps a minimal pip-capable
   Python (`installer\assets\python-embed\`) -- an install-time TOOL used
   only to fetch GPU packages, never run as the app itself. Skipped on
   rebuilds once already prepared.
2. **PyInstaller** bundles `piv_suite_gui.app:main` into a standalone
   `--onedir` folder (`installer\dist\PIV_Suite\`).
3. **Inno Setup** (`piv_suite.iss`) wraps that folder into a single
   installer exe with Start Menu shortcuts, an optional desktop
   shortcut, and a proper uninstaller. Installs per-user
   (`%LOCALAPPDATA%\Programs\PIV Suite`) -- no admin/UAC prompt.

Verified end-to-end on real Windows hardware: install, launch, Start
Menu/uninstall registry entries, and uninstall all confirmed clean, in
both CPU-only and GPU-enabled configurations.

## GPU support: chosen in the installer, fetched at install time

The installer wizard asks: CPU only, or NVIDIA GPU with CUDA 11.x /
12.x / 13.x. If a CUDA version is picked, the installer downloads the
matching `cupy-cudaXXx` wheel + `cuda-pathfinder` (via the bundled
embeddable Python's pip) and `openpiv-python-gpu`'s source, and drops
them into the installed app's `_internal` folder -- confirmed on real
hardware that PyInstaller's onedir bundle imports packages placed there
exactly like its own bundled scipy/numpy, so this makes the GPU backend
fully functional without needing a real venv inside the frozen app.

**Why download instead of bundling GPU packages into the installer
file itself:**

- `cupy`'s own wheel is small (~35MB) because it does NOT bundle CUDA's
  math libraries -- it uses `cuda-pathfinder` to *locate* an existing
  NVIDIA CUDA Toolkit on the machine at runtime (confirmed: the
  installed `cupy_backends` directory contains no DLLs of its own). Pre-
  baking all 3 CUDA variants into the installer file would still balloon
  it for no real benefit, since the wheel must exactly match the frozen
  app's own `cpXY-win_amd64` ABI regardless (see
  `prepare_gpu_assets.ps1`'s `$pythonVersion`, which must track whatever
  Python builds the frozen app -- `cp311` as of v0.2.9, was `cp313` on
  an earlier dev machine's venv), which pip resolves reliably -- a
  hand-rolled PyPI-API parser in Inno Setup's Pascal Script would not
  (this is also why a bundled Python runs the actual fetch, rather than
  Inno Setup's own scripting).
- This means: GPU setup needs internet access *at install time*, and the
  target machine still needs the real NVIDIA CUDA Toolkit installed
  separately, matching the chosen version (developer.nvidia.com/cuda-downloads)
  -- this installer cannot provide that itself (multi-GB, its own
  license/installer). The wizard page says so.
- cupy/cuda-pathfinder/openpiv_gpu are NOT baked into the frozen
  PyInstaller build at all (`--exclude-module cupy` etc. in
  `build_installer.ps1`'s PyInstaller stage) -- an earlier attempt at
  bundling them directly leaked `cupy_backends`/`cupyx` and pulled in raw
  CUDA DLLs from the build machine's system CUDA install
  (`cublasLt64_13.dll` alone is 443MB), inflating the installer from
  ~90MB to over 1GB. The install-time-download approach avoids that
  entirely -- only the small Python packages get fetched.

For scripted/silent deployments, `/GPUCUDA=11|12|13` on the installer's
command line selects a CUDA version without showing the wizard page
(omit it, or CPU-only stays the default under `/VERYSILENT`).

## Debugging a failed GPU setup

`{app}\gpu_setup_log.txt` (e.g. `%LOCALAPPDATA%\Programs\PIV Suite\
gpu_setup_log.txt`) records every GPU setup step's captured stdout/
stderr, written whether setup succeeds or fails -- check it first if the
GPU option stays greyed out after a CUDA version was selected. It's
regenerated on every install-time GPU setup attempt (re-running the
installer overwrites it) and removed on uninstall.

**The actual root cause of "packages installed successfully but GPU
still greyed out"** turned out to be a separate bug from all of the
above, in the PyInstaller build itself, not the install-time fetch:
`cupy` imports the stdlib module `graphlib` internally, but since `cupy`
is `--exclude-module`'d from the frozen build (fetched separately at
install time, see the header comment above), PyInstaller's static
analysis never saw that dependency and didn't bundle `graphlib` --
`import cupy` then failed inside the frozen app with `ModuleNotFoundError:
No module named 'graphlib'`, even though the exact same cupy install
worked fine imported from a normal venv. `is_gpu_available()`
(`engines/gpu_engine.py`) silently swallowed this into a bare `False`
with no visible error anywhere, which is exactly why this was hard to
diagnose -- it's now fixed with `--hidden-import=graphlib` in
`build_installer.ps1`, and `is_gpu_available()` now writes the real
exception to `{app}\gpu_availability_check.log` on any failure instead
of swallowing it silently, so a similar issue wouldn't be a mystery
again. The same PyInstaller gap (something needed at runtime that static
analysis couldn't see because the importing package wasn't part of the
frozen build) also caused a separate, CPU-only bug: `imageio` (pulled in
via `openpiv` -> `scikit-image`) calls `importlib.metadata.version
('imageio')` on itself at import time, which failed with "No package
metadata was found for imageio" since PyInstaller doesn't bundle a
package's `.dist-info` folder unless told to -- fixed with
`--copy-metadata=imageio`.

Two more real bugs were found and fixed by testing the install-time
fetch path repeatedly on real hardware, worth knowing if a similar issue
resurfaces:

- **Extraction can hit a transient file lock.** Antivirus real-time
  scanning a just-downloaded file (the openpiv-python-gpu zip) can
  briefly lock it, making the immediate extraction attempt fail with a
  permission error even though nothing is actually wrong -- confirmed by
  hitting this manually multiple times while developing this feature.
  `InstallGpuPackages` now retries extraction up to 3 times with a 2s
  delay between attempts before giving up.
- **A custom `MsgBox` blocks forever on a silent/unattended install.**
  `/SUPPRESSMSGBOXES` only suppresses Inno Setup's own built-in dialogs,
  not custom `MsgBox` calls from `[Code]` -- with no one there to click
  it, a silent run (or a scripted `/GPUCUDA=` deployment) would hang
  indefinitely waiting for input that never comes. All GPU-setup
  messages now go through `GpuMsgBox`, which checks `WizardSilent` first
  and skips the dialog on unattended runs -- the log file already has
  the same information, so nothing is lost.

Stress-tested with 5 consecutive silent `/GPUCUDA=13` installs after
these fixes: all 5 succeeded cleanly with no retries needed and no
hangs.

**`lvpyio` (the DaVis `.set`-file reader) needs `--collect-all`, not just
`--collect-data`.** A user hit "No module named 'lvpyio.io.paths'"
previewing a real `.set` project in the frozen app -- same root-cause
class as the graphlib/imageio bugs above (something the frozen app needs
at runtime that PyInstaller's static analysis couldn't see), but with a
twist: `lvpyio.io`'s own submodules are almost entirely compiled `.pyd`
extensions (`paths`, `type_mapper`, `attribute`, `component`, `grid`,
`plane`, `scale`, ...), and several of them are never `import`ed from
ANY visible `.py` source in the package at all -- `lvpyio/io/__init__.py`
only imports `read_buffer`, `write_buffer`, `set`, `particles`. The rest
get pulled in at runtime from *inside* another compiled extension (e.g.
`set.pyd` internally importing `lvpyio.io.paths`), which PyInstaller
cannot discover by scanning Python source, and there's no single
`--hidden-import` name to add the way there was for graphlib -- it isn't
knowable in advance which `.pyd` a given `.set` file's code path will
need. Fixed by switching `--collect-data lvpyio` to `--collect-all
lvpyio` in `build_installer.ps1`, which walks the installed package
directory instead of scanning imports, so every submodule PyInstaller
can find on disk is bundled regardless of whether anything visibly
imports it -- closes off the whole class of "some .set files trip a
code path needing a .pyd nothing else visibly imports" bug at once,
rather than adding names to a hidden-import list one crash report at a
time.

Verified against the real `.set` project that surfaced this, two ways:
`paths.cp313-win_amd64.pyd` and `type_mapper.cp313-win_amd64.pyd` (the
two submodules nothing in `lvpyio`'s own source ever visibly imports)
are now present in `_internal\lvpyio\io\`, where they were missing
before; and, since PyInstaller happens to keep this particular package
entirely as loose files rather than splitting it between the PYZ
archive and disk (confirmed by diffing the file list against the source
venv's `lvpyio` install -- unlike numpy/cupy, which DO split, and where
testing via a separate interpreter gave a misleading result earlier in
this project's life), the frozen build's own `lvpyio` could be pointed
at directly from a normal interpreter and actually exercised against
the reporting user's real data: `list_pair_ids_from_set`/
`get_pair_from_set` (the exact functions the GUI's preview panel calls)
successfully read all 200 pairs of `Final_Self_Calibration.set`.
