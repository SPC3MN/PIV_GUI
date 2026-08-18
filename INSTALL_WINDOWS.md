# Installing piv_suite on Windows

This covers a clean Windows 10/11 machine with nothing pre-installed.
Windows is actually the easiest target for this project — `lvpyio` (DaVis
`.set`/`.im7` support) and the GPU backend (CUDA/cupy) both ship official
Windows builds, unlike macOS.

## 1. Get the code onto the machine

Pick whichever is easiest for you:

- **Git clone** (if the repo is pushed to GitHub/GitLab/etc.):
  ```powershell
  git clone <your-repo-url> piv_suite
  cd piv_suite
  ```
- **Copy the folder** — zip up the `piv_suite` folder (from the Mac it was
  built on) and copy it over via USB drive, network share, or a cloud
  drive, then unzip it on the Windows machine.

## 2. Install Python

1. Go to <https://www.python.org/downloads/windows/> and download the
   latest Python 3.11 or 3.12 installer (64-bit). Avoid the Microsoft
   Store version — it has path/permission quirks that cause problems with
   some scientific packages.
2. Run the installer. **Check "Add python.exe to PATH"** at the bottom of
   the first screen before clicking Install — this is the single most
   common thing people forget.
3. Verify it worked — open **PowerShell** (Start menu → type `powershell`)
   and run:
   ```powershell
   python --version
   pip --version
   ```
   Both should print version numbers. If `python` isn't recognized, log
   out and back in (PATH changes need a fresh shell) or reinstall with the
   PATH box checked.

## 3. Create a virtual environment

From PowerShell, `cd` into the `piv_suite` folder, then:

```powershell
cd path\to\piv_suite
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get an error like *"running scripts is disabled on this system"*,
PowerShell's execution policy is blocking the activation script. Fix it
once (per user, doesn't need admin rights):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

then re-run `.venv\Scripts\Activate.ps1`. Your prompt should now start
with `(.venv)`.

## 4. Install piv_suite

**CPU-only (recommended to start — works on any machine, no GPU needed):**

```powershell
pip install --upgrade pip
pip install -e ".[gui]"
```

This pulls in numpy, scipy, matplotlib, openpiv, lvpyio, and PySide6. It
can take a few minutes the first time.

**If you have an NVIDIA GPU and want GPU acceleration too**, additionally:

1. Install the CUDA Toolkit matching your GPU driver from
   <https://developer.nvidia.com/cuda-downloads> (11.2–11.8 or 12.x — check
   `nvidia-smi` in PowerShell for your driver's supported CUDA version
   first).
2. Install the matching cupy wheel, e.g. for CUDA 12.x:
   ```powershell
   pip install cupy-cuda12x
   ```
3. `openpiv-python-gpu` isn't on PyPI — clone it and install it into the
   same environment:
   ```powershell
   git clone https://github.com/OpenPIV/openpiv-python-gpu.git ..\openpiv-python-gpu
   pip install -e ..\openpiv-python-gpu
   ```
   (If you don't have `git`, download the ZIP from GitHub's green "Code"
   button instead, extract it, and `pip install -e` that folder.)

You can always start CPU-only and add GPU support later — the GUI grays
out the GPU option automatically if it isn't set up yet, instead of
crashing.

## 5. Run it

With the venv activated (prompt shows `(.venv)`):

```powershell
piv-suite-gui
```

That launches the desktop app. For the command-line version instead:

```powershell
piv-suite my_project.pivproj --backend cpu --mode planar --input-path .\data
```

(`--backend gpu` once CUDA/cupy/openpiv-python-gpu are set up per step 4.)

## 6. Next time

You don't need to reinstall anything — just re-activate the environment
each time you open a new PowerShell window:

```powershell
cd path\to\piv_suite
.venv\Scripts\Activate.ps1
piv-suite-gui
```

## Troubleshooting

- **`pip install` fails compiling something**: some packages (openpiv,
  scipy) occasionally need a C/C++ compiler if no prebuilt wheel matches
  your exact Python version. Install "Microsoft C++ Build Tools" from
  <https://visualstudio.microsoft.com/visual-cpp-build-tools/> (select the
  "Desktop development with C++" workload), then retry `pip install`.
- **`piv-suite-gui` not found after install**: make sure the venv is
  activated (`(.venv)` visible in the prompt) — console scripts are
  installed inside the venv, not system-wide.
- **GUI window doesn't appear / errors about a display/platform plugin**:
  make sure you're running this on the actual Windows desktop (not over a
  headless SSH session) — PySide6 needs a real display.
- **`lvpyio` import errors when reading `.set`/`.im7` files**: double
  check you installed with the `[gui]` extra (or just `pip install
  lvpyio` directly) and that your Python version is 3.9–3.13 (check
  lvpyio's PyPI page for the current supported range).
