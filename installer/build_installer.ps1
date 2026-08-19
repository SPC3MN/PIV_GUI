# Builds the Windows installer for PIV Suite (CPU backend).
#
# Three-stage build:
#   1. prepare_gpu_assets.ps1 bundles a minimal pip-capable Python
#      (installer\assets\python-embed\) -- an install-time TOOL for
#      fetching cupy/openpiv_gpu, never run as the app itself. Skipped if
#      already prepared.
#   2. PyInstaller bundles the venv's piv_suite_gui app into a standalone
#      --onedir folder (installer\dist\PIV_Suite\) -- no Python install
#      needed on the target machine.
#   3. Inno Setup (ISCC.exe) wraps that folder into a single installer
#      exe (installer\Output\PIV_Suite_Setup_<version>.exe) with Start
#      Menu shortcuts, an uninstaller, and an optional GPU-package
#      install step (see piv_suite.iss).
#
# Prerequisites (one-time):
#   - The project's own .venv, with `pip install -e .[gui]` already done
#   - pip install pyinstaller   (into that same .venv)
#   - Inno Setup 6: winget install --id JRSoftware.InnoSetup -e
#
# The CPU/GUI app itself is bundled as-is; cupy/openpiv-python-gpu are
# NOT baked into the installer file -- the installer downloads them at
# INSTALL time if the user opts into a CUDA version, since they need to
# match cp313-win_amd64 anyway and bundling all CUDA variants would
# balloon the installer file to gigabytes. See piv_suite.iss's header
# comment for the full reasoning. Run this from the repo root or from
# installer\ directly; paths below are relative to this script's own
# location either way.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPyInstaller = Join-Path $repoRoot ".venv\Scripts\pyinstaller.exe"
if (-not (Test-Path $venvPyInstaller)) {
    throw "pyinstaller not found in .venv -- run: .venv\Scripts\python.exe -m pip install pyinstaller"
}

$iscc = Get-ChildItem -Path "C:\Program Files*", "$env:LOCALAPPDATA\Programs" `
    -Filter "ISCC.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
if (-not $iscc) {
    throw "ISCC.exe (Inno Setup) not found -- run: winget install --id JRSoftware.InnoSetup -e"
}

Write-Host "== Stage 1/3: GPU install-time assets ==" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "prepare_gpu_assets.ps1")
if ($LASTEXITCODE -ne 0) { throw "prepare_gpu_assets.ps1 failed (exit $LASTEXITCODE)" }

# --hidden-import=graphlib: cupy imports this stdlib module internally
# (cupy._core._scalar -> ... -> graphlib), but PyInstaller's static
# analysis can't see that need since cupy itself is --exclude-module'd
# from this build (fetched separately at install time, see piv_suite.iss)
# -- confirmed on real hardware: without this, cupy fails with
# "ModuleNotFoundError: No module named 'graphlib'" INSIDE the frozen
# app even though the exact same cupy install works fine from a normal
# venv, and is_gpu_available() silently reports False with no visible
# error (see gpu_engine.py's is_gpu_available() for the diagnostic log
# that surfaced this).
#
# --copy-metadata=imageio: imageio/__init__.py calls
# importlib.metadata.version('imageio') on itself at import time to set
# its own __version__ -- PyInstaller doesn't bundle a package's
# .dist-info metadata folder by default unless told to, so this raised
# "No package metadata was found for imageio" the first time the CPU
# preview path (which imports openpiv -> scikit-image -> imageio) ran
# in the frozen app, despite working fine unfrozen.
Write-Host "== Stage 2/3: PyInstaller bundle ==" -ForegroundColor Cyan
& $venvPyInstaller --noconfirm --windowed --name "PIV_Suite" `
    --distpath installer\dist --workpath installer\build --specpath installer `
    --exclude-module cupy --exclude-module cupyx --exclude-module cupy_backends --exclude-module openpiv_gpu `
    --exclude-module pytest --exclude-module pytest_qt --exclude-module openpiv.test `
    --hidden-import=graphlib `
    --copy-metadata=imageio `
    --paths src `
    --collect-submodules openpiv `
    --collect-data lvpyio `
    --collect-submodules piv_suite `
    --collect-submodules piv_suite_gui `
    "src\piv_suite_gui\app.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed (exit $LASTEXITCODE)" }

Write-Host "== Stage 3/3: Inno Setup installer ==" -ForegroundColor Cyan
& $iscc "installer\piv_suite.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit $LASTEXITCODE)" }

Write-Host "Done -- installer\Output\PIV_Suite_Setup_*.exe" -ForegroundColor Green
