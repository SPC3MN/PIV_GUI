# Builds the Windows installer for PIV Suite (CPU backend).
#
# Two-stage build:
#   1. PyInstaller bundles the venv's piv_suite_gui app into a standalone
#      --onedir folder (installer\dist\PIV_Suite\) -- no Python install
#      needed on the target machine.
#   2. Inno Setup (ISCC.exe) wraps that folder into a single installer
#      exe (installer\Output\PIV_Suite_Setup_<version>.exe) with Start
#      Menu shortcuts and a proper uninstaller.
#
# Prerequisites (one-time):
#   - The project's own .venv, with `pip install -e .[gui]` already done
#   - pip install pyinstaller   (into that same .venv)
#   - Inno Setup 6: winget install --id JRSoftware.InnoSetup -e
#
# GPU backend (cupy + openpiv-python-gpu) is deliberately NOT bundled --
# see piv_suite.iss's header comment for why. Run this from the repo root
# or from installer\ directly; paths below are relative to this script's
# own location either way.

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

Write-Host "== Stage 1/2: PyInstaller bundle ==" -ForegroundColor Cyan
& $venvPyInstaller --noconfirm --windowed --name "PIV_Suite" `
    --distpath installer\dist --workpath installer\build --specpath installer `
    --exclude-module cupy --exclude-module cupyx --exclude-module cupy_backends --exclude-module openpiv_gpu `
    --exclude-module pytest --exclude-module pytest_qt --exclude-module openpiv.test `
    --paths src `
    --collect-submodules openpiv `
    --collect-data lvpyio `
    --collect-submodules piv_suite `
    --collect-submodules piv_suite_gui `
    "src\piv_suite_gui\app.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed (exit $LASTEXITCODE)" }

Write-Host "== Stage 2/2: Inno Setup installer ==" -ForegroundColor Cyan
& $iscc "installer\piv_suite.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit $LASTEXITCODE)" }

Write-Host "Done -- installer\Output\PIV_Suite_Setup_*.exe" -ForegroundColor Green
