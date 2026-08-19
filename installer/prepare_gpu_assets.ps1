# Prepares installer\assets\python-embed\ -- a minimal, pip-capable
# Python 3.13 (matching the frozen app's own interpreter/ABI) bundled
# into the installer purely as a build-time tool for fetching GPU
# packages (cupy-cudaXXx, cuda-pathfinder, openpiv-python-gpu) at INSTALL
# time. Never run as the app itself -- just used by piv_suite.iss's
# [Code] section to run `pip install --target <app>\_internal`.
#
# Idempotent: skips work if python-embed already has pip. Re-run with
# -Force to rebuild from scratch (e.g. after a Python version bump).
#
# Why a bundled interpreter instead of hand-parsing PyPI's API in Inno
# Setup's Pascal Script: pip already resolves the correct cp313-win_amd64
# wheel and its dependencies correctly and robustly; hand-rolling that
# logic in Pascal Script would be far more fragile to maintain.

param([switch]$Force)

$ErrorActionPreference = "Stop"
$assetsDir = Join-Path $PSScriptRoot "assets"
$embedDir = Join-Path $assetsDir "python-embed"
$pythonVersion = "3.13.9"  # embeddable patch version -- doesn't need to
                            # exactly match the venv's 3.13.14; both are
                            # cp313 ABI, which is all pip's wheel
                            # selection at install time actually needs.

if ((Test-Path (Join-Path $embedDir "Scripts\pip.exe")) -and -not $Force) {
    Write-Host "installer\assets\python-embed already prepared (pip present) -- skipping. Use -Force to rebuild." -ForegroundColor Yellow
    exit 0
}

if (Test-Path $embedDir) { Remove-Item -Recurse -Force $embedDir }
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

$zipPath = Join-Path $assetsDir "python-embed-download.zip"
Write-Host "Downloading Python $pythonVersion embeddable package..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip" -OutFile $zipPath

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $embedDir -Force
Remove-Item $zipPath

# Enable site-packages (disabled by default in embeddable builds) so pip
# and installed packages are actually importable.
$pthFile = Get-ChildItem $embedDir -Filter "python3*._pth" | Select-Object -First 1
(Get-Content $pthFile.FullName) -replace '#import site', 'import site' | Set-Content $pthFile.FullName

$getPipPath = Join-Path $assetsDir "get-pip.py"
Write-Host "Bootstrapping pip..." -ForegroundColor Cyan
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
& (Join-Path $embedDir "python.exe") $getPipPath --no-warn-script-location
Remove-Item $getPipPath

Write-Host "Done -- installer\assets\python-embed ready ($(([math]::Round((Get-ChildItem $embedDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1))) MB)" -ForegroundColor Green
