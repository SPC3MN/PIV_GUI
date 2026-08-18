# Windows installer

Builds a standalone Windows installer for the PIV Suite GUI (CPU backend)
-- no Python install required on the target machine.

## One-time setup

```powershell
.venv\Scripts\python.exe -m pip install pyinstaller
winget install --id JRSoftware.InnoSetup -e
```

## Build

```powershell
installer\build_installer.ps1
```

Produces `installer\Output\PIV_Suite_Setup_<version>.exe`. Two stages:

1. **PyInstaller** bundles `piv_suite_gui.app:main` into a standalone
   `--onedir` folder (`installer\dist\PIV_Suite\`).
2. **Inno Setup** (`piv_suite.iss`) wraps that folder into a single
   installer exe with Start Menu shortcuts, an optional desktop
   shortcut, and a proper uninstaller. Installs per-user
   (`%LOCALAPPDATA%\Programs\PIV Suite`) -- no admin/UAC prompt.

Verified end-to-end on real Windows hardware: install, launch, Start
Menu/uninstall registry entries, and uninstall all confirmed clean.

## GPU backend is not bundled

`cupy` + `openpiv-python-gpu` are excluded from the PyInstaller build on
purpose:

- They need a matching CUDA toolkit/driver on the *target* machine
  anyway, so bundling them doesn't make the installer more portable --
  every install still needs GPU-specific setup.
- `cupy`'s CUDA math libraries are enormous when a system CUDA install is
  on the build machine's `PATH` -- an early attempt at this bundle leaked
  `cupy_backends`/`cupyx` and pulled in raw CUDA DLLs (`cublasLt64_13.dll`
  alone is 443MB), inflating the installer from ~90MB to over 1GB.

The installed app is CPU-only -- there's no venv/pip inside a frozen
PyInstaller bundle to add cupy/openpiv_gpu to. GPU support currently
still requires the source + `.venv` workflow from `INSTALL_WINDOWS.md`.
A GPU-enabled installer variant is a separate, not-yet-done effort (would
need a machine-specific CUDA version choice baked in at build time,
rather than "works on any Windows machine" like this one).
