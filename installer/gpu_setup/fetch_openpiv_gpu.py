"""Install-time helper, run by the bundled embeddable Python (see
installer/prepare_gpu_assets.ps1) from piv_suite.iss's [Code] section --
never part of the shipped app itself.

Extracts the openpiv-python-gpu GitHub archive zip and copies just its
openpiv_gpu/ package (the piece piv_suite's gpu_engine.py actually
imports -- see engines/gpu_engine.py's module docstring) into the
installed app's _internal folder, where PyInstaller's onedir bundle looks
for importable packages (confirmed on real hardware: dropping a package
folder there makes it importable exactly like the app's own bundled
scipy/numpy).

Usage: python fetch_openpiv_gpu.py <zip_path> <dest_internal_dir>
"""
import sys
import zipfile
import shutil
import os


def main():
    zip_path, dest_dir = sys.argv[1], sys.argv[2]
    extract_dir = zip_path + "_extracted"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if not names:
            print("ERROR: openpiv-python-gpu archive is empty", file=sys.stderr)
            return 1
        root = names[0].split("/")[0]
        z.extractall(extract_dir)

    src = os.path.join(extract_dir, root, "openpiv_gpu")
    if not os.path.isdir(src):
        print(f"ERROR: openpiv_gpu/ not found in archive (looked in {src})", file=sys.stderr)
        return 1

    dst = os.path.join(dest_dir, "openpiv_gpu")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"openpiv_gpu installed to {dst}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
