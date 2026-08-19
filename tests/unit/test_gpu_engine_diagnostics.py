"""is_gpu_available()'s failure-diagnostic logging -- no real GPU/cupy
needed, this only tests the logging plumbing itself. Added after a real
"GPU stays greyed out" report turned out to be undebuggable: the old
is_gpu_available() swallowed every exception into a bare False, with no
way to tell "not installed" apart from "installed but broken" (the
latter is what was actually happening -- see gpu_engine.py's
is_gpu_available() docstring).
"""

import os

from piv_suite.engines import gpu_engine


def test_log_gpu_check_failure_writes_traceback(tmp_path, monkeypatch):
    log_path = tmp_path / "gpu_availability_check.log"
    monkeypatch.setattr(gpu_engine, "_gpu_check_log_path", lambda: str(log_path))

    try:
        raise RuntimeError("simulated cupy import failure")
    except RuntimeError:
        gpu_engine._log_gpu_check_failure("import cupy / openpiv_gpu")

    assert log_path.exists()
    content = log_path.read_text()
    assert "is_gpu_available() failed at: import cupy / openpiv_gpu" in content
    assert "RuntimeError: simulated cupy import failure" in content


def test_log_gpu_check_failure_overwrites_previous_log(tmp_path, monkeypatch):
    log_path = tmp_path / "gpu_availability_check.log"
    monkeypatch.setattr(gpu_engine, "_gpu_check_log_path", lambda: str(log_path))

    try:
        raise ValueError("first failure")
    except ValueError:
        gpu_engine._log_gpu_check_failure("step one")
    try:
        raise KeyError("second failure")
    except KeyError:
        gpu_engine._log_gpu_check_failure("step two")

    content = log_path.read_text()
    assert "first failure" not in content
    assert "step two" in content


def test_log_gpu_check_failure_never_raises_even_if_path_is_bad(monkeypatch):
    # Diagnostics are best-effort -- a logging failure (e.g. an
    # unwritable path) must not itself break the GPU availability check.
    monkeypatch.setattr(gpu_engine, "_gpu_check_log_path", lambda: "Z:\\definitely\\not\\a\\real\\drive\\log.txt")
    gpu_engine._log_gpu_check_failure("some step")  # must not raise


def test_gpu_check_log_path_uses_executable_dir_when_frozen(monkeypatch, tmp_path):
    fake_exe = tmp_path / "PIV_Suite.exe"
    monkeypatch.setattr(gpu_engine.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_engine.sys, "executable", str(fake_exe))
    path = gpu_engine._gpu_check_log_path()
    assert os.path.dirname(path) == str(tmp_path)
    assert os.path.basename(path) == "gpu_availability_check.log"


def test_is_gpu_available_never_raises():
    # Whatever this machine's actual GPU/cupy state is, the check itself
    # must always return a plain bool, never propagate an exception.
    result = gpu_engine.is_gpu_available()
    assert isinstance(result, bool)
