"""Sanity tests for perf/autotune.py -- the only two things this
codebase lets vary by hardware (see engines/_openpiv_speedups.py's
module docstring for why everything else is an unconditional patch).
Not testing exact numbers (those are machine-dependent by design) --
testing the INVARIANTS: sensible types/ranges, monotonic response to
RAM/worker-count, and that an explicit override always wins.

The whole point of this module (see its own docstring) is that the
installer's frozen app has to run correctly on whatever machine it's
installed on -- not just the machine it happened to be built on. That's
tested explicitly below (test_low_spec_target_machines_get_safe_values)
by simulating a range of target-machine hardware profiles against the
real recommend_* functions together, not just each in isolation.
"""

import pytest

import piv_suite.perf.autotune as autotune


def test_available_ram_bytes_returns_positive_int():
    assert isinstance(autotune.available_ram_bytes(), int)
    assert autotune.available_ram_bytes() > 0


def test_recommended_chunk_size_within_clamps():
    for window_shape in ((16, 16), (32, 32), (64, 64), (128, 128)):
        chunk = autotune.recommended_chunk_size(window_shape)
        assert autotune._MIN_CHUNK_WINDOWS <= chunk <= autotune._MAX_CHUNK_WINDOWS


def test_recommended_chunk_size_shrinks_for_larger_windows():
    # Bigger windows -> more bytes/window -> fewer windows fit the same
    # cache-sized target.
    small = autotune.recommended_chunk_size((16, 16))
    large = autotune.recommended_chunk_size((128, 128))
    assert large <= small


def test_recommended_chunk_size_respects_low_ram_ceiling(monkeypatch):
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: 100 * 1024 * 1024)  # 100 MB
    chunk = autotune.recommended_chunk_size((64, 64))
    assert chunk == autotune._MIN_CHUNK_WINDOWS  # clamped to the floor, not the cache target


def test_recommended_pipeline_chunk_size_within_clamps():
    for window_shape in ((16, 16), (32, 32), (64, 64)):
        chunk = autotune.recommended_pipeline_chunk_size(window_shape, assumed_concurrent_workers=8)
        assert autotune._MIN_PIPELINE_CHUNK_WINDOWS <= chunk <= autotune._MAX_PIPELINE_CHUNK_WINDOWS


def test_recommended_pipeline_chunk_size_much_larger_than_fft_chunk_size():
    # The regression this module's docstring documents: the pipeline
    # (outer, per-worker-memory) target must be well above the FFT
    # (inner, cache-locality) target, not the same value or smaller.
    fft_chunk = autotune.recommended_chunk_size((32, 32))
    pipeline_chunk = autotune.recommended_pipeline_chunk_size((32, 32), assumed_concurrent_workers=8)
    assert pipeline_chunk > fft_chunk


def test_recommended_pipeline_chunk_size_shrinks_with_more_workers():
    few_workers = autotune.recommended_pipeline_chunk_size((32, 32), assumed_concurrent_workers=2)
    many_workers = autotune.recommended_pipeline_chunk_size((32, 32), assumed_concurrent_workers=48)
    assert many_workers <= few_workers


def test_recommended_workers_override_always_wins():
    assert autotune.recommended_workers(n_override=1) == 1
    assert autotune.recommended_workers(n_override=7) == 7
    assert autotune.recommended_workers(n_override=0) == 1  # clamped to at least 1


def test_recommended_workers_auto_does_not_exceed_cpu_count(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert autotune.recommended_workers() <= 8


def test_recommended_workers_auto_is_at_least_one_even_on_low_ram(monkeypatch):
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: 0)
    assert autotune.recommended_workers() >= 1


GB = 1024**3


@pytest.mark.parametrize("name,cpu_count,ram_bytes,max_expected_workers", [
    ("tiny VM (1 core, 2GB)", 1, 2 * GB, 1),
    ("old dual-core laptop (2 cores, 4GB)", 2, 4 * GB, 2),
    ("budget laptop (4 cores, 8GB)", 4, 8 * GB, 4),
    ("mid workstation (8 cores, 16GB)", 8, 16 * GB, 8),
    ("this dev box (48 threads, 186GB)", 48, 186 * GB, 48),
])
def test_low_spec_target_machines_get_safe_values(monkeypatch, name, cpu_count, ram_bytes, max_expected_workers):
    """The installer's frozen app runs on whatever hardware a user
    installs it on, not the machine that built it -- this simulates a
    range of target-machine profiles (including much smaller than this
    dev box) against recommended_workers() and
    recommended_pipeline_chunk_size() TOGETHER (the pipeline chunk size
    depends on the chosen worker count), and checks every profile lands
    on safe, sane values: never zero/negative workers, never a worker
    count exceeding the machine's own core count, and a pipeline chunk
    size that's neither degenerately tiny nor unboundedly huge."""
    monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: ram_bytes)

    workers = autotune.recommended_workers()
    assert 1 <= workers <= max_expected_workers

    for window_shape in ((32, 32), (64, 64)):
        pipeline_chunk = autotune.recommended_pipeline_chunk_size(window_shape)
        assert autotune._MIN_PIPELINE_CHUNK_WINDOWS <= pipeline_chunk <= autotune._MAX_PIPELINE_CHUNK_WINDOWS
        fft_chunk = autotune.recommended_chunk_size(window_shape)
        assert autotune._MIN_CHUNK_WINDOWS <= fft_chunk <= autotune._MAX_CHUNK_WINDOWS


def test_recommended_workers_caps_auto_detect_at_windows_limit(monkeypatch):
    # Real crash reproduced on an actual 72-logical-core dual-socket
    # Windows machine: os.cpu_count()=72 with abundant RAM made the
    # auto-detect path return 72, which ProcessPoolExecutor(max_workers=72)
    # rejects outright on win32 (ValueError: max_workers must be <= 61) --
    # recommended_workers() must never hand back more than the platform
    # allows, regardless of how many cores/how much RAM the machine has.
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("os.cpu_count", lambda: 72)
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: 512 * GB)  # rule out the RAM ceiling binding
    assert autotune.recommended_workers() == 61


def test_recommended_workers_caps_explicit_override_at_windows_limit(monkeypatch):
    # An override is meant to win over the auto-detected value, but not
    # over a hard platform ceiling -- a user-typed n_workers=72 in a
    # .pivproj on this same real machine would otherwise crash identically,
    # just later (mid-batch, inside ProcessPoolExecutor's constructor)
    # instead of at startup.
    monkeypatch.setattr("sys.platform", "win32")
    assert autotune.recommended_workers(n_override=72) == 61
    assert autotune.recommended_workers(n_override=61) == 61  # right at the limit: untouched
    assert autotune.recommended_workers(n_override=8) == 8  # well under: untouched


def test_recommended_workers_no_windows_cap_off_windows(monkeypatch):
    # The 61-worker ceiling is a Windows-specific WaitForMultipleObjects
    # handle limit (see _max_windows_workers's docstring) -- must not
    # apply on other platforms, where fork-based multiprocessing has no
    # such constraint.
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("os.cpu_count", lambda: 128)
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: 512 * GB)
    assert autotune.recommended_workers() == 128
    assert autotune.recommended_workers(n_override=100) == 100


def test_very_low_ram_forces_serial_fallback(monkeypatch):
    # Below this module's _MIN_FREE_RAM_RESERVE_BYTES reserve, workers
    # must clamp to 1 (the CLI/GUI's n_workers<=1 check then routes
    # through the original, unmodified serial loop instead of a worker
    # pool -- see processing.parallel_planar's module docstring) rather
    # than dividing by zero or going negative.
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(autotune, "available_ram_bytes", lambda: 1 * GB)  # well under the 4GB reserve
    assert autotune.recommended_workers() == 1
