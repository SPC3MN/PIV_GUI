"""Sanity tests for perf/autotune.py -- the only two things this
codebase lets vary by hardware (see engines/_openpiv_speedups.py's
module docstring for why everything else is an unconditional patch).
Not testing exact numbers (those are machine-dependent by design) --
testing the INVARIANTS: sensible types/ranges, monotonic response to
RAM/worker-count, and that an explicit override always wins.
"""

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
