"""Runtime hardware auto-tuning -- the ONLY two things this codebase lets
vary by machine (correlation chunk size, worker process count), derived
here instead of hardcoded or asked of the user. Every other optimization
in engines/_openpiv_speedups.py is an unconditional win and is never
gated on anything this module returns.

Three call sites, two different chunk-size questions:
- engines/_openpiv_speedups.fast_fft_correlate_images uses
  recommended_chunk_size() to bound how many windows' worth of FFT
  correlation it materializes at once. This isn't primarily a memory-
  safety mechanism on a well-provisioned machine (a 24-core/192GB box
  has no trouble holding the whole batch) -- it's a cache-locality win
  (measured ~19% faster FFT time from a chunk's working set fitting in
  L2/shared L3 rather than streaming through RAM), which is why the
  target size below is a fixed cache-sized default, only CLAMPED down by
  available RAM rather than derived from it. That clamp is what protects
  the pathological case this investigation started from: a memory-
  starved dev box where available RAM fell to 0.04 GB mid-pass and the
  machine paged to disk -- on a machine like that, the RAM ceiling binds
  hard and chunk size shrinks well below the cache-sized default; on a
  RAM-rich machine it never binds at all.
- engines/_openpiv_speedups.fast_extended_search_area_piv uses
  recommended_pipeline_chunk_size() -- a DIFFERENT, much larger chunk
  target for its own outer (per-grid-row) streaming loop, which exists to
  bound PEAK memory for Tier 3's many-concurrent-worker-processes case,
  not cache locality. Measured directly: reusing recommended_chunk_size's
  cache-sized target for this outer loop was a measured 12% wall-clock
  REGRESSION on this 48-thread machine (a real image's grid has more
  columns per row than that target's window count, so every single grid
  row became its own chunk -- ~370 outer iterations for one fine pass,
  each paying fixed per-call numpy/Python overhead for no corresponding
  memory-safety gain beyond what the coarser target already provides).
  Keeping these as two separate functions, not one shared parameter, is
  deliberate: conflating "small enough to fit in cache" with "small
  enough to bound peak memory across N worker processes" is exactly what
  caused that regression.
- pipeline_worker.py / cli/main.py's batch loops use
  recommended_workers() to size a ProcessPoolExecutor for Tier 3's
  cross-pair parallelism.
"""

import ctypes
import os
import sys

# A chunk's working set (window_a + window_b + rfft2 outputs + the real
# correlation output, per window, at float64) sized to fit comfortably in
# a modern per-core L2/shared-L3 slice -- not a tuned constant, just a
# conservative "definitely fits in cache" target. Tune down if profiling
# on a smaller-cache machine ever shows this is too big to matter.
_TARGET_CHUNK_BYTES = 16 * 1024 * 1024
_MIN_CHUNK_WINDOWS = 64
_MAX_CHUNK_WINDOWS = 20_000

# recommended_pipeline_chunk_size's clamps -- deliberately much larger
# than the cache-locality target above. The floor matters more than the
# ceiling: too small and (as measured) per-chunk Python/numpy call
# overhead dominates and erases the point of chunking at all.
_MIN_PIPELINE_CHUNK_WINDOWS = 4_000
_MAX_PIPELINE_CHUNK_WINDOWS = 250_000
_PIPELINE_RAM_FRACTION = 0.15

# Conservative fallback when the platform RAM probe isn't available
# (non-Windows, or the ctypes call fails) -- deliberately pessimistic so
# an unknown machine gets treated like the memory-starved case, not the
# 192GB one.
_FALLBACK_AVAILABLE_RAM_BYTES = 2 * 1024**3

# Per-worker footprint estimate for recommended_workers() -- one worker
# holds a frame pair (a handful of MB for typical PIV frames, well under
# 100MB even for large 3000x4000 float32 images) plus its own working
# chunk; padded generously since underestimating here is what leads to
# oversubscription/paging, not underuse of cores.
_BYTES_PER_WORKER_ESTIMATE = 2 * 1024**3
_MIN_FREE_RAM_RESERVE_BYTES = 4 * 1024**3


def available_ram_bytes():
    """Currently-available (not total) physical RAM, in bytes. Windows:
    GlobalMemoryStatusEx via ctypes (no extra dependency). Elsewhere:
    a conservative fixed fallback -- see module docstring for why
    conservative is the right default when we can't actually check."""
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

                def __init__(self):
                    self.dwLength = ctypes.sizeof(self)

            status = _MEMORYSTATUSEX()
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * avail_pages)
        except (ValueError, OSError, AttributeError):
            pass
    return _FALLBACK_AVAILABLE_RAM_BYTES


def recommended_chunk_size(window_shape, dtype_itemsize=8, ram_fraction=0.05):
    """Number of windows to correlate per chunk in
    fast_fft_correlate_images. See module docstring: sized to a fixed
    cache-friendly working-set target, clamped down by a fraction of
    currently-available RAM so a memory-starved machine can't be pushed
    into paging by a chunk sized for a bigger box."""
    wy, wx = window_shape[-2], window_shape[-1]
    # Rough per-window working-set: image_a + image_b windows, their
    # rfft2 outputs (complex128, roughly half the real size in the last
    # axis but padded generously here), and the real correlation output.
    # Not exact -- exactness isn't the point, staying comfortably inside
    # a cache-sized budget is.
    bytes_per_window = 6 * wy * wx * dtype_itemsize

    cache_target_windows = max(1, _TARGET_CHUNK_BYTES // bytes_per_window)

    ram_ceiling_bytes = available_ram_bytes() * ram_fraction
    ram_ceiling_windows = max(1, int(ram_ceiling_bytes // bytes_per_window))

    chunk = min(cache_target_windows, ram_ceiling_windows)
    return int(max(_MIN_CHUNK_WINDOWS, min(_MAX_CHUNK_WINDOWS, chunk)))


def recommended_pipeline_chunk_size(window_shape, dtype_itemsize=8, assumed_concurrent_workers=None):
    """Windows-per-chunk for fast_extended_search_area_piv's own outer
    (per-grid-row) streaming loop. Sized to bound ONE worker's peak
    memory (its window-stack + correlation-output working set) to a
    fraction of available RAM divided by how many such workers might run
    concurrently (Tier 3) -- NOT to fit in cache the way
    recommended_chunk_size is; see module docstring for why these are two
    separate functions with two different targets, not one shared value.
    `assumed_concurrent_workers` defaults to recommended_workers()'s own
    answer (os.cpu_count() unless overridden) -- the same worker count
    Tier 3 would actually launch."""
    if assumed_concurrent_workers is None:
        assumed_concurrent_workers = max(1, recommended_workers())

    wy, wx = window_shape[-2], window_shape[-1]
    # Generous per-window estimate for this stage's own working set (the
    # outer chunk's aa/bb windows plus the correlation output written
    # into it) -- see recommended_chunk_size's docstring for the same
    # style of estimate; exactness isn't the point here either.
    bytes_per_window = 6 * wy * wx * dtype_itemsize

    ram_budget_bytes = (available_ram_bytes() * _PIPELINE_RAM_FRACTION) / assumed_concurrent_workers
    windows = max(1, int(ram_budget_bytes // bytes_per_window))
    return int(max(_MIN_PIPELINE_CHUNK_WINDOWS, min(_MAX_PIPELINE_CHUNK_WINDOWS, windows)))


def _max_windows_workers():
    """Windows' hard ProcessPoolExecutor cap, or None off-Windows. Passing
    an explicit max_workers above this raises `ValueError: max_workers
    must be <= N` -- a real WaitForMultipleObjects handle-count limit,
    confirmed on a real 72-logical-core dual-socket Windows machine
    (os.cpu_count() there, with abundant RAM, made recommended_workers()
    auto-detect 72 and crash the first time Tier 3 actually launched).
    CPython's own ProcessPoolExecutor(max_workers=None) already self-
    clamps to this silently -- but recommended_workers() always returns
    an explicit int (from n_override or the auto-detect below), which
    hits the raising branch instead, so this app must apply the same
    clamp itself. Reads CPython's own (private, but stable across many
    versions) constant so this always matches whatever's actually
    running, falling back to the long-standing value if that name is
    ever removed."""
    if sys.platform != "win32":
        return None
    try:
        from concurrent.futures.process import _MAX_WINDOWS_WORKERS
        return _MAX_WINDOWS_WORKERS
    except ImportError:
        return 61


def recommended_workers(n_override=None):
    """Worker process count for Tier 3's cross-pair ProcessPoolExecutor.
    n_override (from config.schema.PerformanceSettings.n_workers) wins
    over the auto-detected value -- but NOT over Windows' hard platform
    ceiling (_max_windows_workers()): a user-typed n_workers=72 on a
    72-core Windows machine would otherwise crash with the exact same
    ValueError the auto-detect path used to hit, just later and more
    confusingly (mid-batch, from inside ProcessPoolExecutor's own
    constructor) -- silently clamping down is strictly better than that."""
    win_cap = _max_windows_workers()

    if n_override is not None:
        workers = max(1, int(n_override))
        return workers if win_cap is None else min(workers, win_cap)

    cpu_count = os.cpu_count() or 1
    avail = available_ram_bytes()
    usable = max(0, avail - _MIN_FREE_RAM_RESERVE_BYTES)
    ram_workers = max(1, usable // _BYTES_PER_WORKER_ESTIMATE)

    workers = int(max(1, min(cpu_count, ram_workers)))
    return workers if win_cap is None else min(workers, win_cap)
