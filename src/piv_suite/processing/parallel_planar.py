"""Tier 3 of the CPU perf overhaul: process-level parallelism across
INDEPENDENT planar frame pairs, via ProcessPoolExecutor. Scoped to the
planar CPU path specifically (the stated goal -- cut planar CPU
processing time); stereo and GPU batches keep their existing serial
loops unchanged (stereo shares per-pair camera-mapping/dewarp state
across two engine calls per pair, and GPU has CUDA-context concerns
neither this investigation nor this module addresses).

Pairs are fully independent -- no state carries from one pair to the
next (each engine call starts fresh from its own frame_a/frame_b) -- so
running them across multiple OS processes changes only wall-clock
throughput, never any individual pair's result. That's what makes this
tier exact "by construction" rather than something that needs its own
numerical A/B: verification here is process-count equivalence (same
output, byte-identical, regardless of how many workers ran it and in
what completion order), not a correctness question about the PIV math
itself (already covered by engines/_openpiv_speedups.py's own tests).

Used by BOTH piv_suite_gui/workers/pipeline_worker.py and
piv_suite/cli/main.py's planar batch loops via
process_one_pair_planar_worker() and run_planar_batch_parallel() --
factored out here ONCE so the two call sites' worker-process logic can't
drift apart into two subtly-different implementations. Each call site
keeps its OWN summary-row / log-line / Qt-signal formatting untouched;
only the actual per-pair computation (build engine, correlate, calibrate,
save outputs) is shared.

n_workers <= 1 is NEVER routed through this module -- both call sites
keep their original, unmodified serial per-pair loop for that case, so
"today's behavior" stays exactly one code path, trivially unchanged,
rather than "a worker pool of size 1" that would still carry process-
boundary/pickling differences worth worrying about.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..config.legacy import to_cpu_settings
from ..engines._openpiv_speedups import apply_speedups
from ..engines.registry import get_engine_factory
from ..plotting.planar import plot_and_save_planar
from . import pipeline
from .postprocess import apply_calibration
from .preprocess import apply_preprocess_pair

# Process-local cache -- persists across tasks within the SAME worker
# process (ProcessPoolExecutor reuses worker processes across submitted
# tasks, it doesn't fork a fresh one per task), matching the serial
# path's own "build the engine once, reuse for every pair" behavior.
_worker_cache = {}


def _worker_init():
    """ProcessPoolExecutor initializer -- runs once per worker process
    before it handles any task.

    OMP_NUM_THREADS/OPENBLAS_NUM_THREADS=1: this app's numpy/scipy are
    OpenBLAS-backed (not MKL) -- confirmed via numpy's own build config.
    Without pinning this, EACH worker process would let OpenBLAS spawn
    its own multi-threaded pool sized to the full core count, and N such
    pools competing for the same physical cores oversubscribes hard
    (measured: this is the difference between N workers actually giving
    ~N x throughput vs. fighting each other for cycles). Must be set
    BEFORE numpy/openpiv/scipy are imported in the worker process (env
    vars read once at BLAS init time) -- ProcessPoolExecutor's
    initializer runs before any task-submitted code, which is early
    enough since numpy isn't imported at module level in the worker
    entry point path until apply_speedups() (and this module's own
    later imports) pull it in.

    apply_speedups(): same monkeypatches the serial path applies once at
    CPUPIVProcess construction -- each worker process is a fresh
    interpreter (Windows only has the "spawn" start method), so this has
    to run again per process; apply_speedups() is idempotent so calling
    it again from inside CPUPIVProcess.__init__ (which every pair's
    engine build still does) is harmless.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    apply_speedups()


def _get_worker_engine(frame_shape, correlation, validation):
    """Builds (or reuses, within this worker process) the CPU engine for
    frame_shape. Cached by shape alone -- this app processes one shape
    per batch in practice; a shape change rebuilds (correct, just not
    the common case)."""
    cached = _worker_cache.get(frame_shape)
    if cached is not None:
        return cached
    cpu_settings = to_cpu_settings(correlation, validation)
    factory = get_engine_factory("cpu")
    engine, x, y = factory(frame_shape, {"cpu_settings": cpu_settings})
    _worker_cache.clear()  # only one shape needs to stay cached at a time
    _worker_cache[frame_shape] = (engine, x, y)
    return engine, x, y


def process_one_pair_planar_worker(idx, pair_id, frame_a, frame_b, cfg, output_dir):
    """The actual per-pair work, run inside a worker process. Same
    sequence as cli.main.handle_pair_planar and
    pipeline_worker.PipelineWorker._process_set_planar's own per-pair
    body: preprocess -> build/reuse engine -> pipeline.process_frames ->
    calibrate -> save npz/plot (done HERE, by the worker, using its own
    filesystem access -- avoids shipping full u/v/x/y arrays back over
    IPC just to have the main process write them to disk).

    Returns a plain dict (picklable) with everything either call site
    needs to build its own summary row / log line / Qt signal payload --
    this function doesn't format any of that itself, so it stays usable
    by both without encoding either site's presentation choices.

    show_plots is deliberately NOT forwarded to plot_and_save_planar
    here (always False) -- popping an interactive plot window from a
    background worker process doesn't make sense (no event loop, no user
    watching that process); this only affects a caller that both sets
    output.show_plots=True AND uses more than one worker, an interactive-
    preview combination outside Tier 3's actual target (batch
    throughput), not a numerical concern.
    """
    t_pre0 = time.time()
    frame_a, frame_b = apply_preprocess_pair(frame_a, frame_b, cfg.preprocess)
    t_pre = time.time() - t_pre0

    engine, x, y = _get_worker_engine(frame_a.shape, cfg.correlation, cfg.validation)

    post = cfg.postprocess.for_pipeline()
    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post)

    t_post0 = time.time()
    u, v = apply_calibration(u, v, cfg.calibration.pixel_pitch_mm, cfg.calibration.frame_dt_s)
    n_valid, n_total = int(valid.sum()), int(valid.size)

    if cfg.output.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_velocity.npz"),
                  x=x, y=y, u=u, v=v, valid=valid)
    if cfg.output.save_plot:
        plot_and_save_planar(x, y, u, v, valid,
                              os.path.join(output_dir, f"{pair_id}_quiver.png"),
                              title=f"PIV velocity field -- {pair_id}",
                              quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi,
                              show_plots=False)
    t_post = time.time() - t_post0

    return {
        "idx": idx,
        "pair_id": pair_id,
        "elapsed": elapsed,
        "t_pre": t_pre,
        "t_post": t_post,
        "n_valid": n_valid,
        "n_total": n_total,
        "n_rejected_range_residual": rejects["range_residual"],
        "n_rejected_std_dev": rejects["std_dev"],
    }


def run_planar_batch_parallel(pair_source, cfg, output_dir, n_workers,
                               on_pair_started=None, on_pair_finished=None,
                               on_pair_error=None, cancel_check=None):
    """Drives process_one_pair_planar_worker() across a ProcessPoolExecutor
    for every (pair_id, frame_a, frame_b) in pair_source. Caller supplies
    n_workers (> 1 -- callers route n_workers<=1 to their own unmodified
    serial loop instead, see module docstring).

    Ordering: results are re-sorted by submission index before being
    returned, so `summary_rows`' order (and therefore any CSV/summary
    output built from it) is deterministic regardless of which worker
    finished which pair first -- matching the serial loop's natural
    (submission-order) ordering exactly.

    Cancellation: cancel_check() is polled before each new submission;
    once it returns True, no FURTHER pairs are submitted, but pairs
    already in flight are allowed to finish (a process pool can't
    "half-cancel" a running task) -- their results are still collected
    and included, matching "cancel stops new work, not in-progress work".

    Errors: if one pair's worker task raises, on_pair_error(pair_id, exc)
    is called and that pair is simply left out of the returned results --
    matching the serial loop's own per-pair try/except (one bad pair
    doesn't abort the whole batch).

    on_pair_started(pair_id) fires at submission time; on_pair_finished
    (pair_id, result_dict) fires as each future completes (not
    necessarily in submission order) -- callers that need strict order
    for anything beyond the final summary_rows list should not assume
    on_pair_finished fires in pair order.
    """
    results = {}
    cancelled = False

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as executor:
        futures = {}
        for idx, (pair_id, frame_a, frame_b) in enumerate(pair_source):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            if on_pair_started is not None:
                on_pair_started(pair_id)
            future = executor.submit(
                process_one_pair_planar_worker, idx, pair_id, frame_a, frame_b, cfg, output_dir)
            futures[future] = (idx, pair_id)

        for future in as_completed(futures):
            idx, pair_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if on_pair_error is not None:
                    on_pair_error(pair_id, exc)
                continue
            results[idx] = result
            if on_pair_finished is not None:
                on_pair_finished(pair_id, result)

    ordered = [results[idx] for idx in sorted(results)]
    return ordered, cancelled
