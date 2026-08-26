"""Tier 3 dual-camera-planar counterpart to parallel_planar.py/
parallel_stereo.py: process-level parallelism across INDEPENDENT
dual-planar pairs (snapshots) within one recording, via
ProcessPoolExecutor.

Reported directly by the user after stereo got this treatment: "the
program doesn't work for parallel computing when the planar 2 camera
setup is used -- it just processes sequentially." Confirmed:
pipeline_worker.PipelineWorker._process_set_dual_planar and cli.main.
process_pairs_dual_planar were BOTH left as plain serial loops when
Tier 3 was added for planar and stereo -- an oversight (dual-planar
pairs are exactly as mutually independent as planar/stereo ones: two
single-camera correlations plus one combine_dual_planar_pair() call,
all derived solely from THIS pair's own fa0/fb0/fa1/fb1, no per-pair
carry-over), not a real technical blocker, matching parallel_stereo.py's
own account of why stereo had been left serial for the same wrong
reason.

Simpler than parallel_stereo.py in one respect: dual-planar has no
CameraMapping/dewarp step at all (see pipeline.combine_dual_planar_pair's
own docstring -- placement is a flat per-axis affine scale applied AFTER
correlation, not a per-camera lens-distortion dewarp before it), so
there's no per-worker calibration object to build in a pool initializer
the way parallel_stereo._worker_init builds cam0/cam1 -- cfg.dual_planar
is just plain data threaded straight into combine_dual_planar_pair() at
combine time, same as the serial loops already do.

One CPU engine, not two, is cached and reused for BOTH cameras within a
worker process, same rationale as parallel_stereo.process_one_pair_
stereo_worker's identical choice: both cameras' raw frames share the
same shape in every real dual-planar project seen so far (same sensor,
same acquisition settings), so they share the same engine-cache key, and
reusing one instance across two sequential calls changes neither
camera's numeric output (see that function's docstring for the full
non-interference argument -- applies unchanged here).

Used by BOTH piv_suite_gui/workers/pipeline_worker.py and
piv_suite/cli/main.py's dual-planar batch loops, matching parallel_planar's
and parallel_stereo's own "factored out here ONCE so the two call sites'
worker-process logic can't drift apart" rationale.

n_workers <= 1 is NEVER routed through this module -- same hard
requirement as parallel_planar/parallel_stereo (see parallel_planar's
module docstring for why): both call sites keep their original,
unmodified serial per-pair loop for that case.
"""

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait

from ..config.legacy import to_cpu_settings
from ..engines._openpiv_speedups import apply_speedups
from ..engines.registry import get_engine_factory
from ..plotting.planar import plot_and_save_planar
from . import pipeline
from ._parallel_cancel import reap_executor_workers, start_cancel_poller
from .preprocess import apply_preprocess_pair

# Process-local cache -- persists across tasks within the SAME worker
# process (ProcessPoolExecutor reuses worker processes across submitted
# tasks), same strategy as parallel_planar._worker_engine_cache /
# parallel_stereo._worker_engine_cache.
_worker_engine_cache = {}


def _worker_init():
    """ProcessPoolExecutor initializer -- runs once per worker process
    before it handles any task. See parallel_planar._worker_init's
    docstring for the OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/apply_speedups
    rationale (identical here, not repeated). No calibration object to
    build here (unlike parallel_stereo._worker_init's cam0/cam1) -- see
    module docstring."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    apply_speedups()


def _get_worker_engine(frame_shape, correlation, validation):
    """Builds (or reuses, within this worker process) the CPU engine for
    frame_shape -- shared by BOTH cameras (see module docstring). Cached
    by shape alone, same as parallel_planar/parallel_stereo's own
    _get_worker_engine: this app processes one shape per batch in
    practice; a shape change rebuilds (correct, just not the common
    case)."""
    cached = _worker_engine_cache.get(frame_shape)
    if cached is not None:
        return cached
    cpu_settings = to_cpu_settings(correlation, validation)
    factory = get_engine_factory("cpu")
    engine, x, y = factory(frame_shape, {"cpu_settings": cpu_settings})
    _worker_engine_cache.clear()  # only one shape needs to stay cached at a time
    _worker_engine_cache[frame_shape] = (engine, x, y)
    return engine, x, y


def _run_camera(frame_a, frame_b, cfg, engine_cache_key_shape):
    """One camera's ordinary single-camera planar correlation, returning
    the RAW (row-down, unflipped) coordinate grid (engine.coords) --
    NOT the display-flipped (x, y) the factory itself also returns --
    since pipeline.combine_dual_planar_pair needs each camera's raw grid
    to place its field correctly on the shared canvas (see that
    function's docstring for why the flipped grid would place it upside
    down). Mirrors cli.main._run_dual_planar_camera /
    PipelineWorker._run_dual_planar_camera exactly, just with a
    process-local cached engine instead of building a fresh one."""
    engine, _x_flipped, _y_flipped = _get_worker_engine(engine_cache_key_shape, cfg.correlation, cfg.validation)
    post = cfg.postprocess.for_pipeline()
    u, v, valid, elapsed, rejects = pipeline.process_frames(engine, frame_a, frame_b, post)
    x_raw, y_raw = engine.coords
    return u, v, x_raw, y_raw, valid, elapsed, rejects


def process_one_pair_dual_planar_worker(idx, pair_id, fa0, fb0, fa1, fb1, cfg, output_dir):
    """The actual per-pair work, run inside a worker process. Same
    sequence as cli.main.handle_pair_dual_planar and pipeline_worker.
    PipelineWorker._process_set_dual_planar's own per-pair body:
    preprocess -> correlate both cameras -> combine_dual_planar_pair ->
    save npz/plot (done HERE, by the worker, same rationale as
    parallel_planar.process_one_pair_planar_worker for doing its own
    filesystem I/O instead of shipping full X/Y/U/V arrays back over
    IPC)."""
    t_pre0 = time.time()
    fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
    fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)
    t_pre = time.time() - t_pre0

    u0, v0, x0, y0, valid0, elapsed0, r0 = _run_camera(fa0, fb0, cfg, fa0.shape)
    u1, v1, x1, y1, valid1, elapsed1, r1 = _run_camera(fa1, fb1, cfg, fa1.shape)
    elapsed = elapsed0 + elapsed1

    t_post0 = time.time()
    X, Y, U, V, valid = pipeline.combine_dual_planar_pair(
        (u0, v0, x0, y0, valid0), (u1, v1, x1, y1, valid1),
        cfg.dual_planar, cfg.calibration.frame_dt_s)
    n_valid, n_total = int(valid.sum()), int(valid.size)

    if cfg.output.save_npz:
        import numpy as np
        np.savez(os.path.join(output_dir, f"{pair_id}_velocity.npz"),
                  x=X, y=Y, u=U, v=V, valid=valid)
    if cfg.output.save_plot:
        plot_and_save_planar(X, Y, U, V, valid,
                              os.path.join(output_dir, f"{pair_id}_quiver.png"),
                              title=f"Dual-camera PIV velocity field -- {pair_id}",
                              quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi,
                              show_plots=False)
    t_post = time.time() - t_post0

    n_range = r0["range_residual"] + r1["range_residual"]
    n_std = r0["std_dev"] + r1["std_dev"]

    return {
        "idx": idx,
        "pair_id": pair_id,
        "elapsed": elapsed,
        "t_pre": t_pre,
        "t_post": t_post,
        "n_valid": n_valid,
        "n_total": n_total,
        "n_rejected_range_residual": n_range,
        "n_rejected_std_dev": n_std,
    }


def run_dual_planar_batch_parallel(pair_source, cfg, output_dir, n_workers,
                                    on_pair_started=None, on_pair_finished=None,
                                    on_pair_error=None, cancel_check=None):
    """Drives process_one_pair_dual_planar_worker() across a
    ProcessPoolExecutor for every (pair_id, fa0, fb0, fa1, fb1) in
    pair_source. Caller supplies n_workers (> 1 -- callers route
    n_workers<=1 to their own unmodified serial loop instead, see module
    docstring).

    Structurally identical to parallel_planar.run_planar_batch_parallel /
    parallel_stereo.run_stereo_batch_parallel -- same ordering/error/
    throttling/cancellation contract; see either's docstring for the full
    rationale on each point. Not shared as a common helper -- see
    parallel_stereo's own docstring for why (applies unchanged here: the
    loop is small, already covered by dedicated tests, and a shared
    generic driver would risk a subtle behavioral difference reaching
    all three call sites at once instead of just one).
    """
    results = {}
    results_lock = threading.Lock()
    semaphore = threading.Semaphore(n_workers)
    all_futures = []
    cancel_event = threading.Event()

    def _on_done(future, idx, pair_id):
        semaphore.release()
        try:
            result = future.result()
        except Exception as exc:
            if not cancel_event.is_set() and on_pair_error is not None:
                on_pair_error(pair_id, exc)
            return
        with results_lock:
            results[idx] = result
        if on_pair_finished is not None:
            on_pair_finished(pair_id, result)

    # Captured BEFORE this executor spawns any worker of its own --
    # reap_executor_workers uses this to safely catch a lazily-spawned
    # straggler this pool's own _processes dict missed, without ever
    # touching a DIFFERENT, pre-existing pool's still-legitimate worker.
    import multiprocessing
    pids_before = {p.pid for p in multiprocessing.active_children()}
    executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)
    poll_thread, poll_stop = start_cancel_poller(cancel_check, executor, cancel_event, semaphore, n_workers)
    try:
        for idx, (pair_id, fa0, fb0, fa1, fb1) in enumerate(pair_source):
            semaphore.acquire()  # blocks here until a slot frees up -- this IS the throttle
            if cancel_event.is_set():
                break
            if on_pair_started is not None:
                on_pair_started(pair_id)
            try:
                future = executor.submit(
                    process_one_pair_dual_planar_worker, idx, pair_id, fa0, fb0, fa1, fb1, cfg, output_dir)
            except Exception:
                # Lost a race against the cancel poller -- see
                # parallel_stereo.run_stereo_batch_parallel's identical
                # except clause for the full account of why this
                # deliberately doesn't pattern-match on BrokenProcessPool
                # specifically.
                if cancel_event.is_set() or (cancel_check is not None and cancel_check()):
                    cancel_event.set()
                    break
                raise
            all_futures.append(future)
            future.add_done_callback(lambda f, idx=idx, pair_id=pair_id: _on_done(f, idx, pair_id))

        wait(all_futures)
    finally:
        poll_stop.set()
        if poll_thread is not None:
            poll_thread.join(timeout=1.0)
        executor.shutdown(wait=False, cancel_futures=True)
        reap_executor_workers(executor, pids_before)

    ordered = [results[idx] for idx in sorted(results)]
    return ordered, cancel_event.is_set()
