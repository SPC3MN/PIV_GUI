"""Tier 3 stereo counterpart to parallel_planar.py: process-level
parallelism across INDEPENDENT stereo pairs (snapshots) within one
recording, via ProcessPoolExecutor.

parallel_planar.py's own module docstring gives "stereo shares per-pair
camera-mapping/dewarp state across two engine calls per pair" as the
reason stereo was left serial. That doesn't survive scrutiny: cam0/cam1
(calibration.camera_mapping.CameraMapping, built once per project from
cfg.stereo.* -- see calibration.camera_mapping.build_camera_mapping) and
the triangulation angles are IDENTICAL for every pair in a recording,
never mutated per pair -- nothing about them is real per-pair state.
What genuinely varies pair to pair is exactly the same shape planar
already parallelizes: two dewarp_image() calls and one
pipeline.process_stereo_pair() call (which itself runs both cameras'
engines, triangulates, and validates the combined result), all derived
solely from THIS pair's own fa0/fb0/fa1/fb1 with no carry-over to the
next pair. Stereo pairs are exactly as mutually independent as planar
ones; leaving them serial was a scoping decision (planar's stated goal
was "cut planar CPU processing time"), not a technical blocker.

cam0/cam1 ARE rebuilt once per WORKER PROCESS (not once per pair) as a
caching optimization, not a correctness requirement: CameraMapping.
dewarp_image caches its (comparatively expensive) world<->raw coordinate
grid on the object itself, keyed by world_shape (camera_mapping.py's
_ensure_grid) -- reconstructing cam0/cam1 from scratch for every pair
would silently recompute that identical grid every time. cfg (and
therefore cfg.stereo.*) is fixed for an entire run_stereo_batch_parallel
call, so _worker_init builds cam0/cam1 exactly once, before this worker
handles its first task, and every pair it processes afterward reuses
those same two objects with their now-warm grid cache -- mirroring
_worker_engine_cache's "build once per worker process, reuse across
pairs" strategy for the PIV engine itself.

One CPU engine, not two, is cached and reused for BOTH cameras within a
worker process (keyed by dewarped frame shape, same cache strategy as
parallel_planar._get_worker_engine): cam0's and cam1's dewarped frames
share cfg.stereo.world_shape, so they share the same engine cache key --
the same engine instance is passed as BOTH engine0 and engine1 to
pipeline.process_stereo_pair, which calls them strictly sequentially
(engine0's call completes before engine1's begins), so there is no
cross-camera state leak -- see process_one_pair_stereo_worker's docstring
for why reusing one engine instance across both cameras changes nothing
about either camera's numeric output.

Deliberately NOT sharing the ProcessPoolExecutor/semaphore/wait
submission-loop scaffolding with parallel_planar.run_planar_batch_parallel
via a common driver, even though the two are structurally near-identical:
that loop is small, already covered by test_parallel_planar.py, and
shaped by real, specifically-documented regressions (the GUI-freeze
sliding-window fix, the wait()-per-call-not-per-completion perf fix --
see run_planar_batch_parallel's own docstring). Refactoring it into a
shared generic driver right now would mean touching already-working,
already-tested planar code for a modest de-duplication win, risking a
subtle behavioral difference reaching BOTH call sites at once instead of
just this new one. What IS shared, by plain import (never copy-paste):
apply_speedups, get_engine_factory, to_cpu_settings, build_camera_mapping,
pipeline.process_stereo_pair, and the preprocess/postprocess helpers --
i.e. everything where duplication would actually risk drift. The submission loop itself is reimplemented independently
(structurally identical, not copy-pasted-then-forgotten) rather than
factored out; a future change to one loop's throttling/ordering/
cancellation behavior should prompt a look at the other's docstring,
flagged here on both sides.

Used by BOTH piv_suite_gui/workers/pipeline_worker.py and
piv_suite/cli/main.py's stereo batch loops, matching parallel_planar's
own "factored out here ONCE so the two call sites' worker-process logic
can't drift apart" rationale for its two call sites.

n_workers <= 1 is NEVER routed through this module -- same hard
requirement as parallel_planar (see that module's docstring for why):
both call sites keep their original, unmodified serial per-pair loop for
that case, so "today's behavior" (n_workers<=1) stays exactly one code
path rather than "a worker pool of size 1" that would still carry
process-boundary/pickling differences worth worrying about.
"""

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

from ..calibration.camera_mapping import (build_stereo_cameras, stereo_angles_for,
                                          stereo_fov_valid)
from ..config.legacy import to_cpu_settings
from ..engines._openpiv_speedups import apply_speedups
from ..engines.registry import get_engine_factory
from ..plotting.stereo import plot_and_save_stereo
from . import pipeline
from ._parallel_cancel import reap_executor_workers, start_cancel_poller
from .preprocess import apply_preprocess_pair

# Process-local caches -- persist across tasks within the SAME worker
# process (ProcessPoolExecutor reuses worker processes across submitted
# tasks). _worker_cameras is a single (cam0, cam1) tuple, not a dict --
# unlike the engine cache, there's only ever one calibration per batch
# (cfg is fixed for the whole run_stereo_batch_parallel call), so no
# shape-keyed lookup is needed.
_worker_engine_cache = {}
_worker_cameras = None


def _worker_init(cfg):
    """ProcessPoolExecutor initializer -- runs once per worker process
    before it handles any task. See parallel_planar._worker_init's
    docstring for the OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/apply_speedups
    rationale (identical here, not repeated).

    Building cam0/cam1 here (rather than lazily, the way
    _get_worker_engine builds the engine on first use) is possible
    because cfg -- and therefore cfg.stereo.* -- is already known at pool
    construction time, unlike frame_shape which isn't known until the
    first pair actually arrives. ProcessPoolExecutor's `initargs` carries
    cfg to every worker process once, at startup, not per task."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    apply_speedups()
    global _worker_cameras
    _worker_cameras = build_stereo_cameras(cfg.stereo)


def _get_worker_engine(frame_shape, correlation, validation):
    """Builds (or reuses, within this worker process) the CPU engine for
    frame_shape -- shared by BOTH cameras (see module docstring). Cached
    by shape alone, same as parallel_planar._get_worker_engine: this app
    processes one shape per batch in practice; a shape change rebuilds
    (correct, just not the common case)."""
    cached = _worker_engine_cache.get(frame_shape)
    if cached is not None:
        return cached
    cpu_settings = to_cpu_settings(correlation, validation)
    factory = get_engine_factory("cpu")
    engine, x, y = factory(frame_shape, {"cpu_settings": cpu_settings})
    _worker_engine_cache.clear()  # only one shape needs to stay cached at a time
    _worker_engine_cache[frame_shape] = (engine, x, y)
    return engine, x, y


def process_one_pair_stereo_worker(idx, pair_id, fa0, fb0, fa1, fb1, cfg, output_dir):
    """The actual per-pair work, run inside a worker process. Same
    sequence as cli.main.handle_pair_stereo and pipeline_worker.
    PipelineWorker._process_set_stereo's own per-pair body: preprocess ->
    dewarp both cameras -> build/reuse engine -> pipeline.process_stereo_
    pair (both engines -> combine -> validate the combined field once) ->
    save npz/plot (done HERE, by the worker, same rationale as
    parallel_planar.process_one_pair_planar_worker for doing its own
    filesystem I/O instead of shipping full x/y/U/V/W arrays back over
    IPC).

    Reuses ONE engine instance for both cameras (see _get_worker_engine)
    rather than building two, as the serial loops both do -- passed as
    both engine0 and engine1 to process_stereo_pair, which calls them
    strictly sequentially, not concurrently (engines/cpu_engine.py's
    CPUPIVProcess sets self.val_locations at the end of __call__, and
    engine1's call only starts after engine0's has fully returned) -- so
    there is no cross-camera state leak, and each camera's own settings/
    coords/scaling (self._settings, self.coords, self.scaling_par) are
    set once at construction and never mutated by __call__, so both
    cameras see the identical, correct engine configuration either way.
    Reusing one engine instead of building two functionally-identical
    ones (same correlation/validation settings, same dewarped
    world_shape) doesn't change either camera's u/v output by a single
    bit -- it just skips rebuilding an equivalent engine a second time.

    show_plots is deliberately NOT forwarded to plot_and_save_stereo here
    (always False) -- see process_one_pair_planar_worker's docstring for
    why (no event loop / no user watching a background worker process).
    """
    cam0, cam1 = _worker_cameras

    t_pre0 = time.time()
    fa0, fb0 = apply_preprocess_pair(fa0, fb0, cfg.preprocess)
    fa1, fb1 = apply_preprocess_pair(fa1, fb1, cfg.preprocess)
    dw_a0 = cam0.dewarp_image(fa0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_b0 = cam0.dewarp_image(fb0, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_a1 = cam1.dewarp_image(fa1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    dw_b1 = cam1.dewarp_image(fb1, cfg.stereo.world_shape, cfg.stereo.dewarp_order)
    t_pre = time.time() - t_pre0

    post = cfg.postprocess.for_pipeline()
    engine, x, y = _get_worker_engine(dw_a0.shape, cfg.correlation, cfg.validation)

    # See preview_panel._compute_stereo's comment on the same mask /
    # calibration.camera_mapping.stereo_fov_valid's own docstring. y is
    # _get_worker_engine's DISPLAY-flipped coordinate -- un-flip back to
    # world_to_raw's row-down grid convention first.
    y_row_down = cfg.stereo.world_shape[0] - y
    fov_valid = stereo_fov_valid(cam0, cam1, x, y_row_down)
    angles = stereo_angles_for(cfg.stereo, cam0, cam1, x, y_row_down)

    # process_stereo_pair validates the COMBINED/triangulated field once
    # (not each camera's raw 2D field independently, then intersected) --
    # see its own docstring for the real-data evidence this replaced the
    # previous process_frames-x2-then-AND approach with. `elapsed` is its
    # own two engine() calls only, matching the original's elapsed1+
    # elapsed2 scope exactly; t_post0 is set AFTER it returns so t_post
    # keeps measuring the same thing it always did here (npz/plot save),
    # not double-counting engine time that process_stereo_pair already
    # accounts for separately.
    U, V, W, valid, elapsed, r = pipeline.process_stereo_pair(
        engine, engine, dw_a0, dw_b0, dw_a1, dw_b1, angles,
        cfg.stereo.world_scale_px_per_mm, cfg.calibration.frame_dt_s, fov_valid, post, x, y)
    n_valid, n_total = int(valid.sum()), int(valid.size)
    t_post0 = time.time()

    if cfg.output.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_stereo_velocity.npz"),
                  x=x, y=y, U=U, V=V, W=W, valid=valid)
    if cfg.output.save_plot:
        plot_and_save_stereo(x, y, U, V, W, valid,
                              os.path.join(output_dir, f"{pair_id}_stereo_quiver.png"),
                              title=f"Stereo PIV -- {pair_id}",
                              quiver_scale=cfg.output.quiver_scale, plot_dpi=cfg.output.plot_dpi,
                              show_plots=False)
    t_post = time.time() - t_post0

    n_range = r["range_residual"]
    n_std = r["std_dev"]

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


def run_stereo_batch_parallel(pair_source, cfg, output_dir, n_workers,
                               on_pair_started=None, on_pair_finished=None,
                               on_pair_error=None, cancel_check=None):
    """Drives process_one_pair_stereo_worker() across a ProcessPoolExecutor
    for every (pair_id, fa0, fb0, fa1, fb1) in pair_source. Caller supplies
    n_workers (> 1 -- callers route n_workers<=1 to their own unmodified
    serial loop instead, see module docstring) and angles (radians, from
    cfg.stereo.alpha1_deg/alpha2_deg/beta1_deg/beta2_deg -- computed once
    by the caller, same as both serial loops already do, not recomputed
    per pair or per worker).

    Structurally identical to parallel_planar.run_planar_batch_parallel --
    same ordering/error/throttling contract; see that function's
    docstring for the full rationale on each point (re-sorting by
    submission index, semaphore-based sliding-window throttling sized to
    n_workers, single wait() call). Not shared as a common helper -- see
    this module's own docstring for why. cfg is passed to the executor's
    initializer (not just to each task) so _worker_init can build cam0/
    cam1 exactly once per worker process before any pair arrives.

    Cancellation: a background poller (processing._parallel_cancel.
    start_cancel_poller) watches cancel_check() independent of the
    submission loop below and, the instant it returns True, HARD-KILLS
    every in-flight worker process (processing._parallel_cancel.
    kill_executor_workers) instead of letting them finish naturally --
    see that module's docstring for the full rationale (this is a fix for
    a real reported bug: "Cancel doesn't actually stop processing", which
    turned out to mean waiting out up to n_workers already-running
    full-resolution stereo pairs). A killed pair's result is simply
    dropped -- not counted as an error, not counted as completed (see
    _on_done below).
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
            # A future killed (or cancelled while still queued) by the
            # cancel poller raises here too (BrokenProcessPool /
            # CancelledError) -- an EXPECTED consequence of cancellation,
            # not a real per-pair error, so it's dropped silently rather
            # than reported via on_pair_error.
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
    executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(cfg,))
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
                    process_one_pair_stereo_worker, idx, pair_id, fa0, fb0, fa1, fb1, cfg, output_dir)
            except Exception:
                # Lost a race against the cancel poller: it killed workers
                # between our semaphore.acquire()/cancel_event check above
                # and this submit() call. The exception type varies --
                # BrokenProcessPool is the common case, but killing a
                # worker mid-replacement-spawn can also surface as a raw
                # OSError/EOFError from the multiprocessing plumbing
                # underneath (confirmed directly) -- so this deliberately
                # doesn't pattern-match on BrokenProcessPool specifically.
                # Only treated as "expected, from cancellation" (swallowed)
                # when cancellation is actually in play; anything else
                # re-raises as a genuine bug, same as before this fix.
                if cancel_event.is_set() or (cancel_check is not None and cancel_check()):
                    cancel_event.set()
                    break
                raise
            all_futures.append(future)
            future.add_done_callback(lambda f, idx=idx, pair_id=pair_id: _on_done(f, idx, pair_id))

        # Every future's own callback already ran (or will run) as it
        # completes -- this just blocks the calling thread until they all
        # have, before returning. A single wait() call on the whole
        # (already-submitted) list, not a per-completion loop -- see
        # run_planar_batch_parallel's docstring for the measured perf
        # reason this matters. If cancellation killed the workers, every
        # outstanding future is already resolved (with BrokenProcessPool)
        # by the time we get here -- this returns near-instantly.
        wait(all_futures)
    finally:
        poll_stop.set()
        if poll_thread is not None:
            poll_thread.join(timeout=1.0)
        # wait=False: workers are already dead (normal completion) or
        # already killed (cancellation) by this point either way. But
        # shutdown(wait=False) alone doesn't CONFIRM they actually
        # exited -- see reap_executor_workers's docstring (parallel_
        # planar.py's identical fix) for the real hang this caused.
        executor.shutdown(wait=False, cancel_futures=True)
        reap_executor_workers(executor, pids_before)

    ordered = [results[idx] for idx in sorted(results)]
    return ordered, cancel_event.is_set()
