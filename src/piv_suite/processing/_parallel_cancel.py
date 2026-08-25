"""Shared cancellation-kill helper for Tier 3's ProcessPoolExecutor
batches (processing/parallel_planar.py, processing/parallel_stereo.py).

THE BUG THIS FIXES: concurrent.futures.ProcessPoolExecutor has no public
way to stop a task that is already RUNNING -- shutdown(cancel_futures=True)
(Python 3.9+) only cancels futures that haven't started executing yet; one
already handed to a worker process runs to completion regardless of
anything the calling side does. Both run_planar_batch_parallel and
run_stereo_batch_parallel throttle submission to a SLIDING WINDOW of at
most n_workers pairs in flight at once (see either module's own
docstring for why: unthrottled submission flooded the GUI with
near-simultaneous signals and was itself a real, separately-fixed bug).
Combined, that meant "Cancel" only ever stopped new submissions -- up to
n_workers pairs already in flight kept running to completion regardless,
which on a real multi-core machine can mean waiting out a dozen-plus
already-running full-resolution correlations, not "one pair" -- reported
directly as "Cancel doesn't actually stop processing."

THE FIX: reach into ProcessPoolExecutor's own `_processes` dict of live
worker Process objects (private, but confirmed directly against this
app's target Python/concurrent.futures version -- see kill_executor_
workers) and .terminate() them. Once a worker process dies unexpectedly,
ProcessPoolExecutor detects it and marks the ENTIRE pool broken -- every
outstanding future (running or merely queued) resolves near-instantly
with BrokenProcessPool, confirmed directly: terminate() -> wait() returns
in ~0ms, not after the killed tasks' natural remaining runtime. Each call
site's own _on_done callback treats a BrokenProcessPool/CancelledError
raised AFTER cancellation was requested as "dropped" (not an error, not a
completed pair) -- that half of the contract lives in each module (it
needs each module's own cancel_event), this one only owns the killing/
polling machinery both share.

WHY A POLLER THREAD, NOT JUST THE SUBMISSION LOOP'S OWN cancel_check()
CALL: the submission loop blocks on semaphore.acquire() whenever all
n_workers slots are already in flight -- which is exactly when a user is
most likely to hit Cancel (a busy batch). That acquire() call doesn't
return (so the loop's own cancel_check() never even runs) until a slot
frees up on its own, i.e. until an in-flight pair finishes NATURALLY --
the exact wait this whole fix exists to remove. A short-interval
background poller sidesteps that entirely: it reacts to cancel_check()
independent of whatever the submission loop happens to be blocked on,
then force-releases the throttling semaphore so a stuck submission loop
unblocks immediately and observes cancel_event for itself.
"""

import threading


def reap_executor_workers(executor, pids_before, timeout=2.0):
    """Guarantee every worker process `executor` ever spawned is actually
    GONE (not just told to shut down) before this returns, bounded by
    `timeout` total -- not `executor.shutdown(wait=True)`, which can hang
    indefinitely, and not bare `shutdown(wait=False)` either, which
    returns immediately without confirming anything actually exited.

    THE BUG THIS FIXES: found via real testing, not code review -- three
    separate full test-suite runs (which construct many ProcessPoolExecutor
    pools across parallel_planar.py's and parallel_stereo.py's own tests)
    each finished their actual test output (pytest's own summary line
    printed, correct pass/fail counts) in under 100s, but the PYTHON
    PROCESS ITSELF then sat alive, at 0% CPU, for over an HOUR before
    being killed manually. `executor.shutdown(wait=False, ...)` (used in
    both modules' cancellation-path `finally` blocks, reasoned there as
    "workers are already dead or already killed either way, no reason to
    block") sends worker processes their shutdown signal but does NOT
    confirm they actually exit before the calling function returns -- on
    Windows (spawn-based multiprocessing, which this app targets), a
    worker that doesn't cleanly notice its call-queue sentinel can be
    left alive indefinitely. Every one of those lingering processes stays
    registered in stdlib multiprocessing's own global active_children()
    list, which the interpreter's `atexit` finalizer (multiprocessing.
    util._exit_function) tries to join before the process is allowed to
    actually exit -- explaining exactly the observed symptom: real test
    work finishes fast, but the interpreter itself never gets to leave.

    This joins each of `executor`'s own worker processes with a share of
    `timeout`, then hard `.terminate()`s (and joins, briefly) anything
    still alive after that -- so by the time this returns, `executor` has
    provably left nothing behind, regardless of whether shutdown()'s own
    signal was received cleanly or not. Safe to call after `shutdown()`
    (idempotent on an already-dead process) or in place of relying on
    `shutdown(wait=True)` alone.

    `pids_before` is the set of `p.pid for p in multiprocessing.
    active_children()` captured by the CALLER right when its executor was
    created (before this pool spawned any worker of its own) -- see below
    for why this, and not `executor._processes` alone, is needed.

    Reads BOTH `executor._processes` (polled a few times over a short
    window, not just once) AND stdlib `multiprocessing.active_children()`
    filtered to pids NOT in `pids_before` (i.e. processes that appeared
    DURING this executor's lifetime and therefore are, with high
    confidence, this pool's own workers even if `_processes` itself never
    captured them).

    Both halves are needed, found via real, reproducible testing:
    - `executor._processes`-only reaping left one specific worker alive
      for minutes after every test run: ProcessPoolExecutor spawns
      workers LAZILY (_adjust_process_count(), called from submit()), so
      a worker whose OS process was created a moment ago but hasn't been
      inserted into `_processes` yet at the exact instant this function's
      first read happens is invisible to a single-snapshot read.
    - Sweeping ALL of `active_children()` unconditionally (no
      `pids_before` filter) fixed that but caused real collateral damage:
      a DIFFERENT, still-legitimately-running pool's worker (pre-existing
      before this call, unrelated to it) got caught by the same
      unconditional sweep and killed mid-task, corrupting its shared
      call_queue out from under it (observed directly: 'handle is
      closed' / garbled-unpickle crashes on an otherwise-valid, unrelated
      pair). Filtering to pids NOT in `pids_before` keeps the "catch a
      lazily-spawned straggler" benefit while never touching a process
      that already existed before this executor did."""
    import multiprocessing
    import time
    # ProcessPoolExecutor sets self._processes = None (not "leaves it
    # unset") once shutdown() has fully torn down its worker-management
    # state -- getattr(..., {}) alone doesn't help, since the ATTRIBUTE
    # exists (as None), so the fallback default never kicks in. Found via
    # a real crash: `executor.shutdown(wait=False, ...)` immediately
    # before this call can race ahead of THIS function reading
    # `_processes`, on a pool that already finished shutting down.
    deadline = time.monotonic() + timeout
    by_pid = {}
    for _ in range(5):  # a handful of quick re-reads catches a lazily-spawned straggler
        for p in (getattr(executor, "_processes", None) or {}).values():
            by_pid.setdefault(p.pid, p)
        for p in multiprocessing.active_children():
            if p.pid not in pids_before:
                by_pid.setdefault(p.pid, p)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    processes = list(by_pid.values())
    if not processes:
        return
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            try:
                process.terminate()
            except Exception:
                pass
            process.join(timeout=1.0)


def kill_executor_workers(executor):
    """Hard-terminate every live worker process in `executor` RIGHT NOW.
    Whatever pair each was mid-processing is abandoned entirely -- there
    is no partial/graceful stop here, matching "cancel means stop, right
    now, keep whatever already fully finished and discard the rest."
    `_processes` is a private ProcessPoolExecutor attribute (no public
    equivalent exists in the stdlib for this -- see module docstring);
    confirmed directly (not just read off documentation) that terminating
    every entry here and then wait()-ing on the pool's futures returns
    near-instantly with BrokenProcessPool, on this app's target Python
    version."""
    for process in list(getattr(executor, "_processes", {}).values()):
        try:
            process.terminate()
        except Exception:
            pass  # already exited on its own -- nothing to do


def start_cancel_poller(cancel_check, executor, cancel_event, semaphore, n_workers, poll_interval=0.02):
    """Starts a daemon thread that calls cancel_check() every
    poll_interval seconds; the first time it returns True, the thread
    sets cancel_event, kills every live worker process (see
    kill_executor_workers), and releases the submission loop's throttling
    semaphore n_workers times -- enough to unblock a submission loop
    currently stuck on semaphore.acquire() regardless of how many permits
    were genuinely outstanding, since nothing acquires the semaphore
    again after cancellation (over-releasing here is harmless, not a
    correctness bug -- see module docstring's "WHY A POLLER THREAD"
    section for why this force-release is the whole point).

    cancel_check=None (the CLI call sites, which never wire up
    cancellation at all) skips starting a thread entirely -- no polling
    overhead for a batch that can never be cancelled.

    Returns (thread_or_None, stop_event). Caller MUST stop_event.set()
    once the batch is done submitting/waiting (cancelled or not) in a
    `finally` block, then join the thread (if not None) with a short
    timeout -- otherwise this poller keeps running for the lifetime of
    whatever cancel_check closure it was given, forever."""
    stop_event = threading.Event()
    if cancel_check is None:
        stop_event.set()
        return None, stop_event

    def _poll():
        while not stop_event.is_set():
            if cancel_check():
                cancel_event.set()
                kill_executor_workers(executor)
                for _ in range(n_workers):
                    semaphore.release()
                return
            stop_event.wait(poll_interval)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    return thread, stop_event
