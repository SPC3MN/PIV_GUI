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
