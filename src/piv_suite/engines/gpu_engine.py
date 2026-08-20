"""GPU engine (openpiv_gpu.piv_gpu) -- lazy-imported everywhere so a
machine without cupy/openpiv-python-gpu installed can still import and use
this whole package (just not the GPU backend). Migrated from
piv_common.py's GPU section and spatial-tiling section (identical across
Stereo_PIV_GPU and Planar_PIV_GPU).
"""

import os
import sys
import time
import traceback

import numpy as np

# The real keyword names piv_gpu.__init__(frame_shape, min_search_size,
# **kwargs) accepts -- min_search_size itself is a separate required
# positional arg, NOT one of these. piv_gpu already ignores unrecognized
# kwargs safely, so this set exists only to WARN about likely typos in
# piv_settings, not to filter/drop anything.
PIV_GPU_SETTINGS_KEYS = frozenset({
    "search_size_iters", "overlap_ratio", "shrink_ratio", "center",
    "normalize", "mask_zero", "subpixel_method", "n_fft", "deforming_par",
    "batch_size", "s2n_method", "s2n_size", "validation_size", "s2n_tol",
    "median_tol", "mad_tol", "mean_tol", "rms_tol", "num_replacing_iters",
    "replacing_method", "replacing_size", "revalidate", "smooth",
    "smoothing_par", "dt", "scaling_par", "mask", "dtype_f",
})


def _gpu_check_log_path():
    """Next to the frozen exe (same folder as the installer's own
    gpu_setup_log.txt) when frozen, so both are discoverable in one
    place; next to this source file otherwise."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
    return os.path.join(base, "gpu_availability_check.log")


def is_gpu_available():
    """Probe whether the GPU backend can actually be used on this
    machine: cupy and openpiv_gpu importable, and at least one CUDA
    device visible. Used by the GUI to gray out the GPU option with an
    explanatory tooltip instead of letting a user pick GPU and crash
    mid-batch on an ImportError.

    Any failure writes the full exception to gpu_availability_check.log
    (overwritten each call) -- confirmed on real hardware that this can
    fail for reasons having nothing to do with the packages themselves
    being missing/broken (e.g. cupy's `import numpy` resolving
    differently inside a frozen app than a plain venv), which a bare
    True/False can't distinguish from "not installed at all". Without
    this, a report of "GPU stays greyed out" was undebuggable -- the
    installer's own gpu_setup_log.txt only covers whether the packages
    were successfully fetched, not whether they actually work once
    installed."""
    try:
        import cupy  # noqa: F401
        import openpiv_gpu  # noqa: F401
    except ImportError:
        _log_gpu_check_failure("import cupy / openpiv_gpu")
        return False
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            return True
        _log_gpu_check_failure(f"cupy.cuda.runtime.getDeviceCount() returned {count}")
        return False
    except Exception:
        _log_gpu_check_failure("cupy.cuda.runtime.getDeviceCount()")
        return False


def gpu_summary():
    """Short human-readable description of the GPU backend for the GUI's
    status header -- e.g. "CUDA 13" -- or None when the GPU backend isn't
    usable at all. Names WHICH CUDA runtime got picked up rather than a
    bare available/unavailable, since the installer offers 11/12/13 and a
    mismatch against the machine's actual NVIDIA toolkit is a real,
    previously hard-to-spot failure mode (see installer/README.md).

    Lives here rather than in the GUI so cupy stays confined to this
    module, matching how is_gpu_available() is already wrapped by
    engines.registry."""
    if not is_gpu_available():
        return None
    try:
        import cupy as cp
        # runtimeGetVersion() reports e.g. 13000 for CUDA 13.0
        return f"CUDA {cp.cuda.runtime.runtimeGetVersion() // 1000}"
    except Exception:
        return "CUDA"  # available, just couldn't name the version


def _log_gpu_check_failure(step):
    try:
        with open(_gpu_check_log_path(), "w") as f:
            f.write(f"is_gpu_available() failed at: {step}\n")
            f.write(f"sys.executable: {sys.executable}\n")
            f.write(f"frozen: {getattr(sys, 'frozen', False)}\n\n")
            traceback.print_exc(file=f)
    except Exception:
        pass  # diagnostics are best-effort -- never let logging itself break the check


def check_piv_settings(piv_settings):
    unknown = sorted(set(piv_settings) - PIV_GPU_SETTINGS_KEYS)
    if unknown:
        print(f"[warn] piv_settings has keys piv_gpu won't recognize: "
              f"{unknown} -- check spelling against piv_gpu's __init__ "
              "kwargs (they're silently ignored, not an error)")


class _NanSafeGPUProcess:
    """Thin __call__-only proxy around a piv_gpu instance -- NaN-median-
    fills the FINAL u/v output, a safety net for validation now living
    entirely in processing.postprocess (config.legacy.to_gpu_settings
    hard-codes every ValidationGPU tolerance to None, so piv_gpu's own
    per-pass validate/replace loop is fully inert, not just non-
    rejecting -- unlike the CPU backend's loose_typical_validation, which
    still catches literal NaN between passes, ValidationGPU's
    `abs(f - stat) > tol`-style comparisons never flag NaN regardless of
    tolerance, so nothing currently protects an intermediate pass's NaN
    from reaching PIVGPU.interpolate_displacement's RegularGridInterp
    -- see the perf/validation investigation notes). This only cleans the
    final returned field, not mid-pipeline NaN; that gap is currently
    latent, not live, since PIV_GUI doesn't wire piv_gpu's `mask` kwarg
    (the only way today to introduce field NaN) into the GUI/CLI at all
    -- revisit with a real per-pass fix if/when that changes.

    Everything else (`.coords`, `.val_locations`, `.scaling_par`,
    `.min_search_size`, ...) passes straight through via __getattr__ so
    run_tiled()/init_gpu_processor() see no difference from the raw
    piv_gpu instance."""

    def __init__(self, process):
        self._process = process

    def __call__(self, frame_a, frame_b):
        u, v = self._process(frame_a, frame_b)
        if np.isnan(u).any():
            u = np.where(np.isnan(u), np.nanmedian(u), u)
        if np.isnan(v).any():
            v = np.where(np.isnan(v), np.nanmedian(v), v)
        return u, v

    def __getattr__(self, name):
        return getattr(self._process, name)


def _init_gpu_processor_raw(frame_shape, min_search_size, piv_settings):
    """Like init_gpu_processor(), but returns coords BEFORE the top-down-
    to-bottom-up y-flip -- used directly by the non-tiled path (which
    flips using the whole frame's height right away) and by the tiling
    code in run_tiled() (which needs each tile's coords in a shared,
    un-flipped global frame before it can stitch tiles together and flip
    ONCE using the full frame's height, not each tile's)."""
    from openpiv_gpu.gpu_process import piv_gpu
    check_piv_settings(piv_settings)
    # piv_gpu asserts isinstance(..., tuple) on sequence-valued settings
    # (search_size_iters, overlap_ratio, ...) -- JSON only has lists, so
    # anything loaded from the config file needs converting back to a
    # tuple, or piv_gpu rejects it even though the values are correct.
    piv_settings = {k: (tuple(v) if isinstance(v, list) else v) for k, v in piv_settings.items()}
    process = _NanSafeGPUProcess(piv_gpu(frame_shape, min_search_size, **piv_settings))
    x, y = process.coords
    return process, x, y


def init_gpu_processor(frame_shape, min_search_size, piv_settings):
    process, x, y = _init_gpu_processor_raw(frame_shape, min_search_size, piv_settings)
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y


def gpu_free_report():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU free: {free / 1024 ** 3:.2f} GB / {total / 1024 ** 3:.2f} GB")


def free_gpu_pools():
    """Release cupy's memory pools back to the driver. Calls gc.collect()
    first -- piv_gpu instances hold internal reference cycles, so a bare
    `del process` doesn't actually drop their arrays' refcount to zero;
    free_all_blocks() only reclaims blocks whose Python objects are
    ALREADY garbage-collected, so without this, memory silently
    accumulates across repeated build/run/free cycles (very noticeable
    across many tiles or many stereo camera pairs) even though each
    individual `del process; free_gpu_pools()` call looks like it should
    have freed everything."""
    import gc
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ======================================================================
# Spatial tiling -- process very large frames as a grid of smaller,
# halo-padded tiles instead of all at once, so peak GPU memory is bounded
# by ONE tile's window count rather than the whole frame's. Each tile's
# PIV engine is built, run, and freed before the next tile starts.
#
# WHY A HALO: a window sitting exactly on a tile's edge needs real image
# data around it for its search area, not a hard crop at the tile
# boundary -- each tile is read out padded by `margin_px` on every side
# (clipped at the real frame edges), so its OWN piv_gpu call sees enough
# context. After the correlation for a tile, vectors from that halo are
# discarded, keeping only vectors that fall inside the tile's exclusive
# "core" region.
#
# WHY THE OUTPUT IS FLAT (not a grid): each tile's local window grid
# starts fresh at that tile's own origin, so neighboring tiles' kept
# vectors don't generally land on a single shared (ny, nx) lattice --
# smoothing is skipped for tiled output since it assumes a regular grid.
# ======================================================================
def compute_tiles(shape, n_tiles_y, n_tiles_x, margin_px):
    """Split a (H, W) frame into an n_tiles_y x n_tiles_x grid of tiles.
    Each tile dict has:
      "row"/"col"  -- this tile's position in the tile grid
      "core"       -- (y0, y1, x0, x1) this tile's EXCLUSIVE region in
                       the frame's global pixel coordinates
      "padded"     -- (py0, py1, px0, px1) the core expanded by
                       margin_px on each side, clipped to the frame"""
    H, W = shape
    y_edges = np.linspace(0, H, n_tiles_y + 1).round().astype(int)
    x_edges = np.linspace(0, W, n_tiles_x + 1).round().astype(int)
    tiles = []
    for row in range(n_tiles_y):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        py0, py1 = max(0, y0 - margin_px), min(H, y1 + margin_px)
        for col in range(n_tiles_x):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            px0, px1 = max(0, x0 - margin_px), min(W, x1 + margin_px)
            tiles.append({
                "row": row, "col": col,
                "core": (y0, y1, x0, x1),
                "padded": (py0, py1, px0, px1),
            })
    return tiles


def default_tile_margin(min_search_size, piv_settings):
    """A safe default halo margin -- 3x the coarsest pass's full window
    extent (window size doubles per level going up from min_search_size,
    same convention as piv_gpu itself).

    1x the coarsest window (the margin needed for windows near a tile's
    edge to see real pixel data across their whole search area) is NOT
    enough on its own -- confirmed on real Windows/CUDA hardware:
    multi-pass window DEFORMATION needs correlation context from a wider
    neighborhood than one window's own search area, since each pass's
    field feeds the next pass's deformation. Cutting that context off at
    a too-small tile boundary measurably degrades vectors near the seam
    (up to ~0.22px error vs. non-tiled processing on the same frame at
    1x margin, ~8x the algorithm's own ~0.01-0.03px noise floor) even
    though the raw pixel data was already sufficient. Measured on a
    512x512 frame, 2x2 tiles, 2-level multi-pass: 1x -> 0.22px max error,
    2x -> 0.03px, 3x -> 0.0008px (down at noise floor). 3x costs more
    GPU memory per tile than 1x (less benefit from tiling at small tile
    counts) -- pass correlation.tile_margin_px explicitly to override
    this default if memory is tighter than accuracy needs, or vice
    versa."""
    search_size_iters = piv_settings.get("search_size_iters", 1)
    num_passes = 1 if isinstance(search_size_iters, int) else len(search_size_iters)
    return 3 * min_search_size * (2 ** (num_passes - 1))


def run_tiled(frame_a, frame_b, ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
              report_gpu_mem=False, free_pools_fn=None):
    """Run a PIV engine (built per-tile via init_raw_fn(tile_shape) ->
    (process, x, y), e.g. a partial application of
    _init_gpu_processor_raw) across spatial tiles of a large frame pair
    instead of the whole frame at once.

    Returns (x, y, u, v, valid, elapsed) -- valid here is directly from
    each tile's val_locations (True = invalid, inverted to the "valid"
    convention), BEFORE any outlier-std/replace/smooth post-processing;
    see process_frames_tiled() for the full pipeline including that."""
    H, W = frame_a.shape
    tiles = compute_tiles((H, W), n_tiles_y, n_tiles_x, margin_px)

    xs, ys, us, vs, valids = [], [], [], [], []
    elapsed_total = 0.0
    scaling_par = None
    min_core = None

    for i, tile in enumerate(tiles):
        y0, y1, x0, x1 = tile["core"]
        py0, py1, px0, px1 = tile["padded"]
        tile_a = frame_a[py0:py1, px0:px1]
        tile_b = frame_b[py0:py1, px0:px1]

        process, tx, ty = init_raw_fn(tile_a.shape)
        if scaling_par is None:
            scaling_par = process.scaling_par
            # Confirmed on real GPU hardware: a tile's core can be too
            # small to safely exclude its halo, WITHOUT tripping piv_gpu's
            # own too-small-to-process assertion (that only guards the
            # padded region, not the core/lattice-alignment) -- at
            # min_core well below min_search_size, the stitched output
            # silently gained 120 extra, misaligned vectors on a real
            # test frame (961 vs. the correct 841) instead of erroring or
            # producing a clean gap. min_core >= min_search_size (the
            # finest pass's own window size) was confirmed clean; smaller
            # sizes were the ones that broke, so this check has real
            # margin, not just the exact observed threshold.
            min_core = min(t["core"][1] - t["core"][0] for t in tiles)
            min_core = min(min_core, min(t["core"][3] - t["core"][2] for t in tiles))
            if min_core < process.min_search_size:
                raise ValueError(
                    f"run_tiled: {n_tiles_y}x{n_tiles_x} tiles gives a smallest tile core of "
                    f"{min_core}px, below the finest pass's window size ({process.min_search_size}px) "
                    f"-- the stitched result would silently be wrong (extra/misaligned vectors), not "
                    f"just less accurate. Use fewer tiles for this frame size, or a smaller finest "
                    f"window/larger frame."
                )

        t0 = time.time()
        u, v = process(tile_a, tile_b)
        elapsed_total += time.time() - t0
        if report_gpu_mem and getattr(ctrl, "verbose", False):
            gpu_free_report()

        val_locations = np.asarray(process.val_locations)
        del process
        if free_pools_fn is not None:
            free_pools_fn()

        # tile-local (still un-flipped, image-row-order) -> global coords
        gx = tx + px0
        gy = ty + py0

        # keep only vectors whose GLOBAL location falls in this tile's
        # exclusive core -- discards the halo, which exists only so this
        # tile's own windows had real context, not to be double-counted
        keep = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)

        xs.append(gx[keep]); ys.append(gy[keep])
        us.append(u[keep]); vs.append(v[keep])
        valids.append(~val_locations[keep])

        if getattr(ctrl, "verbose", False):
            print(f"  tile {i + 1}/{len(tiles)} (row {tile['row']}, col {tile['col']}): "
                  f"{int(keep.sum())} vectors, {elapsed_total:.3f}s cumulative")

    x = np.concatenate(xs)
    u = np.concatenate(us)
    v = np.concatenate(vs)
    valid = np.concatenate(valids)
    # single global y-flip using the FULL frame's height, not each tile's
    y = H * scaling_par - np.concatenate(ys)

    return x, y, u, v, valid, elapsed_total
