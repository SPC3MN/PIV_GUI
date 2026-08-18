"""Backend/mode selection: turn (backend, mode) into the right engine
constructor, without either engine needing to know the other exists.
"""

from . import cpu_engine, gpu_engine

BACKENDS = ("cpu", "gpu")
MODES = ("planar", "stereo")


def get_engine_factory(backend):
    """Return a callable factory(frame_shape, settings) -> (engine, x, y)
    for the given backend ("cpu" or "gpu"). `settings` is the backend-
    specific settings dict produced by config.legacy's adapters -- this
    function doesn't know or care about the canonical schema.

    For "gpu", settings must be {"min_search_size": int, "piv_settings": dict}.
    For "cpu", settings must be {"cpu_settings": dict}.
    """
    if backend == "cpu":
        def factory(frame_shape, settings):
            return cpu_engine.init_cpu_processor(frame_shape, settings["cpu_settings"])
        return factory
    if backend == "gpu":
        def factory(frame_shape, settings):
            return gpu_engine.init_gpu_processor(
                frame_shape, settings["min_search_size"], settings["piv_settings"])
        return factory
    raise ValueError(f"unknown backend {backend!r} -- expected one of {BACKENDS}")


def is_gpu_available():
    return gpu_engine.is_gpu_available()
