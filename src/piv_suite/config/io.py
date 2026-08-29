"""JSON <-> ProjectConfig round-trip (`.pivproj` files), generalizing
piv_common.load_controls()'s "auto-write defaults on first run, user keys
override, everything else falls back to defaults" UX to the new nested
dataclass schema.
"""

import dataclasses
import json
import os

from .schema import (
    CalibrationSettings, CameraMappingSettings, CorrelationSettings,
    DualPlanarCameraSettings, DualPlanarSettings, OutputSettings, PassSettings,
    PerformanceSettings, PostProcessSettings, PreprocessSettings, ProjectConfig,
    ProjectSettings, RangeFilterSettings, StereoSettings, ValidationSettings,
)


def to_dict(config: ProjectConfig) -> dict:
    return dataclasses.asdict(config)


def _passes_from_dicts(items):
    return [PassSettings(**d) for d in items]


def _filtered_kwargs(cls, d):
    """Drop unknown/stale keys before constructing `cls` from a dict --
    protects from_dict against a TypeError when an older .pivproj file
    has fields a newer schema version removed (e.g. ValidationSettings'
    sig2noise_*/validation_first_pass/replace_vectors fields, removed
    when validation moved to PostProcessSettings). Applied to every
    section below so a future field removal doesn't hard-break loading
    every pre-existing project file the same way."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        print(f"[warn] config.io: dropping unknown/stale keys {sorted(unknown)} "
              f"for {cls.__name__} (likely from an older .pivproj file)")
    return {k: v for k, v in d.items() if k in valid}


def from_dict(d: dict) -> ProjectConfig:
    """Reconstruct a typed ProjectConfig from a plain (JSON-loaded) dict.
    Explicit per-section reconstruction rather than generic reflection --
    the schema is small and stable enough that this is more robust than a
    generic dict->dataclass walker, and keeps failures localized/legible
    (a bad section raises with that section's name in the traceback)."""
    project = ProjectSettings(**_filtered_kwargs(ProjectSettings, d.get("project", {})))

    preprocess = PreprocessSettings(**_filtered_kwargs(PreprocessSettings, d.get("preprocess", {})))

    corr_d = dict(d.get("correlation", {}))
    if "passes" in corr_d:
        corr_d["passes"] = _passes_from_dicts(corr_d["passes"])
    correlation = CorrelationSettings(**_filtered_kwargs(CorrelationSettings, corr_d))

    validation = ValidationSettings(**_filtered_kwargs(ValidationSettings, d.get("validation", {})))

    post_d = dict(d.get("postprocess", {}))
    if "range_filter" in post_d and post_d["range_filter"] is not None:
        post_d["range_filter"] = RangeFilterSettings(**_filtered_kwargs(RangeFilterSettings, post_d["range_filter"]))
    postprocess = PostProcessSettings(**_filtered_kwargs(PostProcessSettings, post_d))

    calibration = CalibrationSettings(**_filtered_kwargs(CalibrationSettings, d.get("calibration", {})))

    stereo_d = dict(d.get("stereo", {}))
    for key in ("cam0_mapping", "cam1_mapping", "cam0_mapping_plane2", "cam1_mapping_plane2"):
        if key in stereo_d and stereo_d[key] is not None:
            stereo_d[key] = CameraMappingSettings(**_filtered_kwargs(CameraMappingSettings, stereo_d[key]))
    if "world_shape" in stereo_d and stereo_d["world_shape"] is not None:
        stereo_d["world_shape"] = tuple(stereo_d["world_shape"])
    stereo = StereoSettings(**_filtered_kwargs(StereoSettings, stereo_d))

    dp_d = dict(d.get("dual_planar", {}))
    for key in ("cam0", "cam1"):
        if key in dp_d and dp_d[key] is not None:
            dp_d[key] = DualPlanarCameraSettings(**_filtered_kwargs(DualPlanarCameraSettings, dp_d[key]))
    dual_planar = DualPlanarSettings(**_filtered_kwargs(DualPlanarSettings, dp_d))

    output = OutputSettings(**_filtered_kwargs(OutputSettings, d.get("output", {})))

    performance = PerformanceSettings(**_filtered_kwargs(PerformanceSettings, d.get("performance", {})))

    return ProjectConfig(
        project=project, preprocess=preprocess, correlation=correlation,
        validation=validation, postprocess=postprocess,
        calibration=calibration, stereo=stereo, dual_planar=dual_planar,
        output=output, performance=performance,
    )


def _deep_merge(base, overrides):
    """Recursively merge `overrides` onto a copy of `base` (dicts only --
    lists/scalars in overrides replace base's value outright, matching
    JSON's natural semantics)."""
    merged = dict(base)
    for key, val in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_project(config_path: str, defaults: ProjectConfig = None) -> ProjectConfig:
    """Load a `.pivproj` JSON file onto a ProjectConfig, creating the file
    (populated with `defaults`, or ProjectConfig()'s own defaults if not
    given) if it doesn't exist yet. Keys present in the file override the
    defaults'; anything the file doesn't mention falls back to the
    default -- only settings actually being changed need to be in the
    file, same UX as the original load_controls()."""
    if defaults is None:
        defaults = ProjectConfig()
    default_dict = to_dict(defaults)

    if os.path.exists(config_path):
        with open(config_path) as f:
            text = f.read()
        try:
            user_dict = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"'{config_path}' isn't valid JSON ({e.msg} at line {e.lineno}, column "
                f"{e.colno}) -- .pivproj files are meant to be hand-editable, so this is "
                f"most likely a syntax slip (a trailing comma, an unmatched quote/brace) "
                f"from a manual edit, or a write that was interrupted mid-save (killed "
                f"process, disk full, power loss). Fix the file directly, restore it from "
                f"a backup, or delete it to have a fresh default one written in its place."
            ) from e
        merged = _deep_merge(default_dict, user_dict)
        print(f"[info] loaded config from '{config_path}'")
    else:
        merged = default_dict
        save_project(config_path, defaults)
        print(f"[info] '{config_path}' didn't exist -- wrote defaults there. "
              "Edit it (input_path, correlation passes, etc.) and re-run to "
              "customize.")

    return from_dict(merged)


def save_project(config_path: str, config: ProjectConfig) -> None:
    """Write `config` to `config_path` atomically (write to a temp file in
    the same directory, then os.replace() it into place) -- this app's own
    CLI calls save_project() unconditionally on EVERY invocation (to
    persist CLI overrides back), so a naive direct write left a real
    corruption window: a process killed mid-write (Ctrl+C, closed
    terminal, disk full, power loss) would leave a truncated .pivproj file
    that the next run's load_project() couldn't parse, with no way back
    except hand-editing or deleting the file -- silently destroying
    whatever settings were already saved. os.replace() is atomic on both
    Windows and POSIX (same pattern already used for checkpoint files in
    scripts/compare_dataset.py), so this can only ever leave either the
    complete old file or the complete new one, never a partial write."""
    tmp_path = f"{config_path}.tmp{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(to_dict(config), f, indent=2)
        os.replace(tmp_path, config_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
