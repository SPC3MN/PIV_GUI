#!/usr/bin/env python3
"""Convert an old flat config JSON (from one of the four original repos --
stereo_piv_config.json, planar_piv_config.json, planar_cpu_piv_config.json,
stereo_cpu_piv_config.json) into the new unified `.pivproj` schema.

Usage:
    python scripts/migrate_legacy_config.py old_config.json new_config.pivproj

Backend (cpu/gpu) and mode (planar/stereo) are auto-detected from which
keys are present in the legacy file: "cpu_settings" -> cpu backend,
"piv_settings"+"min_search_size" -> gpu backend; "cam0_mapping" present ->
stereo mode, absent -> planar mode.
"""

import json
import sys

sys.path.insert(0, "src")

from piv_suite.config.io import save_project
from piv_suite.config.legacy import passes_from_cpu, passes_from_gpu
from piv_suite.config.schema import (
    CalibrationSettings, CameraMappingSettings, CorrelationSettings,
    OutputSettings, PostProcessSettings, ProjectConfig, ProjectSettings,
    RangeFilterSettings, StereoSettings, ValidationSettings,
)


def migrate(legacy: dict) -> ProjectConfig:
    is_gpu = "piv_settings" in legacy and "min_search_size" in legacy
    is_stereo = "cam0_mapping" in legacy

    project = ProjectSettings(
        input_mode=legacy.get("input_mode", "set"),
        input_path=legacy.get("input_path", ""),
        output_dir=legacy.get("output_dir", "piv_output"),
        backend="gpu" if is_gpu else "cpu",
        mode="stereo" if is_stereo else "planar",
        multiset_index=legacy.get("multiset_index", 0),
        loose_glob=legacy.get("loose_glob", "*.im7"),
        suffix_a=legacy.get("suffix_a", "_a.im7"),
        suffix_b=legacy.get("suffix_b", "_b.im7"),
        suffix_cam0=legacy.get("suffix_cam0", "_cam1.im7"),
        suffix_cam1=legacy.get("suffix_cam1", "_cam2.im7"),
        stereo_frame_order=legacy.get("stereo_frame_order", "camera_major"),
    )

    if is_gpu:
        piv_settings = legacy.get("piv_settings", {})
        passes = passes_from_gpu(
            legacy["min_search_size"],
            piv_settings.get("search_size_iters", (1,)),
            piv_settings.get("overlap_ratio", (0.5,)),
        )
        correlation = CorrelationSettings(
            passes=passes,
            dt=piv_settings.get("dt", 1.0),
            subpixel_method=piv_settings.get("subpixel_method", "gaussian"),
            batch_size=piv_settings.get("batch_size"),
        )
        validation = ValidationSettings(
            sig2noise_method=piv_settings.get("s2n_method", "peak2mean"),
            sig2noise_threshold=piv_settings.get("s2n_tol", 1.05),
            filter_method=piv_settings.get("replacing_method", "localmean"),
            max_filter_iteration=piv_settings.get("num_replacing_iters", 4),
            filter_kernel_size=piv_settings.get("replacing_size", 2),
            validation_first_pass=piv_settings.get("revalidate", True),
            smoothn=piv_settings.get("smooth", False),
            smoothn_p=piv_settings.get("smoothing_par", 0.05),
        )
        tiling = legacy.get("tiling", {})
        correlation.use_tiling = tiling.get("enabled", False)
        correlation.n_tiles_y = tiling.get("n_tiles_y", 1)
        correlation.n_tiles_x = tiling.get("n_tiles_x", 1)
        correlation.tile_margin_px = tiling.get("margin_px")
    else:
        cpu_settings = legacy.get("cpu_settings", {})
        passes = passes_from_cpu(
            cpu_settings.get("windowsizes", [64, 32, 32, 32]),
            cpu_settings.get("overlap", [32, 24, 24, 24]),
        )
        correlation = CorrelationSettings(
            passes=passes,
            dt=cpu_settings.get("dt", 1.0),
            correlation_method=cpu_settings.get("correlation_method", "circular"),
            subpixel_method=cpu_settings.get("subpixel_method", "gaussian"),
            deformation_method=cpu_settings.get("deformation_method", "symmetric"),
            interpolation_order=cpu_settings.get("interpolation_order", 3),
        )
        validation = ValidationSettings(
            sig2noise_method=cpu_settings.get("sig2noise_method", "peak2mean"),
            sig2noise_threshold=cpu_settings.get("sig2noise_threshold", 1.05),
            sig2noise_validate=cpu_settings.get("sig2noise_validate", True),
            validation_first_pass=cpu_settings.get("validation_first_pass", True),
            replace_vectors=cpu_settings.get("replace_vectors", True),
            filter_method=cpu_settings.get("filter_method", "localmean"),
            max_filter_iteration=cpu_settings.get("max_filter_iteration", 4),
            filter_kernel_size=cpu_settings.get("filter_kernel_size", 2),
            smoothn=cpu_settings.get("smoothn", False),
            smoothn_p=cpu_settings.get("smoothn_p", 0.05),
        )

    postprocess = PostProcessSettings(
        # legacy["apply_v_sign_flip"] is intentionally dropped -- the
        # v-sign-flip option was removed from the new schema entirely
        global_outlier_std=legacy.get("global_outlier_std"),
        range_filter=RangeFilterSettings(),  # didn't exist in any legacy config
        replace_invalid=legacy.get("replace_invalid", False),
        smooth_field=legacy.get("smooth_field", False),
        smooth_sigma=legacy.get("smooth_sigma", 1.0),
    )

    calibration = CalibrationSettings(
        pixel_pitch_mm=legacy.get("pixel_pitch_mm"),
        frame_dt_s=legacy.get("frame_dt_s"),
    )

    stereo = StereoSettings()
    if is_stereo:
        stereo = StereoSettings(
            cam0_mapping=CameraMappingSettings(**legacy["cam0_mapping"]),
            cam1_mapping=CameraMappingSettings(**legacy["cam1_mapping"]),
            world_shape=tuple(legacy.get("world_shape", (0, 0))),
            world_scale_px_per_mm=legacy.get("world_scale_px_per_mm", 1.0),
            dewarp_order=legacy.get("dewarp_order", 1),
            alpha1_deg=legacy.get("alpha1_deg", 0.0),
            alpha2_deg=legacy.get("alpha2_deg", 0.0),
            beta1_deg=legacy.get("beta1_deg", 0.0),
            beta2_deg=legacy.get("beta2_deg", 0.0),
        )

    output = OutputSettings(
        save_npz=legacy.get("save_npz", True),
        save_plot=legacy.get("save_plot", False),
        save_summary_csv=legacy.get("save_summary_csv", False),
        plot_dpi=legacy.get("plot_dpi", 150),
        quiver_scale=legacy.get("quiver_scale", 1000),
        show_plots=legacy.get("show_plots", False),
        verbose=legacy.get("verbose", True),
    )

    return ProjectConfig(project=project, correlation=correlation, validation=validation,
                          postprocess=postprocess, calibration=calibration, stereo=stereo,
                          output=output)


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} old_config.json new_config.pivproj")
    with open(sys.argv[1]) as f:
        legacy = json.load(f)
    cfg = migrate(legacy)
    save_project(sys.argv[2], cfg)
    print(f"Migrated '{sys.argv[1]}' -> '{sys.argv[2]}' "
          f"(backend={cfg.project.backend}, mode={cfg.project.mode})")


if __name__ == "__main__":
    main()
