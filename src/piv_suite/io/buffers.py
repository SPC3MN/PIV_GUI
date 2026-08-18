"""Raw LaVision/DaVis buffer -> numpy frame extraction.

Migrated unchanged from piv_common.frames_from_buffer /
stereo_common.frames_from_stereo_buffer (identical across all four source
repos).
"""


def frames_from_buffer(buf):
    """Pull raw intensity arrays for frame A and frame B out of a
    double-frame Buffer (single camera)."""
    if len(buf.frames) < 2:
        raise ValueError(
            f"expected a double-frame buffer (2 frames), got {len(buf.frames)} "
            "-- your im7s likely store frame A/B as separate files; use "
            "input_mode='loose' with suffix_a/suffix_b set correctly"
        )
    frame_a = buf.frames[0].images[0]
    frame_b = buf.frames[1].images[0]
    return frame_a, frame_b


def frames_from_stereo_buffer(buf, frame_order):
    """Pull raw intensity arrays for BOTH cameras' frame A/B out of a
    combined stereo Buffer's 4 frames (2 cameras x 2 exposures).

    lvpyio doesn't label which frame belongs to which camera -- the ORDER
    is an assumption (see stereo_frame_order in the config). Confirm it
    against your own set before trusting output; flip stereo_frame_order
    if the wrong pair of frames ends up dewarped by each camera's
    mapping."""
    if len(buf.frames) != 4:
        raise ValueError(
            f"expected a 4-frame stereo buffer (2 cameras x 2 frames), got "
            f"{len(buf.frames)} -- if your cameras are stored as SEPARATE "
            "double-frame files, use input_mode='loose' with "
            "suffix_cam0/suffix_cam1 set correctly"
        )
    f0, f1, f2, f3 = (f.images[0] for f in buf.frames)
    if frame_order == "camera_major":     # [cam0_A, cam0_B, cam1_A, cam1_B]
        return f0, f1, f2, f3
    elif frame_order == "frame_major":    # [cam0_A, cam1_A, cam0_B, cam1_B]
        return f0, f2, f1, f3
    raise ValueError(f"unknown stereo_frame_order {frame_order!r}")
