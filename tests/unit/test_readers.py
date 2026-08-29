"""Direct unit tests for io/readers.py -- previously only exercised
indirectly through io/loose_files.py's own tests (test_loose_files.py),
which cover suffix-matching, not readers.py's own contract: is_supported(),
the unrecognized-extension error path, and RGB collapse-to-grayscale
(loose_files.py's existing tests only ever write single-channel/grayscale
test images, never a real RGB one)."""

import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v3")

from piv_suite.io.readers import GENERIC_IMAGE_EXTENSIONS, IM7_EXTENSIONS, is_supported, read_frame


@pytest.mark.parametrize("path,expected", [
    ("frame.im7", True),
    ("frame.IM7", True),  # case-insensitive, matching loose_files.py's own convention
    ("frame.tif", True),
    ("frame.tiff", True),
    ("frame.png", True),
    ("frame.bmp", True),
    ("frame.jpg", True),
    ("frame.jpeg", True),
    ("frame.raw", False),
    ("frame.txt", False),
    ("frame", False),  # no extension at all
])
def test_is_supported(path, expected):
    assert is_supported(path) is expected


def test_is_supported_extensions_are_disjoint_and_match_read_frame_dispatch():
    # is_supported()'s own claim (both sets, unioned) must actually match
    # what read_frame() dispatches on -- a regression here would silently
    # make is_supported() lie about what read_frame() can actually handle.
    assert IM7_EXTENSIONS.isdisjoint(GENERIC_IMAGE_EXTENSIONS)


def test_read_frame_unsupported_extension_raises_value_error(tmp_path):
    bogus = tmp_path / "frame.xyz"
    bogus.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="no reader registered"):
        read_frame(str(bogus))


def test_read_frame_png_grayscale_roundtrip(tmp_path):
    path = tmp_path / "frame.png"
    original = np.full((6, 8), 123, dtype=np.uint8)
    imageio.imwrite(path, original)

    arr = read_frame(str(path))

    assert arr.ndim == 2
    assert arr.shape == (6, 8)
    np.testing.assert_array_equal(arr, original)


def test_read_frame_collapses_rgb_to_grayscale_intensity(tmp_path):
    # PIV correlates on intensity, not color -- a real RGB image (not
    # exercised by loose_files.py's own tests, which only ever write
    # single-channel data) must come back 2-D, averaged over R/G/B.
    path = tmp_path / "frame.png"
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 60
    rgb[..., 2] = 90
    imageio.imwrite(path, rgb)

    arr = read_frame(str(path))

    assert arr.ndim == 2
    assert arr.shape == (4, 5)
    np.testing.assert_allclose(arr, 60.0)  # mean(30, 60, 90) == 60


def test_read_frame_collapses_rgba_to_grayscale_ignoring_alpha(tmp_path):
    # arr[..., :3].mean(...) explicitly drops the alpha channel -- lock
    # that in directly, since a transparent/RGBA PNG is a realistic input
    # this app has no other test coverage for.
    path = tmp_path / "frame.png"
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[..., 0] = 30
    rgba[..., 1] = 60
    rgba[..., 2] = 90
    rgba[..., 3] = 255  # fully opaque alpha -- must not affect the intensity average
    imageio.imwrite(path, rgba)

    arr = read_frame(str(path))

    assert arr.ndim == 2
    np.testing.assert_allclose(arr, 60.0)


def test_read_frame_tif_extension_uses_tifffile_backend(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "frame.tif"
    original = np.full((6, 8), 200, dtype=np.uint16)
    tifffile.imwrite(path, original)

    arr = read_frame(str(path))

    assert arr.ndim == 2
    np.testing.assert_array_equal(arr, original)
