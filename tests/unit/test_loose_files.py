"""Regression tests for io/loose_files.py's suffix-based frame-A/B
pairing -- in particular that it's case-INsensitive (Windows/macOS
filesystems are case-insensitive; users shouldn't have to match the exact
case of their suffix field to their actual file extensions/suffixes)."""

import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v3")

from piv_suite.io.loose_files import iter_pairs_from_loose_files


def _write_png(path, value):
    imageio.imwrite(path, np.full((4, 4), value, dtype=np.uint8))


def test_suffix_matching_is_case_insensitive(tmp_path):
    # actual files use an uppercase extension/suffix ("_a.TIF"/"_b.TIF")
    _write_png(tmp_path / "0001_a.TIF", 10)
    _write_png(tmp_path / "0001_b.TIF", 20)

    # suffix fields typed in lowercase should still match
    pairs = list(iter_pairs_from_loose_files(str(tmp_path), loose_glob="*.TIF",
                                              suffix_a="_a.tif", suffix_b="_b.tif"))
    assert len(pairs) == 1
    pair_id, frame_a, frame_b = pairs[0]
    assert pair_id == "0001"
    assert frame_a[0, 0] == 10
    assert frame_b[0, 0] == 20


def test_suffix_matching_case_insensitive_reverse(tmp_path):
    # actual files use a lowercase suffix, GUI field typed in uppercase
    _write_png(tmp_path / "0002_a.tif", 30)
    _write_png(tmp_path / "0002_b.tif", 40)

    pairs = list(iter_pairs_from_loose_files(str(tmp_path), loose_glob="*.tif",
                                              suffix_a="_A.TIF", suffix_b="_B.TIF"))
    assert len(pairs) == 1
    assert pairs[0][0] == "0002"


def test_suffix_matching_no_match_still_exits(tmp_path):
    _write_png(tmp_path / "0003_x.tif", 50)
    with pytest.raises(SystemExit):
        list(iter_pairs_from_loose_files(str(tmp_path), loose_glob="*.tif",
                                          suffix_a="_a.tif", suffix_b="_b.tif"))
