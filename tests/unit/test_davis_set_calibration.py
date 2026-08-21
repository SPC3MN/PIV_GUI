"""Tests for davis_set.py's calibration auto-extraction (pixel pitch +
frame dt) from a DaVis .set project.

The XML-parsing tests are the important regression coverage: a real
Settings_Acquisition_Timing_*.xml contains multiple <timespan> elements,
and a naive "first one anywhere" query silently returns the wrong value
(0.0 instead of the real dt) rather than raising -- since frame_dt_s is
later used as a divisor, that would corrupt every velocity in a field
instead of failing loudly. See _frame_dt_from_timing_xml's docstring.
"""

import os
import xml.etree.ElementTree as ET

import pytest

from piv_suite.io.davis_set import (
    _find_timing_xml, _frame_dt_from_timing_xml, _read_pixel_pitch_mm,
    read_calibration_from_set, read_stereo_calibration_from_set,
)

REAL_STREAMSET = (
    r"J:\Final_Planar_Swirl\Recording_Date=260722_Time=222343"
    r"\On Time=0.7_Burst On Time=0.0_Burst Off Time=0.0.set"
)
REAL_LAVISION_SAMPLE_SET = (
    r"C:\Users\Germiel\Downloads\PIV_COMP\PIV_COMP\Lavision_Sample"
    r"\PIV_MP(3x32x32_75%ov_ImgCorr).set"
)

REAL_TIMING_XML = """<?xml version="1.0"?>
<root>
  <Offsets>
    <item><timespan class="int64" value="0" /></item>
    <item><timespan class="int64" value="0" /></item>
  </Offsets>
  <aligner class="list">
    <item>
      <aligner class="Acq::DoubleFrameAligner">
        <dt class="RTE::TimeSpan"><timespan class="int64" value="700000000" /></dt>
      </aligner>
    </item>
  </aligner>
</root>
"""


class _TitledDataset:
    def __init__(self, title):
        self.title = title


class _FakeScales:
    def __init__(self, x_slope):
        self.x = type("Axis", (), {"slope": x_slope})()


class _FakeFrame:
    def __init__(self, x_slope):
        self.scales = _FakeScales(x_slope)


class _FakeFrameBroken:
    @property
    def scales(self):
        raise AttributeError("no scales on this frame")


class _FakeBuffer:
    def __init__(self, frames):
        self.frames = frames


class _FakeDataset:
    def __init__(self, buffers, title="fake_title"):
        self._buffers = buffers
        self.title = title
        self.close_count = 0

    def __getitem__(self, i):
        return self._buffers[i]

    def close(self):
        self.close_count += 1


# ---- pure XML parsing (no lvpyio needed) ----

def test_frame_dt_uses_aligner_dt_not_first_timespan(tmp_path):
    xml_path = tmp_path / "Settings_Acquisition_Timing_{guid}.xml"
    xml_path.write_text(REAL_TIMING_XML)
    assert _frame_dt_from_timing_xml(str(xml_path)) == pytest.approx(0.0007)


def test_frame_dt_missing_aligner_returns_none(tmp_path):
    xml_path = tmp_path / "Settings_Acquisition_Timing_{guid}.xml"
    xml_path.write_text('<?xml version="1.0"?><root><Offsets/></root>')
    assert _frame_dt_from_timing_xml(str(xml_path)) is None


def test_frame_dt_malformed_xml_returns_none(tmp_path):
    xml_path = tmp_path / "Settings_Acquisition_Timing_{guid}.xml"
    xml_path.write_text("not xml at all <<<")
    assert _frame_dt_from_timing_xml(str(xml_path)) is None


def test_find_timing_xml_glob_matches_guid_suffixed_filename(tmp_path):
    set_path = str(tmp_path / "myrecording.set")
    data_dir = tmp_path / "myrecording"
    data_dir.mkdir()
    xml_path = data_dir / "Settings_Acquisition_Timing_{5cacfa80-0ada-4250-b7f8-b3db6724dbd9}.xml"
    xml_path.write_text(REAL_TIMING_XML)

    found = _find_timing_xml(set_path, _TitledDataset("myrecording"), is_multiset_sub=False)
    assert found == str(xml_path)


def test_find_timing_xml_multiset_nests_under_set_label_then_title(tmp_path):
    set_path = str(tmp_path / "TopLevel.set")
    nested_dir = tmp_path / "TopLevel" / "SubEntryTitle"
    nested_dir.mkdir(parents=True)
    xml_path = nested_dir / "Settings_Acquisition_Timing_{guid}.xml"
    xml_path.write_text(REAL_TIMING_XML)

    found = _find_timing_xml(set_path, _TitledDataset("SubEntryTitle"), is_multiset_sub=True)
    assert found == str(xml_path)

    # the single-set (non-multiset) layout must NOT find this nested file --
    # locks in the verified divergence between the two directory layouts.
    not_found = _find_timing_xml(set_path, _TitledDataset("SubEntryTitle"), is_multiset_sub=False)
    assert not_found is None


# ---- pixel pitch, lvpyio monkeypatched with fake stand-ins ----

def test_read_calibration_reads_pixel_pitch_from_scales_x_slope(monkeypatch):
    lv = pytest.importorskip("lvpyio")
    fake_buf = _FakeBuffer([_FakeFrame(0.0514883), _FakeFrame(0.0514883)])
    fake_dataset = _FakeDataset([fake_buf])
    monkeypatch.setattr(lv, "is_multiset", lambda p: False)
    monkeypatch.setattr(lv, "read_set", lambda p: fake_dataset)

    result = read_calibration_from_set("fake.set")
    assert result.pixel_pitch_mm == pytest.approx(0.0514883)
    assert fake_dataset.close_count == 1


def test_read_calibration_pixel_pitch_survives_single_frame_buffer(monkeypatch):
    lv = pytest.importorskip("lvpyio")
    # replicates the real Lavision_Sample processing-job .set: only 1
    # frame per buffer, which frames_from_buffer() would reject -- pixel
    # pitch extraction must not route through it.
    fake_buf = _FakeBuffer([_FakeFrame(0.0514883)])
    fake_dataset = _FakeDataset([fake_buf])
    monkeypatch.setattr(lv, "is_multiset", lambda p: False)
    monkeypatch.setattr(lv, "read_set", lambda p: fake_dataset)

    result = read_calibration_from_set("fake.set")
    assert result.pixel_pitch_mm == pytest.approx(0.0514883)


def test_read_calibration_pixel_pitch_failure_returns_none_not_raise(monkeypatch):
    lv = pytest.importorskip("lvpyio")
    fake_buf = _FakeBuffer([_FakeFrameBroken()])
    fake_dataset = _FakeDataset([fake_buf])
    monkeypatch.setattr(lv, "is_multiset", lambda p: False)
    monkeypatch.setattr(lv, "read_set", lambda p: fake_dataset)

    result = read_calibration_from_set("fake.set")
    assert result.pixel_pitch_mm is None


def test_read_pixel_pitch_helper_failure_returns_none_not_raise():
    assert _read_pixel_pitch_mm(_FakeDataset([_FakeBuffer([_FakeFrameBroken()])])) is None


# ---- real-data-gated: skipped unless the local paths exist ----

@pytest.mark.skipif(not os.path.exists(REAL_STREAMSET), reason="real StreamSet not available on this machine")
def test_read_calibration_from_real_streamset():
    pytest.importorskip("lvpyio")
    result = read_calibration_from_set(REAL_STREAMSET)
    assert result.pixel_pitch_mm == pytest.approx(0.0514883, rel=1e-4)
    assert result.frame_dt_s == pytest.approx(0.0007, rel=1e-6)


@pytest.mark.skipif(not os.path.exists(REAL_LAVISION_SAMPLE_SET), reason="bundled sample not available on this machine")
def test_read_calibration_from_bundled_sample_partial_extraction():
    pytest.importorskip("lvpyio")
    # this sample is a DaVis post-processing job set: single-frame
    # buffers, and no timing XML anywhere in its tree -- a real (not
    # synthetic) exercise of the graceful partial-failure path.
    result = read_calibration_from_set(REAL_LAVISION_SAMPLE_SET)
    assert result.pixel_pitch_mm == pytest.approx(0.0514883, rel=1e-4)
    assert result.frame_dt_s is None


# ---- stereo stub ----

def test_read_stereo_calibration_from_set_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        read_stereo_calibration_from_set("dummy.set")
