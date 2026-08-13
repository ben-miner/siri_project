import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

preamble = import_module("04a_preamble")


def test_parse_take_splits_on_first_underscore():
    assert preamble.parse_take("s01_passA_modal") == ("s01", "passA_modal")


def test_parse_take_rejects_missing_underscore():
    with pytest.raises(ValueError):
        preamble.parse_take("noseparator")


def test_find_textgrids_ignores_gitkeep(tmp_path):
    (tmp_path / ".gitkeep").write_bytes(b"")
    (tmp_path / "s01_passA_modal.TextGrid").write_text("", encoding="utf-8")
    (tmp_path / "s02_passC_creak.TextGrid").write_text("", encoding="utf-8")

    found = preamble.find_textgrids(tmp_path)

    assert [p.name for p in found] == ["s01_passA_modal.TextGrid", "s02_passC_creak.TextGrid"]


def test_validate_intervals_accepts_expected_labels():
    intervals = {label: (0.0, 1.0) for label in preamble.EXPECTED_LABELS}
    preamble.validate_intervals(intervals, "s01_passA_modal")  # should not raise


def test_validate_intervals_rejects_missing_label():
    intervals = {label: (0.0, 1.0) for label in preamble.EXPECTED_LABELS[:-1]}
    with pytest.raises(ValueError):
        preamble.validate_intervals(intervals, "s01_passA_modal")


def test_validate_intervals_rejects_extra_label():
    intervals = {label: (0.0, 1.0) for label in preamble.EXPECTED_LABELS}
    intervals["unexpected"] = (5.0, 6.0)
    with pytest.raises(ValueError):
        preamble.validate_intervals(intervals, "s01_passA_modal")


def test_build_index_rows_carries_sentence_block_start_on_every_row():
    intervals = {
        "roomtone": (0.0, 5.0),
        "sustained_a": (5.0, 8.0),
        "glide": (8.0, 11.0),
        "creak_cal": (11.0, 14.0),
        "carrier": (14.0, 17.0),
    }
    rows = preamble.build_index_rows("s01_passA_modal", intervals, sentence_block_start_sec=17.0)

    assert len(rows) == 5
    assert [r["label"] for r in rows] == preamble.EXPECTED_LABELS
    assert all(r["sentence_block_start_sec"] == 17.0 for r in rows)
    assert rows[0]["start_sec"] == 0.0 and rows[0]["end_sec"] == 5.0
    assert all(r["take"] == "s01_passA_modal" for r in rows)


def test_read_labelled_intervals_rejects_duplicate_label(tmp_path):
    from praatio import textgrid as ptg
    from praatio.utilities.constants import Interval

    tg = ptg.Textgrid()
    tier = ptg.IntervalTier(
        "metadata",
        [
            Interval(0.0, 1.0, "roomtone"),
            Interval(1.0, 2.0, "roomtone"),
        ],
        0.0,
        2.0,
    )
    tg.addTier(tier)
    tg_path = tmp_path / "dup.TextGrid"
    tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)

    with pytest.raises(ValueError):
        preamble.read_labelled_intervals(tg_path)


@pytest.fixture
def synthetic_take(tmp_path):
    """Build a real TextGrid + real mono wav for one take, matching the
    project's actual preamble structure, so process_take can be exercised
    end-to-end without touching real project data."""
    import parselmouth
    from praatio import textgrid as ptg
    from praatio.utilities.constants import Interval

    take = "s99_passX_test"
    labels_and_durations = [
        ("roomtone", 1.0),
        ("", 0.2),
        ("sustained_a", 1.0),
        ("", 0.2),
        ("glide", 1.0),
        ("", 0.2),
        ("creak_cal", 1.0),
        ("", 0.2),
        ("carrier", 1.0),
        ("", 2.0),  # sentence block content
    ]

    intervals = []
    t = 0.0
    expected = {}
    for label, dur in labels_and_durations:
        start, end = t, t + dur
        intervals.append(Interval(start, end, label))
        if label:
            expected[label] = (start, end)
        t = end
    total_duration = t

    tg = ptg.Textgrid()
    tier = ptg.IntervalTier("metadata", intervals, 0.0, total_duration)
    tg.addTier(tier)
    tg_dir = tmp_path / "textgrids"
    tg_dir.mkdir()
    tg_path = tg_dir / f"{take}.TextGrid"
    tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)

    sound = parselmouth.Sound(
        values=[[0.0] * int(total_duration * 48000)],
        sampling_frequency=48000,
    )
    audio_dir = tmp_path / "converted"
    audio_dir.mkdir()
    sound.save(str(audio_dir / f"{take}.wav"), parselmouth.SoundFileFormat.WAV)

    return {
        "take": take,
        "tg_path": tg_path,
        "audio_dir": audio_dir,
        "expected_intervals": expected,
        "total_duration": total_duration,
    }


def test_process_take_writes_calibration_clips_and_block(tmp_path, synthetic_take):
    calibration_dir = tmp_path / "calibration"
    blocks_dir = tmp_path / "blocks"

    rows = preamble.process_take(
        synthetic_take["tg_path"], synthetic_take["audio_dir"], calibration_dir, blocks_dir
    )

    assert len(rows) == 5
    for label in preamble.EXPECTED_LABELS:
        clip_path = calibration_dir / f"s99_passX_test_{label}.wav"
        assert clip_path.exists()

    block_path = blocks_dir / "s99_passX_test_block.wav"
    assert block_path.exists()

    import parselmouth
    carrier_end = synthetic_take["expected_intervals"]["carrier"][1]
    block_sound = parselmouth.Sound(str(block_path))
    expected_block_duration = synthetic_take["total_duration"] - carrier_end
    assert block_sound.duration == pytest.approx(expected_block_duration, abs=0.01)

    roomtone_sound = parselmouth.Sound(str(calibration_dir / "s99_passX_test_roomtone.wav"))
    expected_roomtone_duration = (
        synthetic_take["expected_intervals"]["roomtone"][1]
        - synthetic_take["expected_intervals"]["roomtone"][0]
    )
    assert roomtone_sound.duration == pytest.approx(expected_roomtone_duration, abs=0.01)
    assert roomtone_sound.sampling_frequency == 48000


def test_process_take_fails_loudly_when_audio_missing(tmp_path, synthetic_take):
    empty_audio_dir = tmp_path / "no_audio_here"
    empty_audio_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        preamble.process_take(
            synthetic_take["tg_path"], empty_audio_dir, tmp_path / "cal", tmp_path / "blk"
        )
