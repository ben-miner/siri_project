import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest
from praatio import textgrid as ptg
from praatio.utilities.constants import Interval

prep = import_module("08b_prepare_grids")

SONORANTS = {"m", "n", "l", "j", "w", "i", "a", "u"}


class FakeTier:
    def __init__(self, entries, minTimestamp, maxTimestamp):
        self.entries = entries
        self.minTimestamp = minTimestamp
        self.maxTimestamp = maxTimestamp


# -- load_sonorants -----------------------------------------------------------

def test_load_sonorants_reads_yaml_list(tmp_path):
    p = tmp_path / "sonorants.yaml"
    p.write_text("sonorants:\n  - a\n  - m\n  - n\n", encoding="utf-8")
    assert prep.load_sonorants(p) == {"a", "m", "n"}


def test_load_sonorants_exits_when_key_missing(tmp_path):
    p = tmp_path / "sonorants.yaml"
    p.write_text("other_key:\n  - a\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        prep.load_sonorants(p)


def test_load_sonorants_exits_when_list_empty(tmp_path):
    p = tmp_path / "sonorants.yaml"
    p.write_text("sonorants: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        prep.load_sonorants(p)


# -- build_creak_tier -----------------------------------------------------------

def test_build_creak_tier_fills_full_coverage_no_gaps():
    # obstruent [0,0.5) then sonorant [0.5,1.0) then obstruent [1.0,1.5)
    entries = [
        Interval(0.0, 0.5, "t"), Interval(0.5, 1.0, "a"), Interval(1.0, 1.5, "k"),
    ]
    tier = FakeTier(entries, 0.0, 1.5)
    creak = prep.build_creak_tier(tier, SONORANTS)
    assert creak.name == "creak"
    assert creak.minTimestamp == 0.0
    assert creak.maxTimestamp == 1.5
    starts_ends = [(e.start, e.end) for e in creak.entries]
    assert starts_ends == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]


def test_build_creak_tier_all_labels_empty():
    entries = [Interval(0.0, 0.5, "t"), Interval(0.5, 1.0, "a")]
    tier = FakeTier(entries, 0.0, 1.0)
    creak = prep.build_creak_tier(tier, SONORANTS)
    assert all(e.label == "" for e in creak.entries)


def test_build_creak_tier_boundaries_exactly_match_sonorant_span():
    entries = [Interval(0.0, 0.37, "t"), Interval(0.37, 0.81, "a"), Interval(0.81, 1.2, "k")]
    tier = FakeTier(entries, 0.0, 1.2)
    creak = prep.build_creak_tier(tier, SONORANTS)
    sonorant_intervals = [(e.start, e.end) for e in creak.entries if (e.start, e.end) == (0.37, 0.81)]
    assert sonorant_intervals == [(0.37, 0.81)]


def test_build_creak_tier_adjacent_sonorants_stay_separate_intervals():
    """Two adjacent sonorant phones (no gap between them) must produce TWO
    intervals in the creak tier with a boundary at the phone break, not one
    merged span -- boundaries mirror each phone's own interval exactly."""
    entries = [Interval(0.0, 0.3, "m"), Interval(0.3, 0.6, "a"), Interval(0.6, 0.9, "t")]
    tier = FakeTier(entries, 0.0, 0.9)
    creak = prep.build_creak_tier(tier, SONORANTS)
    starts_ends = [(e.start, e.end) for e in creak.entries]
    assert (0.0, 0.3) in starts_ends
    assert (0.3, 0.6) in starts_ends
    assert len(starts_ends) == 3  # not merged into one (0.0, 0.6) span


def test_build_creak_tier_leading_and_trailing_gaps_filled():
    entries = [Interval(0.2, 0.5, "a")]
    tier = FakeTier(entries, 0.0, 1.0)
    creak = prep.build_creak_tier(tier, SONORANTS)
    starts_ends = [(e.start, e.end) for e in creak.entries]
    assert starts_ends == [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]


def test_build_creak_tier_no_sonorants_is_one_blank_span():
    entries = [Interval(0.0, 0.5, "t"), Interval(0.5, 1.0, "k")]
    tier = FakeTier(entries, 0.0, 1.0)
    creak = prep.build_creak_tier(tier, SONORANTS)
    assert [(e.start, e.end) for e in creak.entries] == [(0.0, 1.0)]
    assert creak.entries[0].label == ""


def test_build_creak_tier_ignores_blank_labeled_entries():
    entries = [Interval(0.0, 0.3, ""), Interval(0.3, 0.6, "a"), Interval(0.6, 0.9, "")]
    tier = FakeTier(entries, 0.0, 0.9)
    creak = prep.build_creak_tier(tier, SONORANTS)
    starts_ends = [(e.start, e.end) for e in creak.entries]
    assert (0.3, 0.6) in starts_ends


def test_build_creak_tier_tolerates_merely_unsorted_input():
    """The phone tier's own entries should already be time-ordered, but
    sonorant spans are sorted defensively before the cursor walk -- input
    order alone must not raise."""
    entries = [Interval(0.5, 0.9, "a"), Interval(0.0, 0.4, "m")]  # not time-sorted
    tier = FakeTier(entries, 0.0, 0.9)
    creak = prep.build_creak_tier(tier, SONORANTS)
    assert [(e.start, e.end) for e in creak.entries] == [(0.0, 0.4), (0.4, 0.5), (0.5, 0.9)]


def test_build_creak_tier_rejects_genuinely_overlapping_spans():
    entries = [Interval(0.0, 0.6, "m"), Interval(0.4, 0.9, "a")]  # overlap 0.4-0.6
    tier = FakeTier(entries, 0.0, 0.9)
    with pytest.raises(ValueError):
        prep.build_creak_tier(tier, SONORANTS)


# -- prepare_grid (integration, real TextGrid I/O) -----------------------------------------------------------

def _write_source_grid(path: Path) -> None:
    tg = ptg.Textgrid(minTimestamp=0.0, maxTimestamp=1.0)
    phones = ptg.IntervalTier(
        "phones",
        [Interval(0.0, 0.3, "t"), Interval(0.3, 0.7, "a"), Interval(0.7, 1.0, "k")],
        0.0, 1.0,
    )
    tg.addTier(phones)
    path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def test_prepare_grid_writes_dest_with_creak_tier(tmp_path):
    source_dir = tmp_path / "mfa"
    dest_path = tmp_path / "annotation" / "01.TextGrid"
    _write_source_grid(source_dir / "s01_A_modal_C001.TextGrid")

    prep.prepare_grid("s01_A_modal_C001", source_dir, dest_path, SONORANTS)

    assert dest_path.exists()
    out_tg = ptg.openTextgrid(str(dest_path), includeEmptyIntervals=True)
    assert "creak" in out_tg.tierNames
    assert "phones" in out_tg.tierNames
    creak_entries = [(e.start, e.end, e.label) for e in out_tg.getTier("creak").entries]
    assert (0.3, 0.7, "") in creak_entries


def test_prepare_grid_does_not_modify_source_file(tmp_path):
    source_dir = tmp_path / "mfa"
    source_path = source_dir / "s01_A_modal_C001.TextGrid"
    dest_path = tmp_path / "annotation" / "01.TextGrid"
    _write_source_grid(source_path)
    original_bytes = source_path.read_bytes()

    prep.prepare_grid("s01_A_modal_C001", source_dir, dest_path, SONORANTS)

    assert source_path.read_bytes() == original_bytes
    source_tg = ptg.openTextgrid(str(source_path), includeEmptyIntervals=True)
    assert "creak" not in source_tg.tierNames


def test_prepare_grid_raises_when_source_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        prep.prepare_grid("nonexistent", tmp_path / "mfa", tmp_path / "out.TextGrid", SONORANTS)


# -- copy_audio -----------------------------------------------------------

def test_copy_audio_copies_bytes_to_renamed_dest(tmp_path):
    source_path = tmp_path / "split" / "s01_A_modal_C001.wav"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"RIFF-fake-audio-bytes")
    dest_path = tmp_path / "annotation" / "01.wav"

    prep.copy_audio(source_path, dest_path)

    assert dest_path.exists()
    assert dest_path.read_bytes() == b"RIFF-fake-audio-bytes"


def test_copy_audio_does_not_modify_or_move_source(tmp_path):
    source_path = tmp_path / "split" / "s01_A_modal_C001.wav"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"RIFF-fake-audio-bytes")
    dest_path = tmp_path / "annotation" / "01.wav"

    prep.copy_audio(source_path, dest_path)

    assert source_path.exists()  # still there, not moved
    assert source_path.read_bytes() == b"RIFF-fake-audio-bytes"  # unchanged


def test_copy_audio_raises_when_source_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        prep.copy_audio(tmp_path / "nonexistent.wav", tmp_path / "01.wav")


# -- write_key_csv -----------------------------------------------------------

def test_write_key_csv_columns(tmp_path):
    rows = [{"annotation_order": 1, "utt_id": "s01_A_modal_C001",
             "grid_filename": "01.TextGrid", "wav_filename": "01.wav",
             "wav_path": "data/split/s01_A_modal_C001.wav"}]
    out = tmp_path / "key.csv"
    prep.write_key_csv(rows, out)
    import csv
    with out.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["annotation_order", "utt_id", "grid_filename", "wav_filename", "wav_path"]
