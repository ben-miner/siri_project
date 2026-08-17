import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

cut_mod = import_module("04c_cut")


def make_row(session="s01", pass_name="A_modal", item_id="C001", position=1,
             start=1.0, end=2.0):
    return {
        "session": session, "pass": pass_name, "item_id": item_id,
        "position": position, "start_sec": start, "end_sec": end,
    }


# -- CSV loading -----------------------------------------------------------

def test_load_segmentation_parses_floats_and_blanks(tmp_path):
    csv_path = tmp_path / "seg.csv"
    csv_path.write_text(
        "session,pass,item_id,position,start_sec,end_sec,n_whisper_words_matched,match_ratio,flag\n"
        "s01,A_modal,C001,1,1.0,2.0,3,1.0,ok\n"
        "s01,A_modal,C002,2,,,0,0.0,unmatched\n",
        encoding="utf-8",
    )
    rows = cut_mod.load_segmentation(csv_path)
    assert rows[0]["start_sec"] == 1.0 and rows[0]["end_sec"] == 2.0
    assert rows[1]["start_sec"] is None and rows[1]["end_sec"] is None


def test_load_expected_items_groups_by_pass(tmp_path):
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "pass,position,item_id\nA_modal,1,C001\nA_modal,2,C002\nB_natural,1,D001\n",
        encoding="utf-8",
    )
    expected = cut_mod.load_expected_items(csv_path)
    assert expected["A_modal"] == {"C001", "C002"}
    assert expected["B_natural"] == {"D001"}


# -- item coverage -----------------------------------------------------------

def test_check_item_coverage_no_issues_when_exact_match():
    rows = [make_row(item_id="C001"), make_row(item_id="C002")]
    issues = cut_mod.check_item_coverage(rows, {"C001", "C002"})
    assert issues == []


def test_check_item_coverage_flags_missing():
    rows = [make_row(item_id="C001")]
    issues = cut_mod.check_item_coverage(rows, {"C001", "C002"})
    assert len(issues) == 1
    assert issues[0]["item_ids"] == ["C002"]
    assert "missing" in issues[0]["message"]


def test_check_item_coverage_flags_duplicate():
    rows = [make_row(item_id="C001"), make_row(item_id="C001")]
    issues = cut_mod.check_item_coverage(rows, {"C001"})
    assert len(issues) == 1
    assert issues[0]["item_ids"] == ["C001"]
    assert "duplicate" in issues[0]["message"]


def test_check_item_coverage_flags_unexpected():
    rows = [make_row(item_id="C999")]
    issues = cut_mod.check_item_coverage(rows, set())
    assert len(issues) == 1
    assert "unexpected" in issues[0]["message"]


# -- durations -----------------------------------------------------------

def test_check_durations_flags_missing_boundary():
    rows = [make_row(start=None, end=None)]
    missing, non_positive, out_of_range = cut_mod.check_durations(rows)
    assert len(missing) == 1
    assert non_positive == [] and out_of_range == []


def test_check_durations_flags_non_positive():
    rows = [make_row(start=5.0, end=4.0)]
    missing, non_positive, out_of_range = cut_mod.check_durations(rows)
    assert len(non_positive) == 1


def test_check_durations_flags_too_short_and_too_long():
    rows = [make_row(start=0.0, end=0.1), make_row(item_id="C002", start=0.0, end=20.0)]
    missing, non_positive, out_of_range = cut_mod.check_durations(rows)
    assert len(out_of_range) == 2


def test_check_durations_ok_within_range():
    rows = [make_row(start=0.0, end=1.0)]
    missing, non_positive, out_of_range = cut_mod.check_durations(rows)
    assert missing == [] and non_positive == [] and out_of_range == []


# -- overlaps -----------------------------------------------------------

def test_check_overlaps_none_when_sequential():
    rows = [make_row(item_id="C001", start=0.0, end=1.0), make_row(item_id="C002", start=1.0, end=2.0)]
    assert cut_mod.check_overlaps(rows) == []


def test_check_overlaps_detects_overlap():
    rows = [make_row(item_id="C001", start=0.0, end=1.5), make_row(item_id="C002", start=1.0, end=2.0)]
    issues = cut_mod.check_overlaps(rows)
    assert len(issues) == 1
    assert issues[0]["item_ids"] == ["C001", "C002"]


# -- full validate() -----------------------------------------------------------

def test_validate_passes_clean_data():
    rows = [
        make_row(item_id="C001", start=0.0, end=1.0),
        make_row(item_id="C002", start=1.0, end=2.0),
    ]
    expected = {"A_modal": {"C001", "C002"}}
    report, ok, bad_keys = cut_mod.validate(rows, expected)
    assert ok is True
    assert bad_keys == set()
    assert any("PASS" in line for line in report)


def test_validate_fails_and_collects_bad_keys_for_overlap():
    rows = [
        make_row(item_id="C001", start=0.0, end=1.5),
        make_row(item_id="C002", start=1.0, end=2.0),
    ]
    expected = {"A_modal": {"C001", "C002"}}
    report, ok, bad_keys = cut_mod.validate(rows, expected)
    assert ok is False
    assert ("s01", "A_modal", "C001") in bad_keys
    assert ("s01", "A_modal", "C002") in bad_keys


def test_validate_unknown_pass_marks_all_rows_bad():
    rows = [make_row(pass_name="Z_unknown", item_id="C001", start=0.0, end=1.0)]
    report, ok, bad_keys = cut_mod.validate(rows, {"A_modal": {"C001"}})
    assert ok is False
    assert ("s01", "Z_unknown", "C001") in bad_keys


# -- padding / clamping -----------------------------------------------------------

def test_compute_padded_bounds_pads_when_room_available():
    rows = [make_row(item_id="C001", start=5.0, end=6.0)]
    bounds = cut_mod.compute_padded_bounds(rows, pad=0.15, file_duration=100.0)
    assert bounds == [(pytest.approx(4.85), pytest.approx(6.15))]


def test_compute_padded_bounds_clamps_to_file_edges():
    rows = [make_row(item_id="C001", start=0.05, end=99.95)]
    bounds = cut_mod.compute_padded_bounds(rows, pad=0.15, file_duration=100.0)
    assert bounds == [(0.0, 100.0)]


def test_compute_padded_bounds_clamps_to_neighbours_no_overlap():
    rows = [
        make_row(item_id="C001", start=0.0, end=1.0),
        make_row(item_id="C002", start=1.1, end=2.0),  # only 100ms gap, less than 2*pad
    ]
    bounds = cut_mod.compute_padded_bounds(rows, pad=0.15, file_duration=100.0)
    (s1, e1), (s2, e2) = bounds
    assert e1 <= s2  # padded windows must not overlap
    # gap is [1.0, 1.1], midpoint 1.05; both sides clamp to that midpoint
    # since full padding (0.15) would cross it
    assert e1 == pytest.approx(1.05)
    assert s2 == pytest.approx(1.05)


def test_compute_padded_bounds_three_items_middle_clamped_both_sides():
    rows = [
        make_row(item_id="C001", start=0.0, end=1.0),
        make_row(item_id="C002", start=1.0, end=2.0),
        make_row(item_id="C003", start=2.0, end=3.0),
    ]
    bounds = cut_mod.compute_padded_bounds(rows, pad=0.15, file_duration=100.0)
    _, (s2, e2), _ = bounds
    assert s2 == pytest.approx(1.0)
    assert e2 == pytest.approx(2.0)


# -- block filename -----------------------------------------------------------

def test_block_filename_matches_04a_convention():
    assert cut_mod.block_filename("s01", "A_modal") == "s01_passA_modal_block.wav"


# -- integration: cut_take with synthetic audio -----------------------------------------------------------

@pytest.fixture
def synthetic_block(tmp_path):
    import parselmouth

    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()
    sound = parselmouth.Sound(
        values=[[0.0] * (48000 * 10)],
        sampling_frequency=48000,
    )
    path = blocks_dir / cut_mod.block_filename("s01", "A_modal")
    sound.save(str(path), parselmouth.SoundFileFormat.WAV)
    return blocks_dir


def test_cut_take_writes_expected_files(tmp_path, synthetic_block):
    split_dir = tmp_path / "split"
    take_rows = [
        make_row(item_id="C001", start=1.0, end=2.0),
        make_row(item_id="C002", start=2.5, end=3.5),
    ]
    n_cut, n_skipped = cut_mod.cut_take("s01", "A_modal", take_rows, synthetic_block, split_dir, bad_keys=set())

    assert n_cut == 2
    assert n_skipped == 0
    assert (split_dir / "s01_A_modal_C001.wav").exists()
    assert (split_dir / "s01_A_modal_C002.wav").exists()

    import parselmouth
    clip = parselmouth.Sound(str(split_dir / "s01_A_modal_C001.wav"))
    # 1.0-2.0 with 150ms pad and no crowding neighbour closer than that -> 0.7 to 2.15...
    # but next item starts at 2.5 so end pad isn't clamped by neighbour, only duration+pad
    assert clip.duration == pytest.approx(1.0 + 2 * 0.15, abs=0.01)
    assert clip.sampling_frequency == 48000


def test_cut_take_skips_bad_rows(tmp_path, synthetic_block):
    split_dir = tmp_path / "split"
    take_rows = [
        make_row(item_id="C001", start=1.0, end=2.0),
        make_row(item_id="C002", start=2.5, end=3.5),
    ]
    bad_keys = {("s01", "A_modal", "C002")}
    n_cut, n_skipped = cut_mod.cut_take("s01", "A_modal", take_rows, synthetic_block, split_dir, bad_keys)

    assert n_cut == 1
    assert n_skipped == 1
    assert (split_dir / "s01_A_modal_C001.wav").exists()
    assert not (split_dir / "s01_A_modal_C002.wav").exists()


def test_cut_take_missing_block_file_fails_loudly(tmp_path):
    empty_dir = tmp_path / "no_blocks"
    empty_dir.mkdir()
    take_rows = [make_row(item_id="C001", start=1.0, end=2.0)]
    with pytest.raises(FileNotFoundError):
        cut_mod.cut_take("s01", "A_modal", take_rows, empty_dir, tmp_path / "split", bad_keys=set())
