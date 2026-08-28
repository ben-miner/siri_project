import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest
from praatio import textgrid as ptg
from praatio.utilities.constants import Interval

compute = import_module("08c_compute_proportions")
grids = import_module("08b_prepare_grids")

SONORANTS = {"m", "n", "l", "j", "w", "i", "a", "u"}


# -- load_annotation_key -----------------------------------------------------------

def test_load_annotation_key_reads_rows(tmp_path):
    p = tmp_path / "key.csv"
    p.write_text(
        "annotation_order,utt_id,grid_filename,wav_filename,wav_path\n"
        "1,s01_A_modal_C001,01.TextGrid,01.wav,data/split/s01_A_modal_C001.wav\n",
        encoding="utf-8",
    )
    rows = compute.load_annotation_key(p)
    assert len(rows) == 1
    assert rows[0]["utt_id"] == "s01_A_modal_C001"


# -- load_pass_by_utt_id -----------------------------------------------------------

def test_load_pass_by_utt_id_reads_manifest(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "utt_id,session,pass,item_id\ns01_A_modal_C001,s01,A_modal,C001\n",
        encoding="utf-8",
    )
    result = compute.load_pass_by_utt_id(p)
    assert result == {"s01_A_modal_C001": "A_modal"}


def test_load_pass_by_utt_id_rejects_duplicate_utt_id(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "utt_id,session,pass,item_id\ns01_x,s01,A_modal,C001\ns01_x,s01,A_modal,C002\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        compute.load_pass_by_utt_id(p)


# -- load_phonpipe_creak_rate -----------------------------------------------------------

def test_load_phonpipe_creak_rate_parses_float(tmp_path):
    p = tmp_path / "acoustics_joined.csv"
    p.write_text(
        "utt_id,creak_doubling_rate\ns01_x,0.42\n",
        encoding="utf-8",
    )
    result = compute.load_phonpipe_creak_rate(p)
    assert result == {"s01_x": pytest.approx(0.42)}


def test_load_phonpipe_creak_rate_excludes_nan(tmp_path):
    p = tmp_path / "acoustics_joined.csv"
    p.write_text(
        "utt_id,creak_doubling_rate\ns01_x,0.42\ns01_y,nan\n",
        encoding="utf-8",
    )
    result = compute.load_phonpipe_creak_rate(p)
    assert "s01_y" not in result


# -- _boundaries_match -----------------------------------------------------------

def test_boundaries_match_identical():
    a = [Interval(0.0, 0.5, ""), Interval(0.5, 1.0, "")]
    b = [Interval(0.0, 0.5, "c"), Interval(0.5, 1.0, "")]  # label differs, boundaries don't
    assert compute._boundaries_match(a, b) is True


def test_boundaries_match_within_tolerance():
    a = [Interval(0.0, 0.5, "")]
    b = [Interval(0.00005, 0.50005, "")]  # well within 1e-4
    assert compute._boundaries_match(a, b) is True


def test_boundaries_match_false_beyond_tolerance():
    a = [Interval(0.0, 0.5, "")]
    b = [Interval(0.01, 0.5, "")]  # moved by 10ms, way beyond tolerance
    assert compute._boundaries_match(a, b) is False


def test_boundaries_match_false_on_count_mismatch():
    a = [Interval(0.0, 0.5, ""), Interval(0.5, 1.0, "")]
    b = [Interval(0.0, 1.0, "")]
    assert compute._boundaries_match(a, b) is False


# -- repair_extra_gap_boundaries -----------------------------------------------------------

def test_repair_merges_spurious_blank_split():
    expected = [Interval(0.0, 0.5, ""), Interval(0.5, 1.0, "")]
    # actual: the (0.0, 0.5) gap got accidentally split into two blank pieces
    actual = [Interval(0.0, 0.2, ""), Interval(0.2, 0.5, ""), Interval(0.5, 1.0, "")]
    repaired, merges = compute.repair_extra_gap_boundaries(actual, expected)
    assert [(e.start, e.end, e.label) for e in repaired] == [(0.0, 0.5, ""), (0.5, 1.0, "")]
    assert merges == [(0.2, "")]


def test_repair_does_nothing_when_already_matching():
    expected = [Interval(0.0, 0.5, ""), Interval(0.5, 1.0, "c")]
    repaired, merges = compute.repair_extra_gap_boundaries(expected, expected)
    assert [(e.start, e.end, e.label) for e in repaired] == [(0.0, 0.5, ""), (0.5, 1.0, "c")]
    assert merges == []


def test_repair_raises_when_spurious_boundary_separates_different_labels():
    expected = [Interval(0.0, 0.5, ""), Interval(0.5, 1.0, "")]
    # a spurious split inside the gap, but one half got labeled "c" -- unsafe to merge
    actual = [Interval(0.0, 0.2, ""), Interval(0.2, 0.5, "c"), Interval(0.5, 1.0, "")]
    with pytest.raises(ValueError):
        compute.repair_extra_gap_boundaries(actual, expected)


def test_repair_merges_multiple_consecutive_spurious_splits():
    expected = [Interval(0.0, 1.0, "")]
    actual = [Interval(0.0, 0.3, ""), Interval(0.3, 0.6, ""), Interval(0.6, 1.0, "")]
    repaired, merges = compute.repair_extra_gap_boundaries(actual, expected)
    assert [(e.start, e.end, e.label) for e in repaired] == [(0.0, 1.0, "")]
    assert len(merges) == 2


# -- process_grid (integration, real TextGrid I/O) -----------------------------------------------------------

def _write_grid(path: Path, phone_entries, creak_labels=None) -> None:
    """phone_entries: list of (start, end, label). creak_labels: optional
    list of labels to apply to the freshly-built creak tier's entries, in
    order (must match its entry count) -- default: leave all "" (untouched
    annotation)."""
    tg = ptg.Textgrid(minTimestamp=0.0, maxTimestamp=phone_entries[-1][1])
    phones = ptg.IntervalTier(
        "phones", [Interval(s, e, l) for s, e, l in phone_entries],
        0.0, phone_entries[-1][1],
    )
    tg.addTier(phones)
    creak_tier = grids.build_creak_tier(phones, SONORANTS)
    if creak_labels is not None:
        assert len(creak_labels) == len(creak_tier.entries)
        creak_tier = ptg.IntervalTier(
            "creak",
            [Interval(e.start, e.end, lab) for e, lab in zip(creak_tier.entries, creak_labels)],
            creak_tier.minTimestamp, creak_tier.maxTimestamp,
        )
    tg.addTier(creak_tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


# phone layout: [t](obstruent) [a](sonorant 0.3-0.6) [k](obstruent) [m](sonorant 0.9-1.2)
STANDARD_PHONES = [(0.0, 0.3, "t"), (0.3, 0.6, "a"), (0.6, 0.9, "k"), (0.9, 1.2, "m")]


def test_process_grid_valid_no_creak(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES)  # creak tier left all blank
    result = compute.process_grid(p, SONORANTS)
    assert "error" not in result
    assert result["n_tokens"] == 2
    assert result["n_creaky"] == 0
    assert result["hand_creak_proportion"] == pytest.approx(0.0)


def test_process_grid_valid_one_creaky_token(tmp_path):
    p = tmp_path / "01.TextGrid"
    # creak tier entries (from build_creak_tier over STANDARD_PHONES):
    # (0.0,0.3)="", (0.3,0.6)=token "a", (0.6,0.9)="", (0.9,1.2)=token "m"
    _write_grid(p, STANDARD_PHONES, creak_labels=["", "c", "", ""])
    result = compute.process_grid(p, SONORANTS)
    assert "error" not in result
    assert result["n_tokens"] == 2
    assert result["n_creaky"] == 1
    # token durations: "a" span = 0.3, "m" span = 0.3 -> proportion = 0.3/0.6 = 0.5
    assert result["hand_creak_proportion"] == pytest.approx(0.5)


def test_process_grid_exposes_raw_token_list(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES, creak_labels=["", "c", "", ""])
    result = compute.process_grid(p, SONORANTS)
    assert result["tokens"] == [
        {"start": 0.3, "end": 0.6, "label": "c"},
        {"start": 0.9, "end": 1.2, "label": ""},
    ]


def test_process_grid_relative_position_of_creaky_token(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES, creak_labels=["", "", "", "c"])  # "m" token creaky
    result = compute.process_grid(p, SONORANTS)
    # "m" spans 0.9-1.2, midpoint 1.05, utterance spans 0.0-1.2 -> relative = 1.05/1.2
    assert result["creaky_relative_positions"] == [pytest.approx(1.05 / 1.2)]


def test_process_grid_missing_file_is_an_error(tmp_path):
    result = compute.process_grid(tmp_path / "nonexistent.TextGrid", SONORANTS)
    assert "error" in result


def test_process_grid_missing_creak_tier_is_an_error(tmp_path):
    p = tmp_path / "01.TextGrid"
    tg = ptg.Textgrid(minTimestamp=0.0, maxTimestamp=1.0)
    tg.addTier(ptg.IntervalTier("phones", [Interval(0.0, 1.0, "a")], 0.0, 1.0))
    tg.save(str(p), format="long_textgrid", includeBlankSpaces=True)
    result = compute.process_grid(p, SONORANTS)
    assert "error" in result
    assert "creak" in result["error"]


def test_process_grid_detects_moved_boundary(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES)
    # Now hand-corrupt: reopen and shift a creak-tier boundary by 10ms.
    tg = ptg.openTextgrid(str(p), includeEmptyIntervals=True)
    creak = tg.getTier("creak")
    corrupted_entries = list(creak.entries)
    e0 = corrupted_entries[0]
    corrupted_entries[0] = Interval(e0.start, e0.end + 0.01, e0.label)
    corrupted_entries[1] = Interval(corrupted_entries[1].start + 0.01, corrupted_entries[1].end,
                                    corrupted_entries[1].label)
    corrupted = ptg.IntervalTier("creak", corrupted_entries, creak.minTimestamp, creak.maxTimestamp)
    tg.replaceTier("creak", corrupted)
    tg.save(str(p), format="long_textgrid", includeBlankSpaces=True)

    result = compute.process_grid(p, SONORANTS)
    assert "error" in result
    assert "boundaries" in result["error"]


def _split_gap_interval(tg_path: Path, gap_start: float, split_at: float, gap_end: float,
                        second_half_label: str = "") -> None:
    """Simulate the real-world bug found auditing the 60 annotated grids:
    an extra boundary accidentally inserted inside a blank gap interval,
    splitting (gap_start, gap_end) into (gap_start, split_at) and
    (split_at, gap_end)."""
    tg = ptg.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    creak = tg.getTier("creak")
    new_entries = []
    for e in creak.entries:
        if abs(e.start - gap_start) < 1e-9 and abs(e.end - gap_end) < 1e-9:
            new_entries.append(Interval(gap_start, split_at, e.label))
            new_entries.append(Interval(split_at, gap_end, second_half_label))
        else:
            new_entries.append(e)
    corrupted = ptg.IntervalTier("creak", new_entries, creak.minTimestamp, creak.maxTimestamp)
    tg.replaceTier("creak", corrupted)
    tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)


def test_process_grid_repairs_spurious_blank_split_in_a_gap(tmp_path):
    """Matches the real pattern found in 9 of the 60 hand-annotated grids:
    a stray extra boundary inside a blank gap region, both halves still
    blank -- must be auto-repaired (in memory) and processed normally,
    with the repair reported."""
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES, creak_labels=["", "c", "", ""])  # "a" token creaky
    _split_gap_interval(p, 0.6, 0.75, 0.9, second_half_label="")  # split the (0.6,0.9) gap

    result = compute.process_grid(p, SONORANTS)
    assert "error" not in result
    assert result["n_tokens"] == 2
    assert result["n_creaky"] == 1
    assert result["hand_creak_proportion"] == pytest.approx(0.5)
    assert result["repairs"] == [(0.75, "")]


def test_process_grid_refuses_to_repair_when_split_half_is_labeled_c(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES)  # nothing creaky
    _split_gap_interval(p, 0.6, 0.75, 0.9, second_half_label="c")  # bogus "c" inside a gap

    result = compute.process_grid(p, SONORANTS)
    assert "error" in result


def test_process_grid_detects_bad_label(tmp_path):
    p = tmp_path / "01.TextGrid"
    _write_grid(p, STANDARD_PHONES, creak_labels=["", "creak", "", ""])  # "creak" not "c"
    result = compute.process_grid(p, SONORANTS)
    assert "error" in result
    assert "label" in result["error"]


def test_process_grid_zero_tokens_is_an_error(tmp_path):
    p = tmp_path / "01.TextGrid"
    no_sonorant_phones = [(0.0, 0.5, "t"), (0.5, 1.0, "k")]  # no sonorants at all
    _write_grid(p, no_sonorant_phones)
    result = compute.process_grid(p, SONORANTS)
    assert "error" in result
    assert "zero tokens" in result["error"]


# -- pearson_correlation -----------------------------------------------------------

def test_pearson_correlation_perfect_positive():
    assert compute.pearson_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    assert compute.pearson_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_pearson_correlation_insufficient_data_is_nan():
    r = compute.pearson_correlation([1.0], [1.0])
    assert r != r  # NaN


# -- write_hand_annotation_csv -----------------------------------------------------------

def test_write_hand_annotation_csv_columns(tmp_path):
    rows = [{"utt_id": "s01_x", "hand_creak_proportion": 0.5, "n_tokens": 4,
             "n_creaky": 2, "pass": "C_creak"}]
    out = tmp_path / "hand_annotation.csv"
    compute.write_hand_annotation_csv(rows, out)
    import csv
    with out.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["utt_id", "hand_creak_proportion", "n_tokens", "n_creaky", "pass"]


# -- _summarize -----------------------------------------------------------

def test_summarize_basic_stats():
    s = compute._summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(3.0)
    assert s["median"] == pytest.approx(3.0)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(5.0)
