import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

align_mod = import_module("04b_align")


# -- normalization -----------------------------------------------------

def test_normalize_word_lowercases_and_strips_punctuation():
    assert align_mod.normalize_word("Kahan.") == "kahan"
    assert align_mod.normalize_word("Morningside?") == "morningside"


def test_normalize_word_keeps_internal_apostrophe():
    assert align_mod.normalize_word("isn't") == "isn't"
    assert align_mod.normalize_word("What's") == "what's"


def test_normalize_sentence_splits_and_drops_empty_tokens():
    assert align_mod.normalize_sentence("Set an alarm for six forty-five.") == [
        "set", "an", "alarm", "for", "six", "forty-five",
    ]


# -- CSV loading / validation -------------------------------------------

def test_load_items_reads_id_to_text(tmp_path):
    items_csv = tmp_path / "items.csv"
    items_csv.write_text(
        "item_id,item_type,text\nC001,command,Set a timer.\nD001,declarative,It rained.\n",
        encoding="utf-8",
    )
    items = align_mod.load_items(items_csv)
    assert items == {"C001": "Set a timer.", "D001": "It rained."}


def test_load_items_rejects_duplicate_item_id(tmp_path):
    items_csv = tmp_path / "items.csv"
    items_csv.write_text(
        "item_id,item_type,text\nC001,command,A.\nC001,command,B.\n", encoding="utf-8",
    )
    with pytest.raises(ValueError):
        align_mod.load_items(items_csv)


def test_load_orders_groups_by_pass_and_sorts_by_position(tmp_path):
    orders_csv = tmp_path / "orders.csv"
    rows = ["pass,position,item_id"]
    for pos, item_id in [(2, "C002"), (1, "C001")]:
        rows.append(f"A_modal,{pos},{item_id}")
    for pos in range(1, align_mod.EXPECTED_ITEM_COUNT + 1):
        if pos > 2:
            rows.append(f"A_modal,{pos},C{pos:03d}")
    orders_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    orders = align_mod.load_orders(orders_csv)
    assert orders["A_modal"][0] == (1, "C001")
    assert orders["A_modal"][1] == (2, "C002")
    assert len(orders["A_modal"]) == align_mod.EXPECTED_ITEM_COUNT


def test_load_orders_rejects_incomplete_pass(tmp_path):
    orders_csv = tmp_path / "orders.csv"
    orders_csv.write_text("pass,position,item_id\nA_modal,1,C001\n", encoding="utf-8")
    with pytest.raises(ValueError):
        align_mod.load_orders(orders_csv)


def test_build_expected_sequence_concatenates_in_position_order():
    items = {"C001": "Set a timer.", "C002": "Call Marcus."}
    orders = {"A_modal": [(2, "C002"), (1, "C001")]}

    tokens, item_ids, word_count, position = align_mod.build_expected_sequence(
        "A_modal", items, orders
    )

    assert tokens == ["set", "a", "timer", "call", "marcus"]
    assert item_ids == ["C001", "C001", "C001", "C002", "C002"]
    assert word_count == {"C001": 3, "C002": 2}
    assert position == {"C001": 1, "C002": 2}


def test_build_expected_sequence_missing_item_fails_loudly():
    items = {"C001": "Set a timer."}
    orders = {"A_modal": [(1, "C002")]}
    with pytest.raises(ValueError):
        align_mod.build_expected_sequence("A_modal", items, orders)


# -- block filename parsing ----------------------------------------------

def test_find_block_wavs_sorted(tmp_path):
    (tmp_path / "s02_passA_modal_block.wav").write_bytes(b"")
    (tmp_path / "s01_passA_modal_block.wav").write_bytes(b"")
    found = align_mod.find_block_wavs(tmp_path)
    assert [p.name for p in found] == ["s01_passA_modal_block.wav", "s02_passA_modal_block.wav"]


def test_parse_block_stem_extracts_session_and_canonical_pass():
    session, canonical_pass = align_mod.parse_block_stem("s01_passA_modal_block")
    assert session == "s01"
    assert canonical_pass == "A_modal"


def test_parse_block_stem_rejects_missing_block_suffix():
    with pytest.raises(ValueError):
        align_mod.parse_block_stem("s01_passA_modal")


# -- alignment core --------------------------------------------------------

def test_align_words_exact_match_gives_full_coverage():
    expected_tokens = ["set", "a", "timer", "call", "marcus"]
    expected_item_ids = ["C001", "C001", "C001", "C002", "C002"]
    whisper_tokens = ["set", "a", "timer", "call", "marcus"]
    whisper_times = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.8), (1.0, 1.3), (1.3, 1.8)]
    whisper_raw = whisper_tokens[:]

    matches, insertions = align_mod.align_words(
        expected_tokens, expected_item_ids, whisper_tokens, whisper_times, whisper_raw
    )

    assert matches["C001"].matched_count == 3
    assert matches["C001"].start_sec == 0.0
    assert matches["C001"].end_sec == 0.8
    assert matches["C002"].matched_count == 2
    assert matches["C002"].start_sec == 1.0
    assert matches["C002"].end_sec == 1.8
    assert insertions == []


def test_align_words_reports_insertion_as_flub():
    expected_tokens = ["set", "a", "timer"]
    expected_item_ids = ["C001", "C001", "C001"]
    # whisper: "set a -- no wait -- a timer" (a false start inserted mid-utterance)
    whisper_tokens = ["set", "a", "no", "wait", "a", "timer"]
    whisper_times = [(0.0, 0.1), (0.1, 0.2), (0.3, 0.4), (0.4, 0.6), (0.7, 0.8), (0.8, 1.0)]
    whisper_raw = whisper_tokens[:]

    matches, insertions = align_mod.align_words(
        expected_tokens, expected_item_ids, whisper_tokens, whisper_times, whisper_raw
    )

    # difflib consumes the only expected "a" in the first matching block, so the
    # leftover whisper "a" before "timer" is attributed to the insertion too.
    assert matches["C001"].matched_count == 3
    assert len(insertions) == 1
    assert insertions[0].text == "no wait a"
    assert insertions[0].start_sec == 0.3
    assert insertions[0].end_sec == 0.8


def test_align_words_unmatched_item_has_no_times():
    expected_tokens = ["set", "a", "timer", "call", "marcus"]
    expected_item_ids = ["C001", "C001", "C001", "C002", "C002"]
    whisper_tokens = ["set", "a", "timer"]  # C002 never said / never recognized
    whisper_times = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5)]
    whisper_raw = whisper_tokens[:]

    matches, _ = align_mod.align_words(
        expected_tokens, expected_item_ids, whisper_tokens, whisper_times, whisper_raw
    )

    assert matches["C002"].matched_count == 0
    assert matches["C002"].start_sec is None
    assert matches["C002"].end_sec is None


# -- flag classification -----------------------------------------------

def test_classify_flag_unmatched_when_zero_matches():
    assert align_mod.classify_flag(0, 5, threshold=0.8) == "unmatched"


def test_classify_flag_low_confidence_below_threshold():
    assert align_mod.classify_flag(3, 5, threshold=0.8) == "low_confidence"  # 0.6 < 0.8


def test_classify_flag_ok_at_or_above_threshold():
    assert align_mod.classify_flag(4, 5, threshold=0.8) == "ok"  # 0.8 >= 0.8
    assert align_mod.classify_flag(5, 5, threshold=0.8) == "ok"


# -- row construction ----------------------------------------------------

def test_build_segmentation_rows_sorted_by_position_with_expected_fields():
    item_position = {"C002": 2, "C001": 1}
    item_word_count = {"C001": 3, "C002": 2}
    item_matches = {
        "C001": align_mod.ItemMatch(matched_count=3, start_sec=0.0, end_sec=0.8),
        "C002": align_mod.ItemMatch(matched_count=0, start_sec=None, end_sec=None),
    }

    rows = align_mod.build_segmentation_rows(
        "s01", "A_modal", item_position, item_word_count, item_matches, threshold=0.8
    )

    assert [r["item_id"] for r in rows] == ["C001", "C002"]
    assert rows[0]["match_ratio"] == 1.0
    assert rows[0]["flag"] == "ok"
    assert rows[1]["match_ratio"] == 0.0
    assert rows[1]["flag"] == "unmatched"
    assert rows[1]["start_sec"] is None
    assert all(r["session"] == "s01" and r["pass"] == "A_modal" for r in rows)
