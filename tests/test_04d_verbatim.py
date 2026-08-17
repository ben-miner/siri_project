import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

verbatim = import_module("04d_verbatim")


def seg_row(session="s01", pass_name="A_modal", item_id="C001", position=1,
            match_ratio=1.0, flag="ok"):
    return {
        "session": session, "pass": pass_name, "item_id": item_id,
        "position": position, "match_ratio": match_ratio, "flag": flag,
    }


# -- utt_id / needs_review -----------------------------------------------------------

def test_make_utt_id_joins_with_underscores():
    assert verbatim.make_utt_id("s01", "A_modal", "C001") == "s01_A_modal_C001"


def test_needs_review_true_when_match_ratio_below_one():
    assert verbatim.needs_review(seg_row(match_ratio=0.99, flag="ok")) is True


def test_needs_review_true_when_flag_not_ok():
    assert verbatim.needs_review(seg_row(match_ratio=1.0, flag="ok_manual")) is True


def test_needs_review_false_when_perfect_and_ok():
    assert verbatim.needs_review(seg_row(match_ratio=1.0, flag="ok")) is False


# -- CSV loading -----------------------------------------------------------

def test_load_segmentation_parses_types(tmp_path):
    p = tmp_path / "seg.csv"
    p.write_text(
        "session,pass,item_id,position,start_sec,end_sec,n_whisper_words_matched,match_ratio,flag\n"
        "s01,A_modal,C001,1,0.0,1.0,3,0.75,low_confidence\n",
        encoding="utf-8",
    )
    rows = verbatim.load_segmentation(p)
    assert rows[0]["match_ratio"] == 0.75
    assert rows[0]["position"] == 1
    assert rows[0]["flag"] == "low_confidence"


def test_load_items_maps_id_to_text(tmp_path):
    p = tmp_path / "items.csv"
    p.write_text("item_id,item_type,text\nC001,command,Set a timer.\n", encoding="utf-8")
    assert verbatim.load_items(p) == {"C001": "Set a timer."}


def test_load_review_sheet_round_trip(tmp_path):
    p = tmp_path / "review.csv"
    p.write_text(
        "utt_id,script_text,whisper_text,match_ratio,verbatim_text\n"
        "s01_A_modal_C001,Set a timer.,Set the timer.,0.5,\n",
        encoding="utf-8",
    )
    rows = verbatim.load_review_sheet(p)
    assert rows[0]["match_ratio"] == 0.5
    assert rows[0]["verbatim_text"] == ""


# -- build_review_rows -----------------------------------------------------------

def test_build_review_rows_filters_and_sorts_worst_first():
    rows = [
        seg_row(item_id="C001", match_ratio=0.9, flag="low_confidence", position=1),
        seg_row(item_id="C002", match_ratio=0.5, flag="low_confidence", position=2),
        seg_row(item_id="C003", match_ratio=1.0, flag="ok", position=3),  # excluded
    ]
    items = {"C001": "one", "C002": "two", "C003": "three"}
    whisper_texts = {"s01_A_modal_C001": "wa one", "s01_A_modal_C002": "wa two"}

    result = verbatim.build_review_rows(rows, items, whisper_texts)

    assert [r["utt_id"] for r in result] == ["s01_A_modal_C002", "s01_A_modal_C001"]
    assert result[0]["match_ratio"] == 0.5
    assert result[0]["verbatim_text"] == ""
    assert result[0]["whisper_text"] == "wa two"
    assert result[0]["script_text"] == "two"


def test_build_review_rows_carries_over_existing_verbatim():
    rows = [seg_row(item_id="C001", match_ratio=0.5, flag="low_confidence")]
    items = {"C001": "one"}
    existing = {"s01_A_modal_C001": "actually said this"}

    result = verbatim.build_review_rows(rows, items, {}, existing_verbatim=existing)

    assert result[0]["verbatim_text"] == "actually said this"


def test_build_review_rows_never_autopopulates_from_whisper_text():
    rows = [seg_row(item_id="C001", match_ratio=0.5, flag="low_confidence")]
    items = {"C001": "one"}
    whisper_texts = {"s01_A_modal_C001": "whisper guess"}

    result = verbatim.build_review_rows(rows, items, whisper_texts)

    assert result[0]["verbatim_text"] == ""  # never copied from whisper_text
    assert result[0]["whisper_text"] == "whisper guess"


def test_build_review_rows_missing_item_fails_loudly():
    rows = [seg_row(item_id="C999", match_ratio=0.5, flag="low_confidence")]
    with pytest.raises(ValueError):
        verbatim.build_review_rows(rows, {}, {})


# -- print_breakdown (just check it doesn't crash and counts correctly) --------------

def test_print_breakdown_counts_by_pass(capsys):
    rows = [
        seg_row(pass_name="A_modal", item_id="C001"),
        seg_row(pass_name="A_modal", item_id="C002"),
        seg_row(pass_name="C_creak", item_id="C003"),
    ]
    verbatim.print_breakdown(rows)
    out = capsys.readouterr().out
    assert "3 item(s) need review" in out
    assert "A_modal: 2" in out
    assert "C_creak: 1" in out


# -- finalize gate (hard rule 2) -----------------------------------------------------------

def test_finalize_gate_flags_blank_verbatim_below_threshold():
    rows = [
        {"utt_id": "u1", "script_text": "a", "whisper_text": "b", "match_ratio": 0.5, "verbatim_text": ""},
        {"utt_id": "u2", "script_text": "a", "whisper_text": "b", "match_ratio": 0.9, "verbatim_text": ""},
        {"utt_id": "u3", "script_text": "a", "whisper_text": "b", "match_ratio": 0.5, "verbatim_text": "filled"},
    ]
    failures = verbatim.check_finalize_gate(rows)
    assert [r["utt_id"] for r in failures] == ["u1"]


def test_finalize_gate_passes_when_all_reviewed_or_high_ratio():
    rows = [
        {"utt_id": "u1", "script_text": "a", "whisper_text": "b", "match_ratio": 0.5, "verbatim_text": "filled"},
        {"utt_id": "u2", "script_text": "a", "whisper_text": "b", "match_ratio": 0.9, "verbatim_text": ""},
    ]
    assert verbatim.check_finalize_gate(rows) == []


# -- copy-paste warning (hard rule 1) -----------------------------------------------------------

def test_copy_paste_warning_detects_high_identical_fraction():
    rows = [
        {"utt_id": f"u{i}", "script_text": "a", "whisper_text": "same text", "match_ratio": 0.5,
         "verbatim_text": "same text" if i < 3 else "different"}
        for i in range(4)
    ]
    n_corrected, n_identical, fraction = verbatim.check_copy_paste_warning(rows)
    assert n_corrected == 4
    assert n_identical == 3
    assert fraction == 0.75


def test_copy_paste_warning_ignores_blank_rows():
    rows = [
        {"utt_id": "u1", "script_text": "a", "whisper_text": "w", "match_ratio": 0.9, "verbatim_text": ""},
        {"utt_id": "u2", "script_text": "a", "whisper_text": "w", "match_ratio": 0.9, "verbatim_text": "w"},
    ]
    n_corrected, n_identical, fraction = verbatim.check_copy_paste_warning(rows)
    assert n_corrected == 1
    assert n_identical == 1
    assert fraction == 1.0


def test_copy_paste_warning_zero_corrected_gives_zero_fraction():
    rows = [{"utt_id": "u1", "script_text": "a", "whisper_text": "w", "match_ratio": 0.9, "verbatim_text": ""}]
    n_corrected, n_identical, fraction = verbatim.check_copy_paste_warning(rows)
    assert n_corrected == 0
    assert fraction == 0.0


# -- build_references -----------------------------------------------------------

def test_build_references_uses_script_text_for_unreviewed_rows():
    seg_rows = [seg_row(item_id="C001", match_ratio=1.0, flag="ok")]
    items = {"C001": "Set a timer."}
    refs = verbatim.build_references(seg_rows, items, review_by_utt_id={})

    assert refs[0]["utt_id"] == "s01_A_modal_C001"
    assert refs[0]["verbatim_text"] == "Set a timer."
    assert refs[0]["was_corrected"] is False


def test_build_references_uses_filled_verbatim_when_present():
    seg_rows = [seg_row(item_id="C001", match_ratio=0.5, flag="low_confidence")]
    items = {"C001": "Set a timer."}
    review = {"s01_A_modal_C001": {"verbatim_text": "Set the timer."}}
    refs = verbatim.build_references(seg_rows, items, review_by_utt_id=review)

    assert refs[0]["verbatim_text"] == "Set the timer."
    assert refs[0]["was_corrected"] is True


def test_build_references_blank_reviewed_verbatim_falls_back_to_script():
    seg_rows = [seg_row(item_id="C001", match_ratio=0.9, flag="low_confidence")]
    items = {"C001": "Set a timer."}
    review = {"s01_A_modal_C001": {"verbatim_text": ""}}
    refs = verbatim.build_references(seg_rows, items, review_by_utt_id=review)

    assert refs[0]["verbatim_text"] == "Set a timer."
    assert refs[0]["was_corrected"] is False


def test_build_references_missing_item_fails_loudly():
    seg_rows = [seg_row(item_id="C999")]
    with pytest.raises(ValueError):
        verbatim.build_references(seg_rows, {}, review_by_utt_id={})


# -- batch packing (speed fix: avoid one fixed-cost 30s encoder pass per short clip) --

def test_pack_into_batches_single_batch_when_under_cap():
    clips = [("u1", 3.0), ("u2", 3.0), ("u3", 3.0)]
    batches = verbatim.pack_into_batches(clips, max_batch_duration=25.0, gap=0.5)
    assert batches == [clips]


def test_pack_into_batches_splits_when_exceeding_cap():
    # 3 clips of 10s + 2 gaps of 0.5s = 31s > 25s cap -> must split
    clips = [("u1", 10.0), ("u2", 10.0), ("u3", 10.0)]
    batches = verbatim.pack_into_batches(clips, max_batch_duration=25.0, gap=0.5)
    assert len(batches) == 2
    assert sum(d for _, d in batches[0]) + 0.5 * (len(batches[0]) - 1) <= 25.0


def test_pack_into_batches_oversized_single_clip_gets_own_batch():
    clips = [("u1", 40.0), ("u2", 3.0)]
    batches = verbatim.pack_into_batches(clips, max_batch_duration=25.0, gap=0.5)
    assert batches == [[("u1", 40.0)], [("u2", 3.0)]]


def test_pack_into_batches_preserves_all_clips():
    clips = [(f"u{i}", 2.0) for i in range(20)]
    batches = verbatim.pack_into_batches(clips, max_batch_duration=25.0, gap=0.5)
    flattened = [item for batch in batches for item in batch]
    assert flattened == clips


def test_pack_into_batches_respects_cap_for_every_batch():
    clips = [(f"u{i}", 4.0) for i in range(10)]
    batches = verbatim.pack_into_batches(clips, max_batch_duration=25.0, gap=0.5)
    for batch in batches:
        total = sum(d for _, d in batch) + 0.5 * (len(batch) - 1)
        assert total <= 25.0


# -- batch offset computation -----------------------------------------------------------

def test_compute_batch_offsets_stacks_with_gaps():
    batch = [("u1", 2.0), ("u2", 3.0), ("u3", 1.5)]
    offsets = verbatim.compute_batch_offsets(batch, gap=0.5)
    assert offsets == [
        ("u1", 0.0, 2.0),
        ("u2", 2.5, 5.5),
        ("u3", 6.0, 7.5),
    ]


def test_compute_batch_offsets_single_clip_no_gap():
    offsets = verbatim.compute_batch_offsets([("u1", 4.0)], gap=0.5)
    assert offsets == [("u1", 0.0, 4.0)]


# -- word-to-clip attribution -----------------------------------------------------------

def test_assign_words_to_clips_basic_partition():
    offsets = [("u1", 0.0, 2.0), ("u2", 2.5, 5.5)]
    words = [(0.1, 0.5, "hello"), (1.0, 1.5, "world"), (3.0, 3.5, "foo"), (4.0, 4.5, "bar")]
    result = verbatim.assign_words_to_clips(words, offsets)
    assert result["u1"] == ["hello", "world"]
    assert result["u2"] == ["foo", "bar"]


def test_assign_words_to_clips_word_in_gap_goes_to_nearest():
    offsets = [("u1", 0.0, 2.0), ("u2", 2.5, 5.5)]
    # word spans 2.1-2.3, midpoint 2.2, inside the [2.0, 2.5) silence gap
    words = [(2.1, 2.3, "stray")]
    result = verbatim.assign_words_to_clips(words, offsets)
    # 2.2 is closer to u1's end (2.0, distance 0.2) than u2's start (2.5, distance 0.3)
    assert result["u1"] == ["stray"]
    assert result["u2"] == []


def test_assign_words_to_clips_empty_clip_has_empty_list():
    offsets = [("u1", 0.0, 2.0), ("u2", 2.5, 5.5)]
    words = [(0.1, 0.5, "hello")]
    result = verbatim.assign_words_to_clips(words, offsets)
    assert result["u1"] == ["hello"]
    assert result["u2"] == []
