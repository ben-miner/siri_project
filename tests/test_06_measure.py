import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

measure = import_module("06_measure")


def manifest_row(utt_id="s01_A_modal_C001", verbatim_text="Set a timer."):
    return {
        "utt_id": utt_id, "session": "s01", "pass": "A_modal", "item_id": "C001",
        "item_type": "command", "script_text": "Set a timer.",
        "verbatim_text": verbatim_text, "was_corrected": "False",
        "wav_path": f"data/split/{utt_id}.wav", "duration_sec": "2.48",
    }


# -- manifest loading -----------------------------------------------------------

def test_load_manifest_reads_rows(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "utt_id,session,pass,item_id,item_type,script_text,verbatim_text,was_corrected,wav_path,duration_sec\n"
        "s01_A_modal_C001,s01,A_modal,C001,command,Set a timer.,Set a timer.,False,data/split/s01_A_modal_C001.wav,2.48\n",
        encoding="utf-8",
    )
    rows = measure.load_manifest(p)
    assert len(rows) == 1
    assert rows[0]["utt_id"] == "s01_A_modal_C001"


# -- MFA corpus building -----------------------------------------------------------

def test_build_mfa_corpus_writes_verbatim_text_per_utt_id(tmp_path):
    rows = [manifest_row("s01_A_modal_C001", "Set a timer."), manifest_row("s01_A_modal_C002", "Call Marcus.")]
    measure.build_mfa_corpus(rows, tmp_path)

    assert (tmp_path / "s01_A_modal_C001.txt").read_text(encoding="utf-8") == "Set a timer."
    assert (tmp_path / "s01_A_modal_C002.txt").read_text(encoding="utf-8") == "Call Marcus."


def test_build_mfa_corpus_overwrites_existing_txt(tmp_path):
    (tmp_path / "s01_A_modal_C001.txt").write_text("stale", encoding="utf-8")
    measure.build_mfa_corpus([manifest_row("s01_A_modal_C001", "fresh text")], tmp_path)
    assert (tmp_path / "s01_A_modal_C001.txt").read_text(encoding="utf-8") == "fresh text"


def test_build_mfa_corpus_does_not_touch_wav_files(tmp_path):
    wav = tmp_path / "s01_A_modal_C001.wav"
    wav.write_bytes(b"RIFF-fake-audio-bytes")
    measure.build_mfa_corpus([manifest_row("s01_A_modal_C001")], tmp_path)
    assert wav.read_bytes() == b"RIFF-fake-audio-bytes"


# -- TextGrid completeness -----------------------------------------------------------

def test_find_missing_textgrids_detects_gap(tmp_path):
    (tmp_path / "s01_A_modal_C001.TextGrid").write_text("", encoding="utf-8")
    rows = [manifest_row("s01_A_modal_C001"), manifest_row("s01_A_modal_C002")]
    missing = measure.find_missing_textgrids(rows, tmp_path)
    assert missing == ["s01_A_modal_C002"]


def test_find_missing_textgrids_empty_when_all_present(tmp_path):
    (tmp_path / "s01_A_modal_C001.TextGrid").write_text("", encoding="utf-8")
    missing = measure.find_missing_textgrids([manifest_row("s01_A_modal_C001")], tmp_path)
    assert missing == []


# -- acoustics summary loading -----------------------------------------------------------

def test_load_acoustics_summary_derives_utt_id_from_file_with_extension(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "file,duration_s,f0_mean_hz,f0_quality_flag\n"
        "s01_A_modal_C001.wav,2.48,180.5,\n",
        encoding="utf-8",
    )
    by_utt_id = measure.load_acoustics_summary(p)
    assert set(by_utt_id.keys()) == {"s01_A_modal_C001"}
    assert by_utt_id["s01_A_modal_C001"]["f0_mean_hz"] == "180.5"


def test_load_acoustics_summary_rejects_duplicate_utt_id(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "file,duration_s\ns01_A_modal_C001.wav,2.48\ns01_A_modal_C001.wav,2.48\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        measure.load_acoustics_summary(p)


# -- heavy intervention flag classification -----------------------------------------------------------

def test_has_heavy_intervention_true_for_heavy_repair():
    assert measure.has_heavy_intervention("heavy_repair") is True


def test_has_heavy_intervention_true_for_no_pitch():
    assert measure.has_heavy_intervention("no_pitch") is True


def test_has_heavy_intervention_true_when_combined_with_other_flags():
    assert measure.has_heavy_intervention("octave_uncertain;heavy_repair") is True


def test_has_heavy_intervention_false_for_empty_flag():
    assert measure.has_heavy_intervention("") is False


def test_has_heavy_intervention_false_for_period_doubling_alone():
    # This is the creak signal the project studies, not a measurement defect --
    # must NOT be treated as heavy intervention (see module docstring).
    assert measure.has_heavy_intervention("period_doubling_present") is False


def test_has_heavy_intervention_false_for_mild_flags():
    assert measure.has_heavy_intervention("jumpy;low_voicing;octave_uncertain") is False


# -- join problem detection -----------------------------------------------------------

def test_find_join_problems_clean_case_has_none():
    manifest_rows = [manifest_row("s01_A_modal_C001")]
    acoustics = {"s01_A_modal_C001": {"file": "s01_A_modal_C001.wav", "f0_quality_flag": ""}}
    assert measure.find_join_problems(manifest_rows, acoustics) == []


def test_find_join_problems_detects_missing_from_acoustics():
    manifest_rows = [manifest_row("s01_A_modal_C001")]
    problems = measure.find_join_problems(manifest_rows, {})
    assert len(problems) == 1
    assert problems[0]["category"] == "missing_from_acoustics"
    assert problems[0]["utt_id"] == "s01_A_modal_C001"


def test_find_join_problems_detects_missing_from_manifest():
    acoustics = {"s99_ghost": {"file": "s99_ghost.wav", "f0_quality_flag": ""}}
    problems = measure.find_join_problems([], acoustics)
    assert len(problems) == 1
    assert problems[0]["category"] == "missing_from_manifest"


def test_find_join_problems_detects_heavy_intervention():
    manifest_rows = [manifest_row("s01_A_modal_C001")]
    acoustics = {
        "s01_A_modal_C001": {"file": "s01_A_modal_C001.wav", "f0_quality_flag": "heavy_repair"}
    }
    problems = measure.find_join_problems(manifest_rows, acoustics)
    assert len(problems) == 1
    assert problems[0]["category"] == "heavy_intervention"
    assert problems[0]["detail"] == "heavy_repair"


def test_find_join_problems_collects_all_categories_at_once():
    manifest_rows = [manifest_row("s01_ok"), manifest_row("s01_bad_flag"), manifest_row("s01_missing")]
    acoustics = {
        "s01_ok": {"file": "s01_ok.wav", "f0_quality_flag": ""},
        "s01_bad_flag": {"file": "s01_bad_flag.wav", "f0_quality_flag": "no_pitch"},
        "s01_extra": {"file": "s01_extra.wav", "f0_quality_flag": ""},
    }
    problems = measure.find_join_problems(manifest_rows, acoustics)
    categories = {p["category"] for p in problems}
    assert categories == {"missing_from_acoustics", "missing_from_manifest", "heavy_intervention"}
    assert len(problems) == 3


def test_find_join_problems_keep_flagged_suppresses_heavy_intervention_only():
    manifest_rows = [manifest_row("s01_bad_flag"), manifest_row("s01_missing")]
    acoustics = {
        "s01_bad_flag": {"file": "s01_bad_flag.wav", "f0_quality_flag": "no_pitch"},
    }
    problems = measure.find_join_problems(manifest_rows, acoustics, keep_flagged=True)
    categories = {p["category"] for p in problems}
    assert categories == {"missing_from_acoustics"}  # heavy_intervention suppressed, missing kept


# -- joined row construction -----------------------------------------------------------

def test_build_joined_rows_merges_manifest_and_acoustics():
    manifest_rows = [manifest_row("s01_A_modal_C001")]
    acoustics = {
        "s01_A_modal_C001": {
            "file": "s01_A_modal_C001.wav", "duration_s": "2.48",
            "f0_mean_hz": "180.5", "f0_quality_flag": "",
        }
    }
    rows = measure.build_joined_rows(manifest_rows, acoustics, exclude_utt_ids=set())
    assert len(rows) == 1
    assert rows[0]["utt_id"] == "s01_A_modal_C001"
    assert rows[0]["f0_mean_hz"] == "180.5"
    assert rows[0]["script_text"] == "Set a timer."


def test_build_joined_rows_excludes_flagged_utt_ids():
    manifest_rows = [manifest_row("s01_good"), manifest_row("s01_bad")]
    acoustics = {
        "s01_good": {"file": "s01_good.wav", "f0_quality_flag": ""},
        "s01_bad": {"file": "s01_bad.wav", "f0_quality_flag": "heavy_repair"},
    }
    rows = measure.build_joined_rows(manifest_rows, acoustics, exclude_utt_ids={"s01_bad"})
    assert [r["utt_id"] for r in rows] == ["s01_good"]


def test_build_joined_rows_skips_utt_id_with_no_acoustics_entry():
    manifest_rows = [manifest_row("s01_A_modal_C001")]
    rows = measure.build_joined_rows(manifest_rows, {}, exclude_utt_ids=set())
    assert rows == []


def test_build_joined_rows_adds_f0_tracking_failed_column():
    manifest_rows = [manifest_row("s01_clean"), manifest_row("s01_flagged")]
    acoustics = {
        "s01_clean": {"file": "s01_clean.wav", "f0_quality_flag": ""},
        "s01_flagged": {"file": "s01_flagged.wav", "f0_quality_flag": "heavy_repair"},
    }
    rows = measure.build_joined_rows(manifest_rows, acoustics, exclude_utt_ids=set())
    by_id = {r["utt_id"]: r for r in rows}
    assert by_id["s01_clean"]["f0_tracking_failed"] is False
    assert by_id["s01_flagged"]["f0_tracking_failed"] is True


def test_build_joined_rows_keeps_flagged_rows_when_not_excluded():
    """The --keep-flagged workflow: caller passes an empty exclude set even
    though the row is flagged, and the row is kept with the flag visible."""
    manifest_rows = [manifest_row("s01_flagged")]
    acoustics = {"s01_flagged": {"file": "s01_flagged.wav", "f0_quality_flag": "no_pitch"}}
    rows = measure.build_joined_rows(manifest_rows, acoustics, exclude_utt_ids=set())
    assert len(rows) == 1
    assert rows[0]["f0_tracking_failed"] is True
    assert rows[0]["f0_quality_flag"] == "no_pitch"


# -- CSV writing -----------------------------------------------------------

def test_write_joined_csv_orders_manifest_columns_first_then_extras(tmp_path):
    rows = [{
        "utt_id": "s01_A_modal_C001", "session": "s01", "pass": "A_modal", "item_id": "C001",
        "item_type": "command", "script_text": "Set a timer.", "verbatim_text": "Set a timer.",
        "was_corrected": False, "wav_path": "data/split/s01_A_modal_C001.wav", "duration_sec": "2.48",
        "f0_mean_hz": "180.5", "f0_quality_flag": "",
    }]
    out = tmp_path / "joined.csv"
    measure.write_joined_csv(rows, out)

    import csv as csv_mod
    with out.open(encoding="utf-8") as f:
        reader = csv_mod.reader(f)
        header = next(reader)
    assert header[:10] == measure.MANIFEST_COLUMNS
    assert "f0_mean_hz" in header
    assert "f0_quality_flag" in header


def test_write_joined_csv_raises_on_empty_rows(tmp_path):
    with pytest.raises(ValueError):
        measure.write_joined_csv([], tmp_path / "joined.csv")
