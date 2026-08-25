import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

manifest = import_module("05_manifest")


# -- filename parsing -----------------------------------------------------------

def test_parse_filename_handles_underscore_in_pass_name():
    session, pass_name, item_id = manifest.parse_filename("s01_A_modal_C001")
    assert session == "s01"
    assert pass_name == "A_modal"
    assert item_id == "C001"


def test_parse_filename_handles_all_three_pass_names():
    assert manifest.parse_filename("s02_B_natural_D017")[1] == "B_natural"
    assert manifest.parse_filename("s02_C_creak_F009")[1] == "C_creak"


def test_parse_filename_rejects_missing_item_id():
    with pytest.raises(ValueError):
        manifest.parse_filename("s01_A_modal_notanitem")


def test_parse_filename_rejects_no_underscores():
    with pytest.raises(ValueError):
        manifest.parse_filename("garbage")


def test_parse_filename_item_id_prefix_must_be_cdf():
    with pytest.raises(ValueError):
        manifest.parse_filename("s01_A_modal_X001")


# -- find_split_wavs -----------------------------------------------------------

def test_find_split_wavs_sorted(tmp_path):
    (tmp_path / "s02_A_modal_C001.wav").write_bytes(b"")
    (tmp_path / "s01_A_modal_C001.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = manifest.find_split_wavs(tmp_path)
    assert [p.name for p in found] == ["s01_A_modal_C001.wav", "s02_A_modal_C001.wav"]


# -- CSV loading -----------------------------------------------------------

def test_load_items_maps_id_to_type_and_text(tmp_path):
    p = tmp_path / "items.csv"
    p.write_text("item_id,item_type,text\nC001,command,Set a timer.\n", encoding="utf-8")
    items = manifest.load_items(p)
    assert items["C001"] == {"item_type": "command", "text": "Set a timer."}


def test_load_references_parses_was_corrected_bool(tmp_path):
    p = tmp_path / "references.csv"
    p.write_text(
        "utt_id,script_text,verbatim_text,was_corrected\n"
        "s01_A_modal_C001,Set a timer.,Set a timer.,False\n"
        "s01_A_modal_C002,Call Marcus.,Call the marker.,True\n",
        encoding="utf-8",
    )
    refs = manifest.load_references(p)
    assert refs["s01_A_modal_C001"]["was_corrected"] is False
    assert refs["s01_A_modal_C002"]["was_corrected"] is True
    assert refs["s01_A_modal_C002"]["verbatim_text"] == "Call the marker."


# -- build_manifest_rows -----------------------------------------------------------

def make_inputs():
    items = {"C001": {"item_type": "command", "text": "Set a timer."}}
    references = {
        "s01_A_modal_C001": {
            "script_text": "Set a timer.", "verbatim_text": "Set a timer.", "was_corrected": False,
        }
    }
    return items, references


def test_build_manifest_rows_happy_path(tmp_path):
    path = tmp_path / "s01_A_modal_C001.wav"
    path.write_bytes(b"")
    items, references = make_inputs()
    durations = {path: 2.5}

    rows, errors = manifest.build_manifest_rows([path], items, references, durations)

    assert errors == []
    assert len(rows) == 1
    row = rows[0]
    assert row["utt_id"] == "s01_A_modal_C001"
    assert row["session"] == "s01"
    assert row["pass"] == "A_modal"
    assert row["item_id"] == "C001"
    assert row["item_type"] == "command"
    assert row["script_text"] == "Set a timer."
    assert row["verbatim_text"] == "Set a timer."
    assert row["was_corrected"] is False
    assert row["wav_path"] == path.as_posix()
    assert row["duration_sec"] == 2.5


def test_build_manifest_rows_reports_unparseable_filename(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"")
    items, references = make_inputs()

    rows, errors = manifest.build_manifest_rows([path], items, references, {path: 1.0})

    assert rows == []
    assert len(errors) == 1
    assert "garbage.wav" in errors[0]


def test_build_manifest_rows_reports_missing_item_id(tmp_path):
    path = tmp_path / "s01_A_modal_C999.wav"
    path.write_bytes(b"")
    items, references = make_inputs()  # only has C001

    rows, errors = manifest.build_manifest_rows([path], items, references, {path: 1.0})

    assert rows == []
    assert "C999" in errors[0]
    assert "items.csv" in errors[0]


def test_build_manifest_rows_reports_missing_reference(tmp_path):
    path = tmp_path / "s02_A_modal_C001.wav"  # different session, not in references
    path.write_bytes(b"")
    items, references = make_inputs()

    rows, errors = manifest.build_manifest_rows([path], items, references, {path: 1.0})

    assert rows == []
    assert "references.csv" in errors[0]


def test_build_manifest_rows_reports_script_text_mismatch(tmp_path):
    path = tmp_path / "s01_A_modal_C001.wav"
    path.write_bytes(b"")
    items = {"C001": {"item_type": "command", "text": "Set a timer."}}
    references = {
        "s01_A_modal_C001": {
            "script_text": "SET A DIFFERENT TIMER.", "verbatim_text": "x", "was_corrected": False,
        }
    }

    rows, errors = manifest.build_manifest_rows([path], items, references, {path: 1.0})

    assert rows == []
    assert "mismatch" in errors[0].lower() or "disagrees" in errors[0].lower()


def test_build_manifest_rows_collects_multiple_errors_without_stopping(tmp_path):
    bad1 = tmp_path / "garbage.wav"
    bad1.write_bytes(b"")
    bad2 = tmp_path / "s01_A_modal_C999.wav"
    bad2.write_bytes(b"")
    items, references = make_inputs()

    rows, errors = manifest.build_manifest_rows(
        [bad1, bad2], items, references, {bad1: 1.0, bad2: 1.0}
    )

    assert rows == []
    assert len(errors) == 2


# -- print_grid -----------------------------------------------------------

def test_print_grid_counts_by_session_and_pass(capsys):
    rows = [
        {"session": "s01", "pass": "A_modal"},
        {"session": "s01", "pass": "A_modal"},
        {"session": "s02", "pass": "C_creak"},
    ]
    manifest.print_grid(rows)
    out = capsys.readouterr().out
    assert "2 session x pass cell(s)" in out
    assert "s01 A_modal: 2" in out
    assert "s02 C_creak: 1" in out


# -- get_wav_duration (real audio, via parselmouth to generate a fixture) --------

def test_get_wav_duration_matches_known_length(tmp_path):
    import parselmouth

    sound = parselmouth.Sound(values=[[0.0] * (48000 * 3)], sampling_frequency=48000)
    path = tmp_path / "test.wav"
    sound.save(str(path), parselmouth.SoundFileFormat.WAV)

    duration = manifest.get_wav_duration(path)
    assert duration == pytest.approx(3.0, abs=0.01)
