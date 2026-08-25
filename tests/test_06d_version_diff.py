import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

vdiff = import_module("06d_version_diff")


# -- loading -----------------------------------------------------------

def test_load_rows_keys_by_utt_id(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("utt_id,f0_mean_hz\ns01_A_modal_C001,130.5\n", encoding="utf-8")
    rows = vdiff.load_rows(p)
    assert set(rows.keys()) == {"s01_A_modal_C001"}


def test_load_rows_rejects_duplicate_utt_id(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("utt_id,f0_mean_hz\ns01_x,1\ns01_x,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        vdiff.load_rows(p)


# -- measure_columns -----------------------------------------------------------

def test_measure_columns_excludes_manifest_and_file():
    v1_rows = {"s01_x": {
        "utt_id": "s01_x", "session": "s01", "pass": "A_modal", "item_id": "C001",
        "item_type": "command", "script_text": "x", "verbatim_text": "x",
        "was_corrected": "False", "wav_path": "x.wav", "duration_sec": "1.0",
        "file": "s01_x.wav", "f0_mean_hz": "130.0", "h1_h2_db_mean": "-2.0",
    }}
    cols = vdiff.measure_columns(v1_rows)
    assert cols == ["f0_mean_hz", "h1_h2_db_mean"]


def test_measure_columns_empty_when_no_rows():
    assert vdiff.measure_columns({}) == []


# -- compare_value -----------------------------------------------------------

def test_compare_value_numeric_unchanged():
    r = vdiff.compare_value("130.0", "130.0")
    assert r["kind"] == "numeric"
    assert r["changed"] is False
    assert r["sign_changed"] is False


def test_compare_value_numeric_changed():
    r = vdiff.compare_value("605.6", "44.7")
    assert r["kind"] == "numeric"
    assert r["changed"] is True
    assert r["delta"] == pytest.approx(44.7 - 605.6)
    assert r["sign_changed"] is False


def test_compare_value_sign_change_detected():
    r = vdiff.compare_value("16.2", "-5.3")
    assert r["sign_changed"] is True
    assert r["changed"] is True


def test_compare_value_zero_is_not_a_sign_change():
    assert vdiff.compare_value("0.0", "-5.0")["sign_changed"] is False
    assert vdiff.compare_value("5.0", "0.0")["sign_changed"] is False


def test_compare_value_pct_change_from_nonzero():
    r = vdiff.compare_value("100.0", "110.0")
    assert r["pct_change"] == pytest.approx(10.0)


def test_compare_value_pct_change_from_zero_nonzero_delta_is_inf():
    r = vdiff.compare_value("0.0", "5.0")
    assert r["pct_change"] == math.inf


def test_compare_value_pct_change_from_zero_zero_delta_is_zero():
    r = vdiff.compare_value("0.0", "0.0")
    assert r["pct_change"] == 0.0


def test_compare_value_text_changed():
    r = vdiff.compare_value("heavy_repair", "")
    assert r["kind"] == "text"
    assert r["changed"] is True
    assert r["sign_changed"] is False


def test_compare_value_text_unchanged():
    r = vdiff.compare_value("period_doubling_present", "period_doubling_present")
    assert r["changed"] is False


def test_compare_value_nan_treated_as_missing_falls_back_to_text():
    # empty string parses as neither -> text comparison
    r = vdiff.compare_value("", "44.7")
    assert r["kind"] == "text"
    assert r["changed"] is True


def test_compare_value_boolean_case_difference_is_not_a_change():
    # Regression: v1 was archived after an Excel round-trip that uppercases
    # TRUE/FALSE; v2 is written directly by Python as str(bool) ("True").
    # Comparing case-sensitively would report every boolean column as 100%
    # changed -- verified on real data (creak_sonorant_restricted).
    assert vdiff.compare_value("TRUE", "True")["changed"] is False
    assert vdiff.compare_value("FALSE", "False")["changed"] is False
    assert vdiff.compare_value("true", "True")["changed"] is False


def test_compare_value_boolean_genuine_change_still_detected():
    assert vdiff.compare_value("TRUE", "False")["changed"] is True
    assert vdiff.compare_value("False", "True")["changed"] is True


# -- diff_utt_id / build_diff -----------------------------------------------------------

def test_diff_utt_id_produces_one_entry_per_column():
    v1 = {"f0_mean_hz": "605.6", "h1_h2_db_mean": "16.2"}
    v2 = {"f0_mean_hz": "44.7", "h1_h2_db_mean": "-5.3"}
    diffs = vdiff.diff_utt_id("s02_C_creak_D026", v1, v2, ["f0_mean_hz", "h1_h2_db_mean"])
    assert len(diffs) == 2
    assert {d["measure"] for d in diffs} == {"f0_mean_hz", "h1_h2_db_mean"}
    assert all(d["utt_id"] == "s02_C_creak_D026" for d in diffs)


def test_build_diff_separates_only_v1_only_v2_and_common():
    v1_rows = {"a": {"f0_mean_hz": "100"}, "only_in_v1": {"f0_mean_hz": "1"}}
    v2_rows = {"a": {"f0_mean_hz": "105"}, "only_in_v2": {"f0_mean_hz": "2"}}
    result = vdiff.build_diff(v1_rows, v2_rows, ["f0_mean_hz"])
    assert result["only_v1"] == ["only_in_v1"]
    assert result["only_v2"] == ["only_in_v2"]
    assert result["common"] == ["a"]
    assert len(result["diffs"]) == 1
    assert result["diffs"][0]["utt_id"] == "a"


def test_build_diff_reports_all_comparisons_not_just_changed():
    v1_rows = {"a": {"x": "1", "y": "2"}}
    v2_rows = {"a": {"x": "1", "y": "3"}}
    result = vdiff.build_diff(v1_rows, v2_rows, ["x", "y"])
    assert len(result["diffs"]) == 2  # both x (unchanged) and y (changed) included
    changed_flags = {d["measure"]: d["changed"] for d in result["diffs"]}
    assert changed_flags == {"x": False, "y": True}


# -- CSV writing -----------------------------------------------------------

def test_write_diff_csv_round_trips(tmp_path):
    diffs = [{
        "utt_id": "s01_x", "measure": "h1_h2_db_mean", "kind": "numeric",
        "v1": 16.2, "v2": -5.3, "delta": -21.5, "pct_change": -132.7,
        "sign_changed": True, "changed": True,
    }]
    out = tmp_path / "diff.csv"
    vdiff.write_diff_csv(diffs, out)

    import csv
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["utt_id"] == "s01_x"
    assert rows[0]["sign_changed"] == "True"
