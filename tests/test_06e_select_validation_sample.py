import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

select = import_module("06e_select_validation_sample")


def row(utt_id, pass_name, shr=None, f0=None, failed=False):
    return {
        "utt_id": utt_id, "pass": pass_name, "shr_median": shr,
        "f0_mean_hz": f0, "f0_tracking_failed": failed,
        "wav_path": f"data/split/{utt_id}.wav",
    }


# -- has_heavy_intervention -----------------------------------------------------------

def test_has_heavy_intervention_true_for_heavy_repair():
    assert select.has_heavy_intervention("heavy_repair") is True


def test_has_heavy_intervention_false_for_period_doubling_alone():
    assert select.has_heavy_intervention("period_doubling_present") is False


def test_has_heavy_intervention_false_for_empty():
    assert select.has_heavy_intervention("") is False


# -- build_full_dataset -----------------------------------------------------------

def test_build_full_dataset_joins_and_derives_failed_flag():
    manifest_rows = [{"utt_id": "s01_A_modal_C001", "pass": "A_modal", "wav_path": "x.wav"}]
    acoustics_by_utt_id = {
        "s01_A_modal_C001": {
            "file": "s01_A_modal_C001.wav", "shr_median": "0.2", "f0_mean_hz": "130.5",
            "f0_quality_flag": "heavy_repair",
        }
    }
    rows = select.build_full_dataset(manifest_rows, acoustics_by_utt_id)
    assert rows[0]["shr_median"] == 0.2
    assert rows[0]["f0_mean_hz"] == 130.5
    assert rows[0]["f0_tracking_failed"] is True


def test_build_full_dataset_fails_loudly_on_missing_utt_id():
    manifest_rows = [{"utt_id": "s01_A_modal_C001", "pass": "A_modal", "wav_path": "x.wav"}]
    with pytest.raises(ValueError):
        select.build_full_dataset(manifest_rows, {})


# -- select_loud_c_creak_failures -----------------------------------------------------------

def test_select_loud_includes_forced_and_fills_rest_by_shr():
    rows = [
        row("forced1", "C_creak", shr=0.30, failed=True),
        row("forced2", "C_creak", shr=0.31, failed=True),
        row("high1", "C_creak", shr=0.90, failed=True),
        row("high2", "C_creak", shr=0.80, failed=True),
        row("low1", "C_creak", shr=0.50, failed=True),
        row("not_failed", "C_creak", shr=0.99, failed=False),  # excluded: gate not tripped
        row("other_pass", "B_natural", shr=0.99, failed=True),  # excluded: wrong pass
    ]
    selected = select.select_loud_c_creak_failures(rows, 4, ("forced1", "forced2"))
    ids = {r["utt_id"] for r in selected}
    assert ids == {"forced1", "forced2", "high1", "high2"}
    assert len(selected) == 4


def test_select_loud_raises_if_forced_utt_id_not_a_valid_candidate():
    rows = [row("a", "C_creak", shr=0.5, failed=True)]
    with pytest.raises(ValueError):
        select.select_loud_c_creak_failures(rows, 1, ("nonexistent",))


def test_select_loud_raises_if_not_enough_candidates():
    rows = [row("a", "C_creak", shr=0.5, failed=True)]
    with pytest.raises(ValueError):
        select.select_loud_c_creak_failures(rows, 4, ())


# -- select_silent_c_creak_failures -----------------------------------------------------------

def test_select_silent_filters_by_threshold_and_excludes_failed():
    rows = [
        row("s1", "C_creak", shr=0.45, failed=False),
        row("s2", "C_creak", shr=0.41, failed=False),
        row("below_threshold", "C_creak", shr=0.39, failed=False),
        row("tripped", "C_creak", shr=0.99, failed=True),
    ]
    selected = select.select_silent_c_creak_failures(rows, 2, 0.40, exclude_utt_ids=set())
    assert {r["utt_id"] for r in selected} == {"s1", "s2"}


def test_select_silent_excludes_already_selected_utt_ids():
    rows = [
        row("s1", "C_creak", shr=0.45, failed=False),
        row("s2", "C_creak", shr=0.44, failed=False),
        row("s3", "C_creak", shr=0.43, failed=False),
    ]
    selected = select.select_silent_c_creak_failures(rows, 2, 0.40, exclude_utt_ids={"s1"})
    assert {r["utt_id"] for r in selected} == {"s2", "s3"}


# -- select_highest_shr -----------------------------------------------------------

def test_select_highest_shr_picks_top_n_for_pass():
    rows = [
        row("b1", "B_natural", shr=0.9),
        row("b2", "B_natural", shr=0.7),
        row("b3", "B_natural", shr=0.5),
        row("c1", "C_creak", shr=0.99),  # wrong pass, excluded
    ]
    selected = select.select_highest_shr(rows, "B_natural", 2)
    assert [r["utt_id"] for r in selected] == ["b1", "b2"]


# -- select_median_f0_controls -----------------------------------------------------------

def test_select_median_f0_controls_picks_closest_to_median():
    rows = [
        row("low", "A_modal", f0=100.0),
        row("mid1", "A_modal", f0=130.0),
        row("mid2", "A_modal", f0=132.0),
        row("high", "A_modal", f0=200.0),
    ]
    # median of [100,130,132,200] = (130+132)/2 = 131
    selected = select.select_median_f0_controls(rows, "A_modal", 2)
    assert {r["utt_id"] for r in selected} == {"mid1", "mid2"}


# -- build_validation_sample (integration of all strata) -----------------------------------------------------------

def make_full_population():
    rows = []
    rows.append(row("s02_C_creak_F011", "C_creak", shr=0.50, failed=True))
    rows.append(row("s02_C_creak_F023", "C_creak", shr=0.51, failed=True))
    for i in range(10):
        rows.append(row(f"loud{i}", "C_creak", shr=0.9 - i * 0.01, failed=True))
    for i in range(10):
        rows.append(row(f"silent{i}", "C_creak", shr=0.45 - i * 0.001, failed=False))
    for i in range(10):
        rows.append(row(f"bnat{i}", "B_natural", shr=0.8 - i * 0.01))
    for i in range(10):
        rows.append(row(f"amod{i}", "A_modal", f0=100.0 + i))
    return rows


def test_build_validation_sample_returns_exactly_ten_unique_rows():
    rows = make_full_population()
    selected = select.build_validation_sample(rows)
    assert len(selected) == 10
    assert len({r["utt_id"] for r in selected}) == 10


def test_build_validation_sample_force_includes_specified_utt_ids():
    rows = make_full_population()
    selected = select.build_validation_sample(rows)
    ids = {r["utt_id"] for r in selected}
    assert "s02_C_creak_F011" in ids
    assert "s02_C_creak_F023" in ids


def test_build_validation_sample_strata_sizes():
    rows = make_full_population()
    selected = select.build_validation_sample(rows)
    c_creak_failed = [r for r in selected if r["pass"] == "C_creak" and r["f0_tracking_failed"]]
    c_creak_silent = [r for r in selected if r["pass"] == "C_creak" and not r["f0_tracking_failed"]]
    b_natural = [r for r in selected if r["pass"] == "B_natural"]
    a_modal = [r for r in selected if r["pass"] == "A_modal"]
    assert len(c_creak_failed) == 4
    assert len(c_creak_silent) == 2
    assert len(b_natural) == 2
    assert len(a_modal) == 2


# -- CSV writing -----------------------------------------------------------

def test_write_validation_sample_csv_has_expected_columns(tmp_path):
    rows = [row("s01_A_modal_C001", "A_modal", shr=0.2, f0=130.0, failed=False)]
    out = tmp_path / "validation_sample.csv"
    select.write_validation_sample_csv(rows, out)

    import csv as csv_mod
    with out.open(encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        written = list(reader)
    assert set(reader.fieldnames) == {
        "utt_id", "pass", "shr_median", "f0_mean_hz", "f0_tracking_failed", "wav_path",
        "window_start_sec", "window_end_sec", "n_pulses", "f0_hand_hz",
        "amplitude_alternation", "notes",
    }
    assert written[0]["utt_id"] == "s01_A_modal_C001"
    assert written[0]["n_pulses"] == ""
    assert written[0]["notes"] == ""
