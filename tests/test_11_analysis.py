import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pandas as pd
import pytest

analysis = import_module("11_analysis")

REAL_WAV = Path(r"c:\Users\benmi\siri_project\data\split\s01_A_modal_C001.wav")


# -- load_joined_data -----------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_joined_data_renames_and_joins(tmp_path):
    acoustics_path = tmp_path / "acoustics_joined.csv"
    scored_path = tmp_path / "scored.csv"
    whisper_path = tmp_path / "wer_whisper.csv"

    _write_csv(acoustics_path, [
        {"utt_id": "u1", "session": "s01", "pass": "A_modal", "item_id": "C001",
         "item_type": "command", "shr_median": 0.2, "jitter_local_pct": 1.0,
         "f0_tracking_failed": False, "duration_sec": 2.0, "wav_path": "x.wav",
         "was_corrected": False},  # also duplicated in scored.csv -- must not collide
    ])
    _write_csv(scored_path, [
        {"utt_id": "u1", "session": "s01", "pass": "A_modal", "verbatim_text": "hi",
         "script_text": "hi", "was_corrected": False, "hypothesis": "hi",
         "verbatim_substitutions": 0, "verbatim_deletions": 0, "verbatim_insertions": 0,
         "verbatim_reference_length": 1, "verbatim_wer": 0.0,
         "script_substitutions": 0, "script_deletions": 0, "script_insertions": 0,
         "script_reference_length": 1, "script_wer": 0.0},
    ])
    _write_csv(whisper_path, [
        {"utt_id": "u1", "session": "s01", "pass": "A_modal", "verbatim_text": "hi",
         "hypothesis": "hi", "model_size": "large-v3", "compute_type": "int8",
         "elapsed_ms": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0,
         "reference_length": 1, "wer": 0.0, "alignment": "hi"},
    ])

    df = analysis.load_joined_data(acoustics_path, scored_path, whisper_path)
    assert len(df) == 1
    # no pandas merge-suffixed columns at all -- catches ANY unhandled
    # collision between the three source files, not just the ones this
    # test happens to name explicitly (was_corrected was one such miss).
    suffixed = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    assert suffixed == [], f"unhandled column collision(s): {suffixed}"
    assert "was_corrected" in df.columns
    # whisper columns disambiguated from scored.csv's own columns
    assert "whisper_wer" in df.columns
    assert "apple_hypothesis" in df.columns
    assert "whisper_hypothesis" in df.columns
    assert df.loc[0, "whisper_wer"] == 0.0


def test_load_joined_data_raises_on_mismatched_utt_ids(tmp_path):
    acoustics_path = tmp_path / "acoustics_joined.csv"
    scored_path = tmp_path / "scored.csv"
    whisper_path = tmp_path / "wer_whisper.csv"

    _write_csv(acoustics_path, [
        {"utt_id": "u1", "session": "s01", "pass": "A_modal", "item_id": "C001",
         "item_type": "command", "shr_median": 0.2, "jitter_local_pct": 1.0,
         "f0_tracking_failed": False, "duration_sec": 2.0, "wav_path": "x.wav"},
    ])
    _write_csv(scored_path, [
        {"utt_id": "u_DIFFERENT", "session": "s01", "pass": "A_modal", "verbatim_text": "hi",
         "script_text": "hi", "was_corrected": False, "hypothesis": "hi",
         "verbatim_substitutions": 0, "verbatim_deletions": 0, "verbatim_insertions": 0,
         "verbatim_reference_length": 1, "verbatim_wer": 0.0,
         "script_substitutions": 0, "script_deletions": 0, "script_insertions": 0,
         "script_reference_length": 1, "script_wer": 0.0},
    ])
    _write_csv(whisper_path, [
        {"utt_id": "u1", "session": "s01", "pass": "A_modal", "verbatim_text": "hi",
         "hypothesis": "hi", "model_size": "large-v3", "compute_type": "int8",
         "elapsed_ms": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0,
         "reference_length": 1, "wer": 0.0, "alignment": "hi"},
    ])

    with pytest.raises(Exception):
        analysis.load_joined_data(acoustics_path, scored_path, whisper_path)


# -- add_wer_columns / add_zscored_predictors -----------------------------------------------------------

def test_add_wer_columns_recomputes_from_raw_counts():
    df = pd.DataFrame({
        "verbatim_substitutions": [1], "verbatim_deletions": [1], "verbatim_insertions": [0],
        "verbatim_reference_length": [4],
        "whisper_substitutions": [0], "whisper_deletions": [0], "whisper_insertions": [2],
        "whisper_reference_length": [4],
    })
    result = analysis.add_wer_columns(df)
    assert result.loc[0, "apple_wer"] == pytest.approx(0.5)
    assert result.loc[0, "whisper_wer_full"] == pytest.approx(0.5)


def test_add_zscored_predictors_mean_zero_std_one():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = analysis.add_zscored_predictors(df, ["x"])
    assert result["x_z"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["x_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


# -- check_roomtone_by_session -----------------------------------------------------------

def test_check_roomtone_by_session_computes_pct_diff_and_flags(tmp_path):
    p = tmp_path / "calibration.csv"
    rows = (
        [{"session": "s01", "label": "roomtone", "rms_level": v} for v in [0.0010, 0.0010, 0.0010]]
        + [{"session": "s02", "label": "roomtone", "rms_level": v} for v in [0.0014, 0.0014, 0.0014]]
        + [{"session": "s01", "label": "creak_cal", "rms_level": 0.01}]  # non-roomtone row, must be ignored
    )
    pd.DataFrame(rows).to_csv(p, index=False)
    result = analysis.check_roomtone_by_session(p)
    assert result["mean_1"] == pytest.approx(0.0010)
    assert result["mean_2"] == pytest.approx(0.0014)
    # symmetric % diff: |0.0010-0.0014| / mean(0.0010,0.0014) * 100
    assert result["pct_diff"] == pytest.approx(33.333, abs=0.01)
    assert result["flagged"] == True  # noqa: E712 -- numpy.bool_, `is True` would fail
    assert result["n_1"] == 3
    assert result["n_2"] == 3


def test_check_roomtone_by_session_not_flagged_when_close(tmp_path):
    p = tmp_path / "calibration.csv"
    rows = (
        [{"session": "s01", "label": "roomtone", "rms_level": v} for v in [0.0010, 0.0010, 0.0010]]
        + [{"session": "s02", "label": "roomtone", "rms_level": v} for v in [0.00102, 0.00102, 0.00102]]
    )
    pd.DataFrame(rows).to_csv(p, index=False)
    result = analysis.check_roomtone_by_session(p)
    assert result["flagged"] == False  # noqa: E712 -- numpy.bool_, `is False` would fail


# -- error_type_breakdown -----------------------------------------------------------

def test_error_type_breakdown_hand_computed():
    df = pd.DataFrame({
        "pass": ["A_modal", "A_modal"],
        "verbatim_substitutions": [1, 0], "verbatim_deletions": [0, 2], "verbatim_insertions": [0, 0],
        "verbatim_reference_length": [4, 4],
        "whisper_substitutions": [0, 0], "whisper_deletions": [0, 0], "whisper_insertions": [1, 0],
        "whisper_reference_length": [4, 4],
    })
    result = analysis.error_type_breakdown(df)
    apple_row = result[result["recognizer"] == "Apple"].iloc[0]
    # total substitutions=1, deletions=2, insertions=0, over ref_total=8
    assert apple_row["substitution_rate"] == pytest.approx(1 / 8)
    assert apple_row["deletion_rate"] == pytest.approx(2 / 8)
    assert apple_row["insertion_rate"] == pytest.approx(0.0)

    whisper_row = result[result["recognizer"] == "Whisper"].iloc[0]
    assert whisper_row["insertion_rate"] == pytest.approx(1 / 8)


# -- verbatim_vs_script_gap -----------------------------------------------------------

def test_verbatim_vs_script_gap():
    df = pd.DataFrame({
        "verbatim_substitutions": [1], "verbatim_deletions": [0], "verbatim_insertions": [0],
        "verbatim_reference_length": [4],
        "script_substitutions": [2], "script_deletions": [0], "script_insertions": [0],
        "script_reference_length": [4],
    })
    result = analysis.verbatim_vs_script_gap(df)
    assert result["verbatim_wer"] == pytest.approx(0.25)
    assert result["script_wer"] == pytest.approx(0.5)
    assert result["gap"] == pytest.approx(0.25)


# -- correction_rate_by_pass -----------------------------------------------------------

def test_correction_rate_by_pass():
    df = pd.DataFrame({
        "pass": ["A_modal", "A_modal", "B_natural"],
        "was_corrected": [True, False, True],
    })
    result = analysis.correction_rate_by_pass(df)
    assert result["A_modal"] == pytest.approx(0.5)
    assert result["B_natural"] == pytest.approx(1.0)


# -- sentence_type_effect_within_b_natural -----------------------------------------------------------

def test_sentence_type_effect_within_b_natural_filters_and_summarizes():
    df = pd.DataFrame({
        "pass": ["B_natural", "B_natural", "B_natural", "A_modal"],
        "item_type": ["command", "command", "declarative", "command"],
        "apple_wer": [0.1, 0.3, 0.5, 0.9],  # last row must be excluded (not B_natural)
    })
    summary, f_stat, p_value = analysis.sentence_type_effect_within_b_natural(df)
    assert summary.loc["command", "n"] == 2
    assert summary.loc["command", "mean"] == pytest.approx(0.2)
    assert summary.loc["declarative", "n"] == 1
    assert isinstance(f_stat, float)
    assert isinstance(p_value, float)


# -- compute_intensity_db (integration, real audio) -----------------------------------------------------------

pytestmark_real_wav = pytest.mark.skipif(
    not REAL_WAV.exists(), reason=f"{REAL_WAV} not found (project audio, not part of this repo)")


@pytestmark_real_wav
def test_compute_intensity_db_returns_plausible_value():
    value = analysis.compute_intensity_db(REAL_WAV)
    assert isinstance(value, float)
    assert value == value  # not NaN
    assert -60.0 < value < 0.0  # dBFS-like relative level, sane range for real speech
