import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import numpy as np
import pytest

tune = import_module("08d_tune_thresholds")

REAL_WAV = Path(r"c:\Users\benmi\siri_project\data\split\s01_A_modal_C001.wav")


# -- precision_recall_f1 -----------------------------------------------------------

def test_precision_recall_f1_perfect():
    p, r, f1 = tune.precision_recall_f1([1, 1, 0, 0], [1, 1, 0, 0])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_precision_recall_f1_all_wrong():
    p, r, f1 = tune.precision_recall_f1([1, 1, 0, 0], [0, 0, 1, 1])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_precision_recall_f1_no_positive_predictions():
    p, r, f1 = tune.precision_recall_f1([1, 0], [0, 0])
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_precision_recall_f1_no_positive_labels():
    # predicting positive when there are no true positives: precision 0, recall undefined -> 0
    p, r, f1 = tune.precision_recall_f1([0, 0], [1, 0])
    assert p == 0.0
    assert r == 0.0


# -- nan_rate -----------------------------------------------------------

def test_nan_rate_no_nans():
    assert tune.nan_rate([1.0, 2.0, 3.0]) == 0.0


def test_nan_rate_all_nans():
    assert tune.nan_rate([float("nan"), float("nan")]) == 1.0


def test_nan_rate_mixed():
    assert tune.nan_rate([1.0, float("nan"), 3.0, float("nan")]) == pytest.approx(0.5)


def test_nan_rate_empty():
    assert tune.nan_rate([]) == 0.0


# -- trivial_classifier_f1 -----------------------------------------------------------

def test_trivial_classifier_f1_matches_always_predict_positive():
    labels = [1, 1, 0, 0, 0]  # prevalence 0.4
    expected_precision = 0.4
    expected_f1 = 2 * expected_precision / (1 + expected_precision)
    assert tune.trivial_classifier_f1(labels) == pytest.approx(expected_f1)


def test_trivial_classifier_f1_zero_when_no_positives():
    assert tune.trivial_classifier_f1([0, 0, 0]) == 0.0


def test_trivial_classifier_f1_one_when_all_positive():
    assert tune.trivial_classifier_f1([1, 1, 1]) == pytest.approx(1.0)


def test_trivial_classifier_f1_zero_for_empty_labels():
    assert tune.trivial_classifier_f1([]) == 0.0


def test_precision_recall_f1_partial():
    # true: 1,1,1,0 pred: 1,1,0,0 -> tp=2,fp=0,fn=1 -> precision=1, recall=2/3
    p, r, f1 = tune.precision_recall_f1([1, 1, 1, 0], [1, 1, 0, 0])
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))


# -- grid_search_threshold -----------------------------------------------------------

def test_grid_search_threshold_finds_perfect_separator_greater_than():
    values = [1.0, 2.0, 8.0, 9.0]
    labels = [0, 0, 1, 1]
    best = tune.grid_search_threshold(values, labels, np.arange(0, 10, 0.5), ">")
    assert best["f1"] == pytest.approx(1.0)
    assert 2.0 <= best["threshold"] < 8.0


def test_grid_search_threshold_finds_perfect_separator_less_than():
    values = [1.0, 2.0, 8.0, 9.0]
    labels = [1, 1, 0, 0]
    best = tune.grid_search_threshold(values, labels, np.arange(0, 10, 0.5), "<")
    assert best["f1"] == pytest.approx(1.0)
    assert 2.0 <= best["threshold"] < 8.0


def test_grid_search_threshold_drops_nan_values():
    values = [1.0, float("nan"), 8.0, 9.0]
    labels = [0, 1, 1, 1]
    best = tune.grid_search_threshold(values, labels, np.arange(0, 10, 0.5), ">")
    assert best["n"] == 3  # the NaN row excluded


def test_grid_search_threshold_returns_none_when_all_nan():
    values = [float("nan"), float("nan")]
    labels = [0, 1]
    best = tune.grid_search_threshold(values, labels, np.arange(0, 10, 0.5), ">")
    assert best is None


# -- grid_search_low_f0_percentile -----------------------------------------------------------

def test_grid_search_low_f0_percentile_finds_separator():
    reference = list(np.linspace(100, 200, 100))  # speaker's modal F0 spread 100-200Hz
    token_f0s = [60.0, 65.0, 190.0, 195.0]  # first two are clearly low (creaky), last two aren't
    labels = [1, 1, 0, 0]
    best = tune.grid_search_low_f0_percentile(token_f0s, labels, reference, range(1, 51))
    assert best["f1"] == pytest.approx(1.0)
    assert best["cutoff_hz"] < 190.0


def test_grid_search_low_f0_percentile_none_when_no_reference():
    best = tune.grid_search_low_f0_percentile([60.0], [1], [], range(1, 51))
    assert best is None


def test_grid_search_low_f0_percentile_none_when_no_valid_tokens():
    best = tune.grid_search_low_f0_percentile([float("nan")], [1], [100.0, 200.0], range(1, 51))
    assert best is None


def test_grid_search_low_f0_percentile_detects_plateau():
    # reference is bimodal (50x100Hz, 50x200Hz): percentiles 1-49 all map to
    # exactly 100.0Hz (verified via np.percentile), so F1 is tied across that
    # whole range before jumping at 50 (150.0Hz) and 51+ (200.0Hz).
    reference = [100.0] * 50 + [200.0] * 50
    token_f0s = [90.0, 95.0, 150.0, 160.0]
    labels = [1, 1, 0, 0]
    best = tune.grid_search_low_f0_percentile(token_f0s, labels, reference, range(1, 60))
    assert best["threshold"] == 1
    assert best["f1"] == pytest.approx(1.0)
    assert best["plateau_pct_max"] == 49
    assert best["plateau_hz_max"] == pytest.approx(100.0)
    assert best["n_reference"] == 100


# -- predict_component -----------------------------------------------------------

def test_predict_component_returns_none_for_nan():
    best = {"threshold": 0.0}
    assert tune.predict_component("h1_h2_db", float("nan"), best, []) is None


def test_predict_component_h1_h2_direction_is_less_than():
    best = {"threshold": 0.0}
    assert tune.predict_component("h1_h2_db", -5.0, best, []) == 1
    assert tune.predict_component("h1_h2_db", 5.0, best, []) == 0


def test_predict_component_shr_direction_is_greater_than():
    best = {"threshold": 0.3}
    assert tune.predict_component("shr_doubling", 0.5, best, []) == 1
    assert tune.predict_component("shr_doubling", 0.1, best, []) == 0


def test_predict_component_jitter_direction_is_greater_than():
    best = {"threshold": 2.0}
    assert tune.predict_component("jitter_irregularity_pct", 5.0, best, []) == 1
    assert tune.predict_component("jitter_irregularity_pct", 1.0, best, []) == 0


def test_predict_component_low_f0_uses_cutoff_hz():
    best = {"threshold": 10, "cutoff_hz": 100.0}
    assert tune.predict_component("low_f0_percentile", 80.0, best, []) == 1
    assert tune.predict_component("low_f0_percentile", 150.0, best, []) == 0


# -- evaluate_by_pass -----------------------------------------------------------

def test_evaluate_by_pass_splits_correctly():
    token_rows = [
        {"pass": "A_modal", "hand_creaky": 0, "h1_h2_db": 5.0, "shr": 0.1,
         "f0_hz": 150.0, "jitter_pct": 1.0},
        {"pass": "C_creak", "hand_creaky": 1, "h1_h2_db": -5.0, "shr": 0.5,
         "f0_hz": 60.0, "jitter_pct": 5.0},
    ]
    results = tune.tune_all_components(token_rows, reference_f0s=[100.0, 150.0, 200.0])
    by_pass = tune.evaluate_by_pass(token_rows, results, reference_f0s=[100.0, 150.0, 200.0])
    assert ("A_modal", "h1_h2_db") in by_pass
    assert ("C_creak", "h1_h2_db") in by_pass
    assert by_pass[("A_modal", "h1_h2_db")]["n"] == 1
    assert by_pass[("C_creak", "h1_h2_db")]["n"] == 1


def test_tune_all_components_attaches_trivial_f1():
    token_rows = [
        {"pass": "A_modal", "hand_creaky": 0, "h1_h2_db": 5.0, "shr": 0.1,
         "f0_hz": 150.0, "jitter_pct": 1.0},
        {"pass": "C_creak", "hand_creaky": 1, "h1_h2_db": -5.0, "shr": 0.5,
         "f0_hz": 60.0, "jitter_pct": 5.0},
    ]
    results = tune.tune_all_components(token_rows, reference_f0s=[100.0, 150.0, 200.0])
    for name, best in results.items():
        assert best is not None
        assert "trivial_f1" in best


# -- CSV loading -----------------------------------------------------------

def test_load_acoustics_by_utt_id_reads_rows(tmp_path):
    p = tmp_path / "acoustics_joined.csv"
    p.write_text(
        "utt_id,pass,f0_floor_used,f0_ceiling_used,formant_ceiling_hz,f0_median_hz\n"
        "s01_x,A_modal,80,300,5000,150\n",
        encoding="utf-8",
    )
    result = tune.load_acoustics_by_utt_id(p)
    assert result["s01_x"]["pass"] == "A_modal"


def test_load_acoustics_by_utt_id_rejects_duplicates(tmp_path):
    p = tmp_path / "acoustics_joined.csv"
    p.write_text(
        "utt_id,pass\ns01_x,A_modal\ns01_x,A_modal\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        tune.load_acoustics_by_utt_id(p)


def test_load_speaker_modal_f0_filters_to_a_modal():
    acoustics = {
        "a": {"pass": "A_modal", "f0_median_hz": "150.0"},
        "b": {"pass": "C_creak", "f0_median_hz": "60.0"},
        "c": {"pass": "A_modal", "f0_median_hz": "nan"},
    }
    result = tune.load_speaker_modal_f0(acoustics)
    assert result == [150.0]


# -- per-token measurement (integration, real audio) -----------------------------------------------------------

pytestmark_real_wav = pytest.mark.skipif(
    not REAL_WAV.exists(), reason=f"{REAL_WAV} not found (project audio, not part of this repo)")


@pytestmark_real_wav
def test_measure_token_h1_h2_returns_a_float():
    import parselmouth
    snd = parselmouth.Sound(str(REAL_WAV))
    value = tune.measure_token_h1_h2(snd, 0.78, 0.86, 60.0, 300.0, 5000.0)
    assert isinstance(value, float)


@pytestmark_real_wav
def test_measure_token_shr_returns_a_float():
    import parselmouth
    snd = parselmouth.Sound(str(REAL_WAV))
    value = tune.measure_token_shr(snd, 0.78, 0.86, 60.0, 300.0)
    assert isinstance(value, float)


@pytestmark_real_wav
def test_measure_token_f0_returns_a_plausible_value_or_nan():
    import parselmouth
    snd = parselmouth.Sound(str(REAL_WAV))
    value = tune.measure_token_f0(snd, 0.78, 0.86, 60.0, 300.0)
    assert value != value or 40.0 <= value <= 400.0  # NaN or plausible Hz


@pytestmark_real_wav
def test_measure_token_jitter_returns_a_float():
    import parselmouth
    snd = parselmouth.Sound(str(REAL_WAV))
    value = tune.measure_token_jitter(snd, 0.78, 0.86, 60.0, 300.0)
    assert isinstance(value, float)


@pytestmark_real_wav
def test_measure_token_functions_handle_a_tiny_span_without_raising():
    import parselmouth
    snd = parselmouth.Sound(str(REAL_WAV))
    # a near-zero-width span should degrade to NaN, not raise
    h1h2 = tune.measure_token_h1_h2(snd, 0.780, 0.781, 60.0, 300.0, 5000.0)
    shr = tune.measure_token_shr(snd, 0.780, 0.781, 60.0, 300.0)
    f0 = tune.measure_token_f0(snd, 0.780, 0.781, 60.0, 300.0)
    jitter = tune.measure_token_jitter(snd, 0.780, 0.781, 60.0, 300.0)
    for v in (h1h2, shr, f0, jitter):
        assert isinstance(v, float)
