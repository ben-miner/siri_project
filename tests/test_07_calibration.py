import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

calib = import_module("07_calibration")


# -- filename parsing (the underscore-in-pass-AND-label trap) -----------------------------------------------------------

def test_parse_calibration_filename_handles_underscore_in_pass_and_label():
    session, pass_label, label = calib.parse_calibration_filename("s01_passA_modal_creak_cal")
    assert session == "s01"
    assert pass_label == "passA_modal"
    assert label == "creak_cal"


def test_parse_calibration_filename_handles_sustained_a():
    session, pass_label, label = calib.parse_calibration_filename("s02_passC_creak_sustained_a")
    assert session == "s02"
    assert pass_label == "passC_creak"
    assert label == "sustained_a"


def test_parse_calibration_filename_handles_single_word_label():
    session, pass_label, label = calib.parse_calibration_filename("s01_passB_natural_roomtone")
    assert label == "roomtone"
    assert pass_label == "passB_natural"


def test_parse_calibration_filename_rejects_unknown_label():
    with pytest.raises(ValueError):
        calib.parse_calibration_filename("s01_passA_modal_notalabel")


def test_canonical_pass_strips_prefix():
    assert calib.canonical_pass("passA_modal") == "A_modal"


def test_canonical_pass_rejects_missing_prefix():
    with pytest.raises(ValueError):
        calib.canonical_pass("A_modal")


# -- find_calibration_wavs -----------------------------------------------------------

def test_find_calibration_wavs_sorted(tmp_path):
    (tmp_path / "s02_passA_modal_roomtone.wav").write_bytes(b"")
    (tmp_path / "s01_passA_modal_roomtone.wav").write_bytes(b"")
    found = calib.find_calibration_wavs(tmp_path)
    assert [p.name for p in found] == ["s01_passA_modal_roomtone.wav", "s02_passA_modal_roomtone.wav"]


# -- phonpipe summary loading -----------------------------------------------------------

def test_load_phonpipe_summary_keys_by_stem(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text("file,f0_median_hz\ns01_passA_modal_roomtone.wav,120.5\n", encoding="utf-8")
    by_stem = calib.load_phonpipe_summary(p)
    assert set(by_stem.keys()) == {"s01_passA_modal_roomtone"}


def test_load_phonpipe_summary_rejects_duplicate_stem(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "file,f0_median_hz\ns01_x.wav,1\ns01_x.wav,2\n", encoding="utf-8",
    )
    with pytest.raises(ValueError):
        calib.load_phonpipe_summary(p)


# -- symmetric_pct_diff -----------------------------------------------------------

def test_symmetric_pct_diff_identical_values_is_zero():
    assert calib.symmetric_pct_diff(100.0, 100.0) == 0.0


def test_symmetric_pct_diff_known_case():
    # |110-100| / ((110+100)/2) * 100 = 10/105*100 ~= 9.52%
    assert calib.symmetric_pct_diff(110.0, 100.0) == pytest.approx(9.5238, abs=0.001)


def test_symmetric_pct_diff_both_zero_is_zero():
    assert calib.symmetric_pct_diff(0.0, 0.0) == 0.0


def test_symmetric_pct_diff_nan_propagates():
    assert math.isnan(calib.symmetric_pct_diff(math.nan, 5.0))


def test_symmetric_pct_diff_sign_crossing():
    # H1-H2 can be negative (creak); formula should still produce a sane, large number
    pct = calib.symmetric_pct_diff(-2.0, 1.0)
    assert pct == pytest.approx(200.0)


# -- find_session_flags -----------------------------------------------------------

def make_report(s1_vals, s2_vals):
    return {"s01": s1_vals, "s02": s2_vals}


def test_find_session_flags_detects_large_difference():
    report = make_report({"hnr_db": 20.0}, {"hnr_db": 10.0})
    flags = calib.find_session_flags(report)
    assert len(flags) == 1
    assert flags[0]["measure"] == "hnr_db"


def test_find_session_flags_none_when_close():
    report = make_report({"hnr_db": 20.0}, {"hnr_db": 20.5})
    assert calib.find_session_flags(report) == []


def test_find_session_flags_handles_multiple_measures():
    report = make_report(
        {"hnr_db": 20.0, "jitter_pct": 1.0},
        {"hnr_db": 10.0, "jitter_pct": 1.01},
    )
    flags = calib.find_session_flags(report)
    measures_flagged = {f["measure"] for f in flags}
    assert measures_flagged == {"hnr_db"}


def test_find_session_flags_empty_with_wrong_session_count():
    assert calib.find_session_flags({"s01": {"x": 1.0}}) == []
    assert calib.find_session_flags({"s01": {"x": 1.0}, "s02": {"x": 1.0}, "s03": {"x": 1.0}}) == []


# -- aggregate_session_report (pure, given plain dict rows) -----------------------------------------------------------

def make_row(session, label, **measures):
    row = {"session": session, "pass": "A_modal", "label": label, "wav_path": "x.wav"}
    row.update(measures)
    return row


def test_aggregate_session_report_averages_across_passes():
    rows = [
        make_row("s01", "creak_cal", creak_f0_median_hz="100", creak_f0_p10_hz="80",
                 creak_f0_p90_hz="120", creak_shr_median="0.3", h1_h2_db="-2.0"),
        make_row("s01", "creak_cal", creak_f0_median_hz="120", creak_f0_p10_hz="90",
                 creak_f0_p90_hz="140", creak_shr_median="0.4", h1_h2_db="-3.0"),
        make_row("s01", "sustained_a", jitter_local_pct="0.5", shimmer_local_pct="2.0", hnr_db="20"),
        make_row("s01", "roomtone", rms_level=0.001),
    ]
    report = calib.aggregate_session_report(rows, session_speech_rms={"s01": 0.1})
    s01 = report["s01"]
    assert s01["creak_f0_median_hz"] == pytest.approx(110.0)
    assert s01["creak_h1_h2_db"] == pytest.approx(-2.5)
    assert s01["sustained_hnr_db"] == pytest.approx(20.0)
    assert s01["roomtone_rms_level"] == pytest.approx(0.001)
    assert s01["roomtone_snr_db"] == pytest.approx(20 * math.log10(0.1 / 0.001))


def test_aggregate_session_report_missing_speech_rms_gives_nan_snr():
    rows = [make_row("s01", "roomtone", rms_level=0.001)]
    report = calib.aggregate_session_report(rows, session_speech_rms={})
    assert math.isnan(report["s01"]["roomtone_snr_db"])


# -- CSV writing -----------------------------------------------------------

def test_write_calibration_csv_orders_lead_columns_first(tmp_path):
    rows = [{
        "session": "s01", "pass": "A_modal", "label": "roomtone", "wav_path": "x.wav",
        "f0_median_hz": "100", "rms_level": 0.001,
    }]
    out = tmp_path / "calibration.csv"
    calib.write_calibration_csv(rows, out)

    with out.open(encoding="utf-8") as f:
        header = next(csv_reader(f))
    assert header[:4] == ["session", "pass", "label", "wav_path"]
    assert "f0_median_hz" in header


def csv_reader(f):
    import csv
    return csv.reader(f)


# -- integration: build_calibration_rows + compute_rms with real synthetic audio --------------------------------------

@pytest.fixture
def synthetic_calibration_clip(tmp_path):
    import parselmouth

    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    # a simple tone, not silence, so rms > 0
    sound = parselmouth.Sound(
        values=[[0.1] * (48000 * 2)], sampling_frequency=48000,
    )
    path = calib_dir / "s01_passA_modal_roomtone.wav"
    sound.save(str(path), parselmouth.SoundFileFormat.WAV)
    return path


def test_compute_rms_nonzero_for_real_signal(synthetic_calibration_clip):
    import parselmouth
    snd = parselmouth.Sound(str(synthetic_calibration_clip))
    rms = calib.compute_rms(snd)
    assert rms > 0


def test_build_calibration_rows_end_to_end(synthetic_calibration_clip):
    summary_by_stem = {
        "s01_passA_modal_roomtone": {
            "file": "s01_passA_modal_roomtone.wav",
            "f0_median_hz": "100.0", "f0_floor_used": "60", "f0_ceiling_used": "400",
            "formant_ceiling_hz": "5500",
        }
    }
    rows = calib.build_calibration_rows([synthetic_calibration_clip], summary_by_stem)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "s01"
    assert row["pass"] == "A_modal"
    assert row["label"] == "roomtone"
    assert row["rms_level"] > 0
    # roomtone isn't creak_cal, so H1-H2 should be NaN, not computed
    assert math.isnan(row["h1_h2_db"])
    # ...and the constrained creak-only measures should also be NaN
    assert math.isnan(row["creak_f0_median_hz"])
    assert math.isnan(row["creak_shr_median"])


# -- constrained F0/SHR for creak_cal (the fix for the bad adaptive search) -----------------------------------------------------------

@pytest.fixture
def synthetic_low_f0_tone(tmp_path):
    import numpy as np
    import parselmouth

    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    sr = 48000
    f0 = 80.0  # within CREAK_F0_FLOOR_HZ..CREAK_F0_CEILING_HZ, well outside phonpipe's
    # bad adaptive floor of ~350-369Hz seen on the real corpus
    t = np.arange(int(sr * 2)) / sr
    tone = 0.3 * np.sin(2 * np.pi * f0 * t)
    sound = parselmouth.Sound(values=[tone], sampling_frequency=sr)
    path = calib_dir / "s01_passA_modal_creak_cal.wav"
    sound.save(str(path), parselmouth.SoundFileFormat.WAV)
    return path, f0


def test_compute_constrained_f0_recovers_known_frequency(synthetic_low_f0_tone):
    import parselmouth
    path, f0 = synthetic_low_f0_tone
    snd = parselmouth.Sound(str(path))
    result = calib.compute_constrained_f0(snd)
    assert result["creak_f0_median_hz"] == pytest.approx(f0, abs=2.0)
    assert result["creak_f0_pct_voiced"] > 0


def test_compute_constrained_shr_runs_without_error(synthetic_low_f0_tone):
    import parselmouth
    path, _ = synthetic_low_f0_tone
    snd = parselmouth.Sound(str(path))
    result = calib.compute_constrained_shr(snd)
    assert "creak_shr_median" in result


def test_build_calibration_rows_populates_constrained_creak_measures(synthetic_low_f0_tone):
    path, f0 = synthetic_low_f0_tone
    summary_by_stem = {
        "s01_passA_modal_creak_cal": {
            "file": "s01_passA_modal_creak_cal.wav",
            "f0_median_hz": "650.0",  # the bad phonpipe-adaptive value, left untouched
            "f0_floor_used": "360", "f0_ceiling_used": "750",
            "formant_ceiling_hz": "5500",
        }
    }
    rows = calib.build_calibration_rows([path], summary_by_stem)
    row = rows[0]
    # the new constrained column should be near the true low F0, not phonpipe's bad one
    assert row["creak_f0_median_hz"] == pytest.approx(f0, abs=2.0)
    # phonpipe's own (contaminated) column is preserved untouched for comparison
    assert row["f0_median_hz"] == "650.0"
    assert not math.isnan(row["h1_h2_db"])
