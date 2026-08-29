import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

baseline = import_module("10b_whisper_baseline")

REAL_WAV = Path(r"c:\Users\benmi\siri_project\data\split\s01_A_modal_C001.wav")


# -- already_done_utt_ids -----------------------------------------------------------

def test_already_done_utt_ids_missing_file_returns_empty(tmp_path):
    assert baseline.already_done_utt_ids(tmp_path / "nope.csv") == set()


def test_already_done_utt_ids_reads_existing_rows(tmp_path):
    p = tmp_path / "wer_whisper.csv"
    p.write_text("utt_id,hypothesis\ns01_x,hello\ns01_y,world\n", encoding="utf-8")
    assert baseline.already_done_utt_ids(p) == {"s01_x", "s01_y"}


# -- score_against_verbatim -----------------------------------------------------------

def test_score_against_verbatim_perfect_match():
    row = {"verbatim_text": "Set a timer for twelve minutes."}
    result = baseline.score_against_verbatim(row, "Set a timer for 12 minutes.")
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["reference_length"] == 6
    assert result["wer"] == pytest.approx(0.0)


def test_score_against_verbatim_with_a_substitution():
    row = {"verbatim_text": "The cat sat."}
    result = baseline.score_against_verbatim(row, "The dog sat.")
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["reference_length"] == 3
    assert result["wer"] == pytest.approx(1 / 3, abs=1e-4)  # score_against_verbatim rounds to 4dp
    assert result["alignment"] == "the [cat->dog] sat"


# -- resolve_device_and_compute_type -----------------------------------------------------------

def test_resolve_device_and_compute_type_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 0)
    assert baseline.resolve_device_and_compute_type() == ("cpu", "int8")


def test_resolve_device_and_compute_type_cuda_when_available(monkeypatch):
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 1)
    assert baseline.resolve_device_and_compute_type() == ("cuda", "float16")


# -- transcribe_one (integration, real audio + a small real model) -----------------------------------------------------------

pytestmark_real_wav = pytest.mark.skipif(
    not REAL_WAV.exists(), reason=f"{REAL_WAV} not found (project audio, not part of this repo)")


@pytestmark_real_wav
def test_transcribe_one_returns_expected_structure():
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
    result = baseline.transcribe_one(model, REAL_WAV, "tiny", "int8")
    assert isinstance(result["hypothesis"], str)
    assert result["hypothesis"] != ""
    assert result["model_size"] == "tiny"
    assert result["compute_type"] == "int8"
    assert isinstance(result["elapsed_ms"], float)
    assert result["elapsed_ms"] > 0
