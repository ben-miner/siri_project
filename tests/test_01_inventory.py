import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

inventory = import_module("01_inventory")


def test_is_conforming_accepts_expected_format():
    assert inventory.is_conforming("44100", 1, "s16") is True


def test_is_conforming_rejects_wrong_sample_rate():
    assert inventory.is_conforming("48000", 1, "s16") is False


def test_is_conforming_rejects_stereo():
    assert inventory.is_conforming("44100", 2, "s16") is False


def test_is_conforming_rejects_wrong_sample_fmt():
    assert inventory.is_conforming("44100", 1, "s16p") is False


def test_is_conforming_rejects_none_channels():
    assert inventory.is_conforming("44100", None, "s16") is False


def test_find_audio_files_walks_recursively_and_filters_extensions(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "sub" / "b.m4a").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "c.MP3").write_bytes(b"")

    found = inventory.find_audio_files(tmp_path)
    names = sorted(p.name for p in found)

    assert names == ["a.wav", "b.m4a", "c.MP3"]
