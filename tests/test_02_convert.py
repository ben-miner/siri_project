import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

convert = import_module("02_convert")


def test_find_source_wavs_excludes_originals_m4a_subfolder(tmp_path):
    (tmp_path / "sess1").mkdir()
    (tmp_path / "originals_m4a").mkdir()
    (tmp_path / "sess1" / "a.wav").write_bytes(b"")
    (tmp_path / "originals_m4a" / "b.wav").write_bytes(b"")
    (tmp_path / "c.wav").write_bytes(b"")

    found = convert.find_source_wavs(tmp_path)
    names = sorted(p.name for p in found)

    assert names == ["a.wav", "c.wav"]


def test_find_source_wavs_ignores_non_wav_files(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "a.m4a").write_bytes(b"")

    found = convert.find_source_wavs(tmp_path)

    assert [p.name for p in found] == ["a.wav"]


def test_needs_conversion_true_when_output_missing(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"")
    dst = tmp_path / "dst.wav"

    assert convert.needs_conversion(src, dst) is True


def test_needs_conversion_false_when_output_newer(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"")
    dst = tmp_path / "dst.wav"
    dst.write_bytes(b"")
    # ensure a filesystem-visible mtime gap
    now = time.time()
    import os
    os.utime(src, (now, now))
    os.utime(dst, (now + 10, now + 10))

    assert convert.needs_conversion(src, dst) is False


def test_needs_conversion_true_when_source_newer(tmp_path):
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    dst.write_bytes(b"")
    src.write_bytes(b"")
    now = time.time()
    import os
    os.utime(dst, (now, now))
    os.utime(src, (now + 10, now + 10))

    assert convert.needs_conversion(src, dst) is True


def test_find_output_collisions_detects_shared_stem(tmp_path):
    sources = [
        tmp_path / "sess1" / "utt001.wav",
        tmp_path / "sess2" / "utt001.wav",
        tmp_path / "sess1" / "utt002.wav",
    ]

    collisions = convert.find_output_collisions(sources)

    assert set(collisions.keys()) == {"utt001"}
    assert len(collisions["utt001"]) == 2


def test_find_output_collisions_empty_when_all_unique(tmp_path):
    sources = [
        tmp_path / "sess1" / "utt001.wav",
        tmp_path / "sess1" / "utt002.wav",
    ]

    assert convert.find_output_collisions(sources) == {}
