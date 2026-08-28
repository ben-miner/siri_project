import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

select = import_module("08a_select_annotation_sample")


def make_row(utt_id, pass_name, creak_rate):
    return {
        "utt_id": utt_id, "pass": pass_name, "wav_path": f"data/split/{utt_id}.wav",
        "creak_doubling_rate": str(creak_rate),
    }


# -- rows_with_creak_rate -----------------------------------------------------------

def test_rows_with_creak_rate_parses_float():
    rows = [make_row("a", "A_modal", "0.25")]
    out = select.rows_with_creak_rate(rows)
    assert out[0]["_creak_rate"] == pytest.approx(0.25)


def test_rows_with_creak_rate_drops_nan(capsys):
    rows = [make_row("a", "A_modal", "0.25"), {"utt_id": "b", "pass": "A_modal",
             "wav_path": "x.wav", "creak_doubling_rate": "nan"}]
    out = select.rows_with_creak_rate(rows)
    assert [r["utt_id"] for r in out] == ["a"]
    assert "b" in capsys.readouterr().out


def test_rows_with_creak_rate_drops_missing_value():
    rows = [make_row("a", "A_modal", "0.25"), {"utt_id": "b", "pass": "A_modal", "wav_path": "x.wav"}]
    out = select.rows_with_creak_rate(rows)
    assert [r["utt_id"] for r in out] == ["a"]


# -- assign_terciles -----------------------------------------------------------

def test_assign_terciles_splits_by_rank_into_equal_groups():
    rows = [{"_creak_rate": v, "utt_id": str(v)} for v in range(9)]  # 0..8
    terciles = select.assign_terciles(rows)
    assert [r["_creak_rate"] for r in terciles["low"]] == [0, 1, 2]
    assert [r["_creak_rate"] for r in terciles["mid"]] == [3, 4, 5]
    assert [r["_creak_rate"] for r in terciles["high"]] == [6, 7, 8]


def test_assign_terciles_low_has_lowest_values_regardless_of_input_order():
    rows = [{"_creak_rate": v, "utt_id": str(v)} for v in [5, 1, 8, 0, 4, 7, 2, 6, 3]]
    terciles = select.assign_terciles(rows)
    assert sorted(r["_creak_rate"] for r in terciles["low"]) == [0, 1, 2]
    assert sorted(r["_creak_rate"] for r in terciles["high"]) == [6, 7, 8]


# -- sample_pass_balanced -----------------------------------------------------------

def test_sample_pass_balanced_splits_evenly_when_all_passes_available():
    pool = (
        [make_row(f"a{i}", "A_modal", 0.1) for i in range(10)]
        + [make_row(f"b{i}", "B_natural", 0.1) for i in range(10)]
        + [make_row(f"c{i}", "C_creak", 0.1) for i in range(10)]
    )
    rng = random.Random(0)
    picked = select.sample_pass_balanced(pool, target=9, rng=rng)
    assert len(picked) == 9
    from collections import Counter
    counts = Counter(r["pass"] for r in picked)
    assert counts == {"A_modal": 3, "B_natural": 3, "C_creak": 3}


def test_sample_pass_balanced_reallocates_when_a_pass_is_completely_absent():
    """The real corpus's low tercile has zero C_creak utterances -- the
    round-robin must fall back to splitting the target across the passes
    that DO exist, not silently return fewer than target."""
    pool = (
        [make_row(f"a{i}", "A_modal", 0.1) for i in range(10)]
        + [make_row(f"b{i}", "B_natural", 0.1) for i in range(10)]
    )
    rng = random.Random(0)
    picked = select.sample_pass_balanced(pool, target=10, rng=rng)
    assert len(picked) == 10
    from collections import Counter
    counts = Counter(r["pass"] for r in picked)
    assert counts == {"A_modal": 5, "B_natural": 5}
    assert "C_creak" not in counts


def test_sample_pass_balanced_uses_all_of_a_scarce_pass_then_fills_from_dominant():
    pool = (
        [make_row(f"a{i}", "A_modal", 0.1) for i in range(2)]  # scarce
        + [make_row(f"c{i}", "C_creak", 0.1) for i in range(50)]  # dominant
    )
    rng = random.Random(0)
    picked = select.sample_pass_balanced(pool, target=10, rng=rng)
    assert len(picked) == 10
    from collections import Counter
    counts = Counter(r["pass"] for r in picked)
    assert counts["A_modal"] == 2  # all available used
    assert counts["C_creak"] == 8


def test_sample_pass_balanced_returns_short_list_if_pool_too_small():
    pool = [make_row("a0", "A_modal", 0.1), make_row("b0", "B_natural", 0.1)]
    rng = random.Random(0)
    picked = select.sample_pass_balanced(pool, target=10, rng=rng)
    assert len(picked) == 2  # cannot manufacture utterances that don't exist


# -- build_sample (integration) -----------------------------------------------------------

def _synthetic_corpus():
    rows = []
    for i in range(80):
        rows.append(make_row(f"a{i}", "A_modal", i / 240.0))  # low range
    for i in range(80):
        rows.append(make_row(f"b{i}", "B_natural", 0.34 + i / 240.0))  # mid range
    for i in range(80):
        rows.append(make_row(f"c{i}", "C_creak", 0.67 + i / 240.0))  # high range
    return rows


def test_build_sample_selects_60_total_20_per_tercile():
    rng = random.Random(20260821)
    selected = select.build_sample(_synthetic_corpus(), rng)
    assert len(selected) == 60
    from collections import Counter
    tercile_counts = Counter(r["tercile"] for r in selected)
    assert tercile_counts == {"low": 20, "mid": 20, "high": 20}


def test_build_sample_exits_when_a_tercile_pool_cannot_fill_target():
    tiny_corpus = [make_row(f"x{i}", "A_modal", i) for i in range(5)]  # way < 60
    rng = random.Random(0)
    with pytest.raises(SystemExit):
        select.build_sample(tiny_corpus, rng)


# -- assign_annotation_order -----------------------------------------------------------

def test_assign_annotation_order_is_a_permutation_of_1_to_n():
    rows = [{"utt_id": str(i)} for i in range(10)]
    rng = random.Random(20260821)
    out = select.assign_annotation_order(rows, rng)
    assert sorted(r["annotation_order"] for r in out) == list(range(1, 11))


def test_assign_annotation_order_is_reproducible_with_same_seed():
    rows_a = [{"utt_id": str(i)} for i in range(10)]
    rows_b = [{"utt_id": str(i)} for i in range(10)]
    out_a = select.assign_annotation_order(rows_a, random.Random(20260821))
    out_b = select.assign_annotation_order(rows_b, random.Random(20260821))
    assert [r["utt_id"] for r in out_a] == [r["utt_id"] for r in out_b]


def test_assign_annotation_order_sorts_output_by_order():
    rows = [{"utt_id": str(i)} for i in range(10)]
    rng = random.Random(20260821)
    out = select.assign_annotation_order(rows, rng)
    assert [r["annotation_order"] for r in out] == list(range(1, 11))


# -- CSV writing -----------------------------------------------------------

def _annotated_rows():
    return [
        {"utt_id": "a1", "wav_path": "data/split/a1.wav", "pass": "A_modal",
         "_creak_rate": 0.12, "tercile": "low", "annotation_order": 2},
        {"utt_id": "c1", "wav_path": "data/split/c1.wav", "pass": "C_creak",
         "_creak_rate": 0.91, "tercile": "high", "annotation_order": 1},
    ]


def test_write_sample_csv_includes_stratification_columns(tmp_path):
    out = tmp_path / "sample.csv"
    select.write_sample_csv(_annotated_rows(), out)
    import csv
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {
        "utt_id", "wav_path", "pass", "creak_doubling_rate", "tercile", "annotation_order"}


def test_write_blind_csv_excludes_creak_rate_pass_and_tercile(tmp_path):
    out = tmp_path / "blind.csv"
    select.write_blind_csv(_annotated_rows(), out)
    import csv
    with out.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["utt_id", "annotation_order"]
    assert "pass" not in header
    assert "creak_doubling_rate" not in header
    assert "tercile" not in header
