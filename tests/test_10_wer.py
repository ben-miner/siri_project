import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

import pytest

wer_mod = import_module("10_wer")


# -- number_to_words -----------------------------------------------------------

@pytest.mark.parametrize("n, expected", [
    (0, "zero"),
    (7, "seven"),
    (12, "twelve"),
    (19, "nineteen"),
    (20, "twenty"),
    (45, "forty-five"),
    (99, "ninety-nine"),
    (100, "one hundred"),
    (143, "one hundred forty-three"),
    (1000, "one thousand"),
    (2024, "two thousand twenty-four"),
])
def test_number_to_words(n, expected):
    assert wer_mod.number_to_words(n) == expected


def test_number_to_words_rejects_negative():
    with pytest.raises(ValueError):
        wer_mod.number_to_words(-1)


# -- expand_numerals -----------------------------------------------------------

def test_expand_numerals_basic():
    assert wer_mod.expand_numerals("set a timer for 12 minutes") == "set a timer for twelve minutes"


def test_expand_numerals_zero():
    assert wer_mod.expand_numerals("i have 0 items") == "i have zero items"


def test_expand_numerals_no_digits_unchanged():
    assert wer_mod.expand_numerals("no numbers here") == "no numbers here"


# -- expand_symbols -----------------------------------------------------------

def test_expand_symbols_currency_reorders_symbol_after_number():
    assert wer_mod.expand_symbols("$43") == "forty-three dollars"


def test_expand_symbols_percent_no_reorder_needed():
    assert wer_mod.expand_symbols("20%") == "twenty percent"


def test_expand_symbols_degree_no_reorder_needed():
    assert wer_mod.expand_symbols("80°") == "eighty degrees"


def test_expand_symbols_in_context():
    assert wer_mod.expand_symbols("cost is $5 today") == "cost is five dollars today"
    assert wer_mod.expand_symbols("50% off and $10 more") == "fifty percent off and ten dollars more"


def test_expand_symbols_bare_symbol_not_adjacent_to_digit():
    assert wer_mod.expand_symbols("just a $ sign") == "just a dollars sign"


def test_expand_symbols_degree_adjacent_to_letter_no_space():
    # real ASR output for the "eighty degrees Fahrenheit" stimulus item:
    # "80°F" with no space before the unit abbreviation. Without padding,
    # this used to fuse into the single garbage token "degreesf".
    assert wer_mod.expand_symbols("80°f") == "eighty degrees f"


# -- lowercase -----------------------------------------------------------

def test_lowercase():
    assert wer_mod.lowercase("HELLO World") == "hello world"


# -- strip_punctuation -----------------------------------------------------------

def test_strip_punctuation_keeps_intra_word_apostrophe():
    assert wer_mod.strip_punctuation("don't stop") == "don't stop"


def test_strip_punctuation_drops_leading_and_trailing_apostrophe():
    assert wer_mod.strip_punctuation("'quoted' word") == "quoted word"


def test_strip_punctuation_hyphen_becomes_space():
    assert wer_mod.strip_punctuation("forty-five") == "forty five"


def test_strip_punctuation_removes_generic_punctuation():
    assert wer_mod.strip_punctuation("hello, world!") == "hello world"
    assert wer_mod.strip_punctuation("Tokyo?") == "Tokyo"


def test_strip_punctuation_colon_becomes_space():
    # real ASR output for the "six forty-five" stimulus item: "6:45".
    assert wer_mod.strip_punctuation("6:45") == "6 45"


# -- collapse_whitespace -----------------------------------------------------------

def test_collapse_whitespace_multiple_kinds():
    assert wer_mod.collapse_whitespace("a   b\tc\n\nd") == "a b c d"


def test_collapse_whitespace_strips_leading_and_trailing():
    assert wer_mod.collapse_whitespace("  leading and trailing  ") == "leading and trailing"


# -- tokenize -----------------------------------------------------------

def test_tokenize_basic():
    assert wer_mod.tokenize("a b c") == ["a", "b", "c"]


def test_tokenize_empty_string():
    assert wer_mod.tokenize("") == []


def test_tokenize_whitespace_only():
    assert wer_mod.tokenize("   ") == []


# -- normalize (integration) -----------------------------------------------------------

def test_normalize_basic_sentence():
    assert wer_mod.normalize("What's the weather in Newton tomorrow?") == [
        "what's", "the", "weather", "in", "newton", "tomorrow",
    ]


def test_normalize_hyphenated_number_splits_into_two_tokens():
    assert wer_mod.normalize("Set an alarm for six forty-five.") == [
        "set", "an", "alarm", "for", "six", "forty", "five",
    ]


def test_normalize_spelled_out_numerals_and_units():
    assert wer_mod.normalize("What is twenty percent of forty-three dollars?") == [
        "what", "is", "twenty", "percent", "of", "forty", "three", "dollars",
    ]


def test_normalize_digit_hypothesis_matches_spelled_out_reference():
    # The whole point of expand_numerals/expand_symbols: a hypothesis using
    # ASR's digit/symbol conventions must normalize identically to the
    # reference script's spelled-out convention.
    reference = wer_mod.normalize("What is twenty percent of forty-three dollars?")
    hypothesis = wer_mod.normalize("What is 20% of $43?")
    assert reference == hypothesis


def test_normalize_bare_numeral_hypothesis():
    assert wer_mod.normalize("Set a timer for 12 minutes.") == [
        "set", "a", "timer", "for", "twelve", "minutes",
    ]


def test_normalize_compact_temperature_matches_spelled_out_reference():
    # real ASR output ("Convert 80°F to Celsius") vs the reference script
    # ("Convert eighty degrees Fahrenheit to Celsius.") for stimulus C014 --
    # regression test for the degree/letter-fusion bug.
    hypothesis = wer_mod.normalize("Convert 80°F to Celsius")
    assert hypothesis == ["convert", "eighty", "degrees", "f", "to", "celsius"]


def test_normalize_compact_clock_time_matches_spelled_out_reference():
    # real ASR output ("Set an alarm for 6:45") vs the reference script
    # ("Set an alarm for six forty-five.") for stimulus C017 -- regression
    # test for the colon-fusion bug.
    reference = wer_mod.normalize("Set an alarm for six forty-five.")
    hypothesis = wer_mod.normalize("Set an alarm for 6:45")
    assert hypothesis == ["set", "an", "alarm", "for", "six", "forty", "five"]
    assert hypothesis == reference


# -- format_alignment -----------------------------------------------------------

def test_format_alignment():
    alignment = [("a", "a"), ("b", "x"), ("c", None), (None, "d")]
    assert wer_mod.format_alignment(alignment) == "a [b->x] [-c] [+d]"


# -- wer -----------------------------------------------------------

def test_wer_identical():
    result = wer_mod.wer(["the", "cat", "sat"], ["the", "cat", "sat"])
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["reference_length"] == 3
    assert result["wer"] == pytest.approx(0.0)
    assert result["alignment"] == [("the", "the"), ("cat", "cat"), ("sat", "sat")]


def test_wer_substitution_only():
    result = wer_mod.wer(["the", "cat", "sat"], ["the", "dog", "sat"])
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["reference_length"] == 3
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["alignment"] == [("the", "the"), ("cat", "dog"), ("sat", "sat")]


def test_wer_empty_hypothesis():
    # hand-computed: every reference token is a deletion.
    result = wer_mod.wer(["the", "cat", "sat"], [])
    assert result["substitutions"] == 0
    assert result["deletions"] == 3
    assert result["insertions"] == 0
    assert result["reference_length"] == 3
    assert result["wer"] == pytest.approx(1.0)
    assert result["alignment"] == [("the", None), ("cat", None), ("sat", None)]


def test_wer_insertion_heavy():
    # hand-computed: one match, three insertions -- WER can exceed 1.0.
    result = wer_mod.wer(["a"], ["a", "b", "c", "d"])
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 3
    assert result["reference_length"] == 1
    assert result["wer"] == pytest.approx(3.0)
    assert result["alignment"] == [("a", "a"), (None, "b"), (None, "c"), (None, "d")]


def test_wer_mixed_substitution_deletion():
    # hand-computed via the full DP table:
    # ref: i went to the store
    # hyp: i go   to     store
    # -> match(i), sub(went->go), match(to), del(the), match(store)
    result = wer_mod.wer(["i", "went", "to", "the", "store"], ["i", "go", "to", "store"])
    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert result["reference_length"] == 5
    assert result["wer"] == pytest.approx(0.4)
    assert result["alignment"] == [
        ("i", "i"), ("went", "go"), ("to", "to"), ("the", None), ("store", "store"),
    ]


def test_wer_empty_reference_and_empty_hypothesis():
    result = wer_mod.wer([], [])
    assert result["reference_length"] == 0
    assert result["alignment"] == []
    assert result["wer"] == pytest.approx(0.0)


def test_wer_empty_reference_nonempty_hypothesis_is_infinite():
    result = wer_mod.wer([], ["a", "b"])
    assert result["insertions"] == 2
    assert result["reference_length"] == 0
    assert result["wer"] == float("inf")


# -- CSV loading -----------------------------------------------------------

def test_load_manifest_reads_rows(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "utt_id,session,pass,script_text,verbatim_text,was_corrected\n"
        "s01_x,s01,A_modal,hello there,hello there,False\n",
        encoding="utf-8",
    )
    result = wer_mod.load_manifest(p)
    assert result["s01_x"]["pass"] == "A_modal"


def test_load_manifest_rejects_duplicate_utt_id(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "utt_id,pass\ns01_x,A_modal\ns01_x,A_modal\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        wer_mod.load_manifest(p)


def test_load_hypotheses_rejects_duplicate_utt_id(tmp_path):
    p = tmp_path / "hypotheses.csv"
    p.write_text(
        "utt_id,hypothesis,os_version,recognizer,elapsed_ms\n"
        "s01_x,hello,15.7.9,SFSpeechRecognizer,120.0\n"
        "s01_x,hello again,15.7.9,SFSpeechRecognizer,130.0\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        wer_mod.load_hypotheses(p)


# -- score_row (integration) -----------------------------------------------------------

def test_score_row_joins_and_scores_both_references():
    manifest_row = {
        "utt_id": "s01_x",
        "session": "s01",
        "pass": "A_modal",
        "was_corrected": "True",
        "script_text": "Play the new Noah Kahan album.",
        "verbatim_text": "Play the new Noah Kahn album.",
    }
    row = wer_mod.score_row(manifest_row, "play the new noah kahn album")
    assert row["utt_id"] == "s01_x"
    # hypothesis matches verbatim_text exactly (mirrors the real Kahan/Kahn
    # correction case in this project's manifest) -> zero verbatim error.
    assert row["verbatim_wer"] == pytest.approx(0.0)
    # script_text says "Kahan", not "Kahn" -> one substitution.
    assert row["script_substitutions"] == 1
    assert row["script_wer"] > 0
