"""Score ASR hypotheses against reference transcripts.

Two importable pieces:
  - normalize(text) -> list[str]: text normalization, so reference and
    hypothesis text become directly comparable token lists regardless of
    orthographic convention (numerals vs spelled-out words, "$5" vs "five
    dollars", punctuation, casing).
  - wer(ref_tokens, hyp_tokens) -> dict: a from-scratch Levenshtein
    alignment between two token lists, with error counts and the full
    aligned pair list.

Each normalization rule is its own function so it's independently
testable. normalize() composes them, but NOT in the order they're most
naturally described in: symbols/numerals must be expanded to words
*before* generic punctuation stripping runs, or strip_punctuation would
just delete "$"/"%"/"deg" outright instead of converting them -- silently
losing exactly the information this normalization exists to preserve.

CLI: joins results/hypotheses.csv to results/manifest.csv on utt_id and
writes results/scored.csv with error counts, WER against both
verbatim_text and script_text, and each alignment dumped as a string.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# -- number/symbol expansion -----------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"), (100, "hundred")]


def number_to_words(n: int) -> str:
    """Spells out a non-negative integer in standard American English
    words, no "and" (e.g. 143 -> "one hundred forty-three", not "one
    hundred and forty-three") -- matches how numbers are conventionally
    read aloud and transcribed. Two-digit remainders are hyphenated (45
    -> "forty-five"); strip_punctuation later treats hyphens as word
    separators, so this becomes two tokens either way, same as a spoken
    two-digit number is two words.
    """
    if n < 0:
        raise ValueError(f"number_to_words only supports non-negative integers, got {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        word = _TENS[tens]
        return f"{word}-{_ONES[ones]}" if ones else word
    for value, name in _SCALES:
        if n >= value:
            count, remainder = divmod(n, value)
            head = f"{number_to_words(count)} {name}"
            return f"{head} {number_to_words(remainder)}" if remainder else head
    raise AssertionError("unreachable: _SCALES bottoms out at 100, and n >= 100 here")


_NUMERAL_RE = re.compile(r"\d+")


def expand_numerals(text: str) -> str:
    """Converts any remaining bare digit sequence to spelled-out words.
    Meant to run after expand_symbols, which already handles digits
    adjacent to $/%/deg -- this only sees standalone numbers (e.g. "12"
    in a hypothesis with no unit attached)."""
    return _NUMERAL_RE.sub(lambda m: number_to_words(int(m.group(0))), text)


_CURRENCY_RE = re.compile(r"\$(\d+)")
_PERCENT_RE = re.compile(r"(\d+)%")
_DEGREE_RE = re.compile(r"(\d+)°")


def expand_symbols(text: str) -> str:
    """Rewrites currency/percent/degree symbols into words, in the order
    a person actually says them: "$43" -> "forty-three dollars" (number
    then unit) even though "$" is written before the number, because
    that's the spoken order and how this project's reference script
    always writes it out ("forty-three dollars", never "$43"). Percent
    and degree symbols already follow their number in both written and
    spoken order, so no reordering is needed there. Any occurrence of
    these symbols NOT adjacent to a digit still becomes a word rather
    than being silently dropped by punctuation stripping later.

    Every substitution is padded with spaces so it can't fuse with an
    adjacent word character that had no whitespace of its own -- real
    ASR output in this project writes compact unit abbreviations with no
    separating space ("80°F"), and naively substituting just "degrees"
    for "°" there would produce the single garbage token "degreesF"
    instead of "degrees" and "F" as two separate, individually
    comparable tokens. The extra spacing this introduces is collapsed
    back down before returning, so this function's own output never has
    doubled internal spaces for callers/tests that use it standalone.
    """
    text = _CURRENCY_RE.sub(lambda m: f" {number_to_words(int(m.group(1)))} dollars ", text)
    text = _PERCENT_RE.sub(lambda m: f" {number_to_words(int(m.group(1)))} percent ", text)
    text = _DEGREE_RE.sub(lambda m: f" {number_to_words(int(m.group(1)))} degrees ", text)
    text = text.replace("$", " dollars ")
    text = text.replace("%", " percent ")
    text = text.replace("°", " degrees ")
    return re.sub(r" +", " ", text).strip()


# -- the other three rules -----------------------------------------------------------

def lowercase(text: str) -> str:
    return text.lower()


_SEPARATOR_PUNCTUATION = "-:"


def strip_punctuation(text: str) -> str:
    """Removes punctuation except an apostrophe with a word character on
    both sides (so "don't"/"it's" stay one token; a leading or trailing
    apostrophe does not). Hyphens and colons are replaced with a space
    rather than deleted outright, because both show up in this project's
    real data as a separator between two words with no whitespace of
    their own: a hyphenated compound number in the reference script
    ("forty-five"), and a compact clock time in real ASR hypotheses
    ("6:45", for the "six forty-five" stimulus item) -- expand_numerals
    converts "6" and "45" independently, leaving the colon literally in
    place between them, so without this the two number-words would fuse
    into one garbage token ("sixforty") instead of tokenizing the same
    way the reference does.
    """
    for ch in _SEPARATOR_PUNCTUATION:
        text = text.replace(ch, " ")
    kept = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch.isspace():
            kept.append(ch)
        elif ch == "'" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
            kept.append(ch)
    return "".join(kept)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return text.split()


def normalize(text: str) -> list[str]:
    text = lowercase(text)
    text = expand_symbols(text)
    text = expand_numerals(text)
    text = strip_punctuation(text)
    text = collapse_whitespace(text)
    return tokenize(text)


# -- WER -----------------------------------------------------------

def _wer_ratio(substitutions: int, deletions: int, insertions: int, reference_length: int) -> float:
    if reference_length == 0:
        return 0.0 if (substitutions == 0 and insertions == 0) else float("inf")
    return (substitutions + deletions + insertions) / reference_length


def wer(ref_tokens: list[str], hyp_tokens: list[str]) -> dict:
    """Token-level Levenshtein alignment between ref_tokens and
    hyp_tokens, written from scratch: a standard O(n*m) edit-distance
    dynamic program (match/substitution cost 0/1, deletion cost 1,
    insertion cost 1), backtraced into an aligned pair list.

    Returns a dict with substitutions, deletions, insertions,
    reference_length, wer (= (subs+dels+ins)/reference_length, defined as
    0.0 for an empty reference with no errors and inf for an empty
    reference with any insertions), and alignment: a list of
    (ref_token_or_None, hyp_token_or_None) pairs in reference order --
    None on the ref side is an insertion, None on the hyp side is a
    deletion.
    """
    n, m = len(ref_tokens), len(hyp_tokens)

    # cost[i][j] = min edit cost aligning ref_tokens[:i] to hyp_tokens[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = cost[i - 1][j - 1] + (0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1)
            delete_cost = cost[i - 1][j] + 1
            insert_cost = cost[i][j - 1] + 1
            cost[i][j] = min(match_cost, delete_cost, insert_cost)

    alignment: list[tuple[str | None, str | None]] = []
    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (
            0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
        ):
            if ref_tokens[i - 1] != hyp_tokens[j - 1]:
                substitutions += 1
            alignment.append((ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            deletions += 1
            alignment.append((ref_tokens[i - 1], None))
            i -= 1
        else:
            insertions += 1
            alignment.append((None, hyp_tokens[j - 1]))
            j -= 1
    alignment.reverse()

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_length": n,
        "wer": _wer_ratio(substitutions, deletions, insertions, n),
        "alignment": alignment,
    }


def format_alignment(alignment: list[tuple[str | None, str | None]]) -> str:
    """Dumps an alignment as a single readable string: a bare word for a
    match, [-word] for a deletion, [+word] for an insertion, [ref->hyp]
    for a substitution."""
    parts = []
    for ref, hyp in alignment:
        if ref == hyp:
            parts.append(ref)
        elif ref is None:
            parts.append(f"[+{hyp}]")
        elif hyp is None:
            parts.append(f"[-{ref}]")
        else:
            parts.append(f"[{ref}->{hyp}]")
    return " ".join(parts)


# -- CLI -----------------------------------------------------------

def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"{path} not found")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_utt_id: dict[str, dict] = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in by_utt_id:
            sys.exit(f"duplicate utt_id '{utt_id}' in {path} -- refusing to proceed")
        by_utt_id[utt_id] = row
    return by_utt_id


def load_hypotheses(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in seen:
            sys.exit(f"duplicate utt_id '{utt_id}' in {path} -- refusing to proceed")
        seen.add(utt_id)
    return rows


SCORED_FIELDNAMES = [
    "utt_id", "session", "pass", "was_corrected", "hypothesis",
    "script_text", "verbatim_text",
    "verbatim_substitutions", "verbatim_deletions", "verbatim_insertions",
    "verbatim_reference_length", "verbatim_wer", "verbatim_alignment",
    "script_substitutions", "script_deletions", "script_insertions",
    "script_reference_length", "script_wer", "script_alignment",
]


def score_row(manifest_row: dict, hypothesis: str) -> dict:
    hyp_tokens = normalize(hypothesis)
    verbatim_result = wer(normalize(manifest_row["verbatim_text"]), hyp_tokens)
    script_result = wer(normalize(manifest_row["script_text"]), hyp_tokens)
    return {
        "utt_id": manifest_row["utt_id"],
        "session": manifest_row["session"],
        "pass": manifest_row["pass"],
        "was_corrected": manifest_row["was_corrected"],
        "hypothesis": hypothesis,
        "script_text": manifest_row["script_text"],
        "verbatim_text": manifest_row["verbatim_text"],
        "verbatim_substitutions": verbatim_result["substitutions"],
        "verbatim_deletions": verbatim_result["deletions"],
        "verbatim_insertions": verbatim_result["insertions"],
        "verbatim_reference_length": verbatim_result["reference_length"],
        "verbatim_wer": round(verbatim_result["wer"], 4),
        "verbatim_alignment": format_alignment(verbatim_result["alignment"]),
        "script_substitutions": script_result["substitutions"],
        "script_deletions": script_result["deletions"],
        "script_insertions": script_result["insertions"],
        "script_reference_length": script_result["reference_length"],
        "script_wer": round(script_result["wer"], 4),
        "script_alignment": format_alignment(script_result["alignment"]),
    }


def write_scored_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, default=Path("results/hypotheses.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/scored.csv"))
    args = parser.parse_args()

    manifest_by_utt_id = load_manifest(args.manifest)
    hypothesis_rows = load_hypotheses(args.hypotheses)

    scored_rows = []
    for row in hypothesis_rows:
        utt_id = row["utt_id"]
        manifest_row = manifest_by_utt_id.get(utt_id)
        if manifest_row is None:
            sys.exit(f"utt_id '{utt_id}' in {args.hypotheses} has no match in {args.manifest} "
                      f"-- refusing to proceed")
        scored_rows.append(score_row(manifest_row, row["hypothesis"]))

    write_scored_csv(scored_rows, args.output)

    mean_verbatim_wer = (
        sum(r["verbatim_wer"] for r in scored_rows) / len(scored_rows) if scored_rows else float("nan")
    )
    mean_script_wer = (
        sum(r["script_wer"] for r in scored_rows) / len(scored_rows) if scored_rows else float("nan")
    )
    print(f"Wrote {len(scored_rows)}/{len(manifest_by_utt_id)} scored utterances to {args.output} "
          f"(mean verbatim WER {mean_verbatim_wer:.3f}, mean script WER {mean_script_wer:.3f}).")


if __name__ == "__main__":
    main()
