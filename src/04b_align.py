"""Word-align Whisper transcripts of each sentence block to the expected script.

For each block wav in data/blocks/ this script:

  1. Transcribes with faster-whisper (word_timestamps=True, language="en"),
     using the same model size phonpipe defaults to (PipelineConfig.whisper_model).
  2. Builds the expected word sequence for that pass by concatenating the
     100 sentences from stimuli/orders.csv (joined to stimuli/items.csv on
     item_id) in read order.
  3. Aligns Whisper's word stream to the expected word stream with
     difflib.SequenceMatcher (autojunk disabled -- with ~800 words per
     block, autojunk's "popular element" heuristic would start treating
     common words like "the" as junk and ignoring them, which is exactly
     wrong for this task). Matching blocks give each item_id's start/end
     time as the min/max timestamp of its matched Whisper words.
  4. Writes results/segmentation.csv (session, pass, item_id, position,
     start_sec, end_sec, n_whisper_words_matched, match_ratio, flag).
  5. Reports Whisper-side insertions (words that don't map to any expected
     word at all) -- these are flubs and retakes, not transcription noise
     within a sentence, so they're kept separate from per-item match stats.

This script only produces the boundary table -- it does not cut any audio.

A low match_ratio on the creak pass is expected, not necessarily a bug:
Whisper's accuracy is at its worst on sustained creaky phonation. Flagged
items are reported worst-first (match_ratio ascending) for manual review.
"""

import argparse
import csv
import string
import sys
from dataclasses import dataclass
from pathlib import Path

EXPECTED_ITEM_COUNT = 100
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.8
_PUNCT = string.punctuation


def normalize_word(word: str) -> str:
    return word.strip().lower().strip(_PUNCT)


def normalize_sentence(text: str) -> list[str]:
    return [w for w in (normalize_word(tok) for tok in text.split()) if w]


def load_items(items_csv: Path) -> dict[str, str]:
    with items_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = {row["item_id"]: row["text"] for row in rows}
    if len(items) != len(rows):
        raise ValueError(f"{items_csv}: duplicate item_id values")
    return items


def load_orders(orders_csv: Path) -> dict[str, list[tuple[int, str]]]:
    with orders_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    orders: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        orders.setdefault(row["pass"], []).append((int(row["position"]), row["item_id"]))

    for pass_name, entries in orders.items():
        entries.sort(key=lambda pair: pair[0])
        positions = [p for p, _ in entries]
        if positions != list(range(1, EXPECTED_ITEM_COUNT + 1)):
            raise ValueError(
                f"{orders_csv}: pass '{pass_name}' does not have positions "
                f"1..{EXPECTED_ITEM_COUNT} exactly once (got {len(positions)} rows)"
            )
    return orders


def build_expected_sequence(
    pass_name: str,
    items: dict[str, str],
    orders: dict[str, list[tuple[int, str]]],
) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    if pass_name not in orders:
        raise ValueError(f"pass '{pass_name}' not found in orders (have: {sorted(orders)})")

    expected_tokens: list[str] = []
    expected_item_ids: list[str] = []
    item_word_count: dict[str, int] = {}
    item_position: dict[str, int] = {}

    for position, item_id in sorted(orders[pass_name], key=lambda pair: pair[0]):
        if item_id not in items:
            raise ValueError(f"item_id '{item_id}' in orders but not in items.csv")
        words = normalize_sentence(items[item_id])
        if not words:
            raise ValueError(f"item_id '{item_id}' normalized to zero words")
        expected_tokens.extend(words)
        expected_item_ids.extend([item_id] * len(words))
        item_word_count[item_id] = len(words)
        item_position[item_id] = position

    return expected_tokens, expected_item_ids, item_word_count, item_position


def find_block_wavs(blocks_dir: Path) -> list[Path]:
    return sorted(blocks_dir.glob("*.wav"))


def parse_block_stem(stem: str) -> tuple[str, str]:
    suffix = "_block"
    if not stem.endswith(suffix):
        raise ValueError(f"block filename '{stem}' does not end with '{suffix}'")
    take = stem[: -len(suffix)]
    if "_" not in take:
        raise ValueError(f"take '{take}' has no underscore to split session from pass")
    session, pass_file_label = take.split("_", 1)
    if not pass_file_label.startswith("pass"):
        raise ValueError(f"pass label '{pass_file_label}' does not start with 'pass'")
    canonical_pass = pass_file_label[len("pass"):]
    return session, canonical_pass


@dataclass
class ItemMatch:
    matched_count: int
    start_sec: float | None
    end_sec: float | None


@dataclass
class Insertion:
    start_sec: float
    end_sec: float
    text: str


def align_words(
    expected_tokens: list[str],
    expected_item_ids: list[str],
    whisper_tokens: list[str],
    whisper_times: list[tuple[float, float]],
    whisper_raw_words: list[str],
) -> tuple[dict[str, ItemMatch], list[Insertion]]:
    import difflib

    matcher = difflib.SequenceMatcher(None, expected_tokens, whisper_tokens, autojunk=False)

    matched_times: dict[str, list[tuple[float, float]]] = {}
    matched_count: dict[str, int] = {}
    insertions: list[Insertion] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                item_id = expected_item_ids[i1 + k]
                matched_times.setdefault(item_id, []).append(whisper_times[j1 + k])
                matched_count[item_id] = matched_count.get(item_id, 0) + 1
        elif tag == "insert":
            insertions.append(Insertion(
                start_sec=whisper_times[j1][0],
                end_sec=whisper_times[j2 - 1][1],
                text=" ".join(whisper_raw_words[j1:j2]),
            ))

    item_matches: dict[str, ItemMatch] = {}
    for item_id in set(expected_item_ids):
        times = matched_times.get(item_id, [])
        count = matched_count.get(item_id, 0)
        if times:
            start = min(t[0] for t in times)
            end = max(t[1] for t in times)
        else:
            start = end = None
        item_matches[item_id] = ItemMatch(matched_count=count, start_sec=start, end_sec=end)

    return item_matches, insertions


def classify_flag(matched_count: int, expected_count: int, threshold: float) -> str:
    if matched_count == 0:
        return "unmatched"
    if matched_count / expected_count < threshold:
        return "low_confidence"
    return "ok"


def build_segmentation_rows(
    session: str,
    canonical_pass: str,
    item_position: dict[str, int],
    item_word_count: dict[str, int],
    item_matches: dict[str, ItemMatch],
    threshold: float,
) -> list[dict]:
    rows = []
    for item_id, position in sorted(item_position.items(), key=lambda pair: pair[1]):
        match = item_matches[item_id]
        expected_count = item_word_count[item_id]
        match_ratio = round(match.matched_count / expected_count, 4)
        rows.append({
            "session": session,
            "pass": canonical_pass,
            "item_id": item_id,
            "position": position,
            "start_sec": match.start_sec,
            "end_sec": match.end_sec,
            "n_whisper_words_matched": match.matched_count,
            "match_ratio": match_ratio,
            "flag": classify_flag(match.matched_count, expected_count, threshold),
        })
    return rows


def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        return model, model_size
    except Exception as exc:
        print(
            f"WARNING: failed to load whisper model '{model_size}' ({exc}); "
            f"falling back to medium/cpu/int8",
            file=sys.stderr,
        )
        model = WhisperModel("medium", device="cpu", compute_type="int8")
        return model, "medium"


def transcribe_words(model, wav_path: Path) -> tuple[list[str], list[tuple[float, float]], list[str]]:
    segments, _info = model.transcribe(str(wav_path), language="en", word_timestamps=True)

    tokens: list[str] = []
    times: list[tuple[float, float]] = []
    raw_words: list[str] = []
    for segment in segments:
        for w in segment.words:
            raw = w.word.strip()
            norm = normalize_word(raw)
            if not norm:
                continue
            tokens.append(norm)
            times.append((w.start, w.end))
            raw_words.append(raw)
    return tokens, times, raw_words


def print_insertions_report(take: str, insertions: list[Insertion]) -> None:
    if not insertions:
        return
    print(f"\n=== Insertions (unmatched Whisper words) -- {take} ===")
    for ins in sorted(insertions, key=lambda i: i.start_sec):
        print(f"  {ins.start_sec:7.2f}-{ins.end_sec:7.2f}s: \"{ins.text}\"")


def print_flagged_summary(rows: list[dict], threshold: float) -> None:
    flagged = [r for r in rows if r["flag"] != "ok"]
    n_low = sum(1 for r in flagged if r["flag"] == "low_confidence")
    n_unmatched = sum(1 for r in flagged if r["flag"] == "unmatched")
    print(
        f"\n=== Flagged items ({len(flagged)}/{len(rows)} total; "
        f"{n_low} low_confidence, {n_unmatched} unmatched; "
        f"threshold={threshold}) -- worst match_ratio first ==="
    )
    for row in sorted(flagged, key=lambda r: r["match_ratio"]):
        print(
            f"  {row['session']} {row['pass']} {row['item_id']} "
            f"(pos {row['position']:3d}): match_ratio={row['match_ratio']:.4f} "
            f"matched={row['n_whisper_words_matched']} flag={row['flag']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-dir", type=Path, default=Path("data/blocks"))
    parser.add_argument("--orders-csv", type=Path, default=Path("stimuli/orders.csv"))
    parser.add_argument("--items-csv", type=Path, default=Path("stimuli/items.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument(
        "--low-confidence-threshold", type=float, default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help=f"match_ratio below this is flagged low_confidence (default: {DEFAULT_LOW_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        help="faster-whisper model size (default: phonpipe's PipelineConfig.whisper_model)",
    )
    args = parser.parse_args()

    if args.whisper_model is None:
        from phonpipe.config import PipelineConfig
        args.whisper_model = PipelineConfig().whisper_model

    block_paths = find_block_wavs(args.blocks_dir)
    if not block_paths:
        sys.exit(f"No block wavs found under {args.blocks_dir}")

    items = load_items(args.items_csv)
    orders = load_orders(args.orders_csv)

    model, actual_model_size = load_whisper_model(args.whisper_model)
    print(f"Using faster-whisper model: {actual_model_size}")

    all_rows: list[dict] = []
    for wav_path in block_paths:
        session, canonical_pass = parse_block_stem(wav_path.stem)
        take = wav_path.stem[: -len("_block")]

        expected_tokens, expected_item_ids, item_word_count, item_position = build_expected_sequence(
            canonical_pass, items, orders
        )

        whisper_tokens, whisper_times, whisper_raw_words = transcribe_words(model, wav_path)

        item_matches, insertions = align_words(
            expected_tokens, expected_item_ids, whisper_tokens, whisper_times, whisper_raw_words
        )

        rows = build_segmentation_rows(
            session, canonical_pass, item_position, item_word_count, item_matches,
            args.low_confidence_threshold,
        )
        all_rows.extend(rows)

        n_flagged = sum(1 for r in rows if r["flag"] != "ok")
        print(f"{take}: {len(whisper_tokens)} whisper words, {n_flagged}/{len(rows)} items flagged")
        print_insertions_report(take, insertions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session", "pass", "item_id", "position", "start_sec", "end_sec",
                "n_whisper_words_matched", "match_ratio", "flag",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print_flagged_summary(all_rows, args.low_confidence_threshold)
    print(f"\nWrote {len(all_rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
