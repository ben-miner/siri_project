"""Build and finalize the verbatim-correction review queue.

Default mode reads results/segmentation.csv, finds every utterance where
match_ratio < 1.0 or flag != "ok", re-transcribes its already-cut clip
(data/split/{utt_id}.wav) fresh with faster-whisper for display, and writes
results/verbatim_review.csv (utt_id, script_text, whisper_text, match_ratio,
verbatim_text) sorted worst match_ratio first. verbatim_text is always
written blank -- it exists for the reviewer to fill in by ear.

--finalize reads the filled-in sheet and writes results/references.csv
(utt_id, script_text, verbatim_text, was_corrected) covering every utterance
in segmentation.csv: reviewed rows use the (possibly hand-corrected)
verbatim_text, blank verbatim_text falls back to script_text, and rows that
never needed review also fall back to script_text. This is the project's
reference transcript -- corrected by ear from the read script, never from
ASR output.

Two hard rules enforced in code, not just by convention:
1. verbatim_text is NEVER auto-populated from whisper_text -- whisper_text
   is for locating the discrepancy only. --finalize prints a loud warning
   if verbatim_text is byte-identical to whisper_text for more than 20% of
   rows that were actually filled in (a sign of copy-pasting instead of
   transcribing by ear).
2. --finalize refuses to run if any row has both a blank verbatim_text and
   match_ratio < 0.85 -- that combination means a badly-matched item was
   never actually reviewed.

Re-running the default mode preserves any verbatim_text already filled in
for a given utt_id in an existing results/verbatim_review.csv, so hand
review work already done is never silently clobbered by a re-run.
"""

import argparse
import csv
import sys
from pathlib import Path

FINALIZE_MATCH_RATIO_GATE = 0.85
COPY_PASTE_WARNING_FRACTION = 0.20


def make_utt_id(session: str, pass_name: str, item_id: str) -> str:
    return f"{session}_{pass_name}_{item_id}"


def load_segmentation(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "session": row["session"],
            "pass": row["pass"],
            "item_id": row["item_id"],
            "position": int(row["position"]),
            "match_ratio": float(row["match_ratio"]),
            "flag": row["flag"],
        }
        for row in rows
    ]


def load_items(items_csv: Path) -> dict:
    with items_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["item_id"]: row["text"] for row in rows}


def load_review_sheet(review_csv: Path) -> list[dict]:
    with review_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "utt_id": row["utt_id"],
            "script_text": row["script_text"],
            "whisper_text": row["whisper_text"],
            "match_ratio": float(row["match_ratio"]),
            "verbatim_text": row["verbatim_text"],
        }
        for row in rows
    ]


def needs_review(row: dict) -> bool:
    return row["match_ratio"] < 1.0 or row["flag"] != "ok"


def build_review_rows(
    segmentation_rows: list[dict],
    items: dict,
    whisper_texts: dict,
    existing_verbatim: dict | None = None,
) -> list[dict]:
    existing_verbatim = existing_verbatim or {}
    flagged = [row for row in segmentation_rows if needs_review(row)]

    review_rows = []
    for row in flagged:
        utt_id = make_utt_id(row["session"], row["pass"], row["item_id"])
        if row["item_id"] not in items:
            raise ValueError(f"item_id '{row['item_id']}' not found in items.csv")
        review_rows.append({
            "utt_id": utt_id,
            "script_text": items[row["item_id"]],
            "whisper_text": whisper_texts.get(utt_id, ""),
            "match_ratio": row["match_ratio"],
            "verbatim_text": existing_verbatim.get(utt_id, ""),
            "_position_key": (row["pass"], row["session"], row["position"]),
        })

    review_rows.sort(key=lambda r: (r["match_ratio"], r["_position_key"]))
    for row in review_rows:
        del row["_position_key"]
    return review_rows


def print_breakdown(flagged_segmentation_rows: list[dict]) -> None:
    by_pass: dict[str, int] = {}
    for row in flagged_segmentation_rows:
        by_pass[row["pass"]] = by_pass.get(row["pass"], 0) + 1
    print(f"{len(flagged_segmentation_rows)} item(s) need review:")
    for pass_name in sorted(by_pass):
        print(f"  {pass_name}: {by_pass[pass_name]}")


BATCH_GAP_SEC = 0.5
MAX_BATCH_DURATION_SEC = 25.0  # stay safely under Whisper's fixed 30s encoder window
WHISPER_SAMPLE_RATE = 16000


def pack_into_batches(
    clip_durations: list[tuple[str, float]], max_batch_duration: float, gap: float
) -> list[list[tuple[str, float]]]:
    """Greedily pack (utt_id, duration) pairs so each batch's total duration
    (including inter-clip gaps) stays under max_batch_duration. This exists
    because Whisper's encoder always processes a fixed ~30s window regardless
    of actual audio length -- transcribing 136 short clips one at a time each
    pays that full fixed cost, so packing several clips into one window turns
    N encoder passes into N/batch_size passes for a large real speedup."""
    batches: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    current_total = 0.0
    for utt_id, duration in clip_durations:
        added_gap = gap if current else 0.0
        if current and current_total + added_gap + duration > max_batch_duration:
            batches.append(current)
            current = []
            current_total = 0.0
            added_gap = 0.0
        current.append((utt_id, duration))
        current_total += added_gap + duration
    if current:
        batches.append(current)
    return batches


def compute_batch_offsets(
    batch: list[tuple[str, float]], gap: float
) -> list[tuple[str, float, float]]:
    """Where each clip in a packed batch lands within the concatenated buffer."""
    offsets = []
    t = 0.0
    for i, (utt_id, duration) in enumerate(batch):
        if i:
            t += gap
        offsets.append((utt_id, t, t + duration))
        t += duration
    return offsets


def assign_words_to_clips(
    words: list[tuple[float, float, str]], offsets: list[tuple[str, float, float]]
) -> dict[str, list[str]]:
    """Attribute each transcribed word to the clip whose window contains its
    midpoint; a word landing in an inter-clip silence gap (shouldn't happen
    given the gap size, but not impossible at a boundary) goes to the nearest
    clip instead of being dropped."""
    result: dict[str, list[str]] = {utt_id: [] for utt_id, _, _ in offsets}
    for start, end, text in words:
        mid = (start + end) / 2
        match = next((utt_id for utt_id, o_start, o_end in offsets if o_start <= mid < o_end), None)
        if match is None:
            def distance(offset):
                _, o_start, o_end = offset
                if mid < o_start:
                    return o_start - mid
                if mid >= o_end:
                    return mid - o_end
                return 0.0
            match = min(offsets, key=distance)[0]
        result[match].append(text)
    return result


def transcribe_batch(
    model, clip_arrays: dict[str, "object"], batch: list[tuple[str, float]]
) -> dict[str, str]:
    """Concatenate one packed batch's already-decoded 16kHz arrays, transcribe
    it in a single encoder pass, and split the words back out per clip."""
    import numpy as np

    gap_samples = np.zeros(int(BATCH_GAP_SEC * WHISPER_SAMPLE_RATE), dtype="float32")
    parts = []
    for i, (utt_id, _duration) in enumerate(batch):
        if i:
            parts.append(gap_samples)
        parts.append(clip_arrays[utt_id])
    combo = np.concatenate(parts)

    offsets = compute_batch_offsets(batch, BATCH_GAP_SEC)
    segments, _info = model.transcribe(combo, language="en", word_timestamps=True)
    words = [(w.start, w.end, w.word.strip()) for seg in segments for w in seg.words]

    by_clip = assign_words_to_clips(words, offsets)
    return {utt_id: " ".join(by_clip[utt_id]).strip() for utt_id, _ in batch}


def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8", cpu_threads=8)
        return model, model_size
    except Exception as exc:
        print(
            f"WARNING: failed to load whisper model '{model_size}' ({exc}); "
            f"falling back to medium/cpu/int8",
            file=sys.stderr,
        )
        model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=8)
        return model, "medium"


def check_finalize_gate(review_rows: list[dict]) -> list[dict]:
    return [
        row for row in review_rows
        if row["verbatim_text"].strip() == "" and row["match_ratio"] < FINALIZE_MATCH_RATIO_GATE
    ]


def check_copy_paste_warning(review_rows: list[dict]) -> tuple[int, int, float]:
    corrected = [row for row in review_rows if row["verbatim_text"].strip() != ""]
    identical = [row for row in corrected if row["verbatim_text"] == row["whisper_text"]]
    fraction = len(identical) / len(corrected) if corrected else 0.0
    return len(corrected), len(identical), fraction


def build_references(
    segmentation_rows: list[dict], items: dict, review_by_utt_id: dict
) -> list[dict]:
    references = []
    for row in segmentation_rows:
        utt_id = make_utt_id(row["session"], row["pass"], row["item_id"])
        if row["item_id"] not in items:
            raise ValueError(f"item_id '{row['item_id']}' not found in items.csv")
        script_text = items[row["item_id"]]

        reviewed = review_by_utt_id.get(utt_id)
        if reviewed is not None and reviewed["verbatim_text"].strip() != "":
            verbatim_text = reviewed["verbatim_text"]
        else:
            verbatim_text = script_text

        references.append({
            "utt_id": utt_id,
            "script_text": script_text,
            "verbatim_text": verbatim_text,
            "was_corrected": verbatim_text != script_text,
        })
    return references


def run_build(args) -> None:
    segmentation_rows = load_segmentation(args.segmentation_csv)
    items = load_items(args.items_csv)

    flagged = [row for row in segmentation_rows if needs_review(row)]

    existing_verbatim = {}
    if args.review_out.exists():
        existing = load_review_sheet(args.review_out)
        existing_verbatim = {
            row["utt_id"]: row["verbatim_text"]
            for row in existing if row["verbatim_text"].strip() != ""
        }

    whisper_model = args.whisper_model
    if whisper_model is None:
        # large-v3-turbo, not phonpipe's default large-v3: this script transcribes
        # many short (0.5-12s) review clips rather than long continuous audio, and
        # turbo's decoder speedup plus batching (see pack_into_batches) are what
        # actually matter for that shape of workload -- verified empirically to
        # produce clean transcripts, not just assumed.
        whisper_model = "large-v3-turbo"

    whisper_texts = {}
    if flagged:
        from faster_whisper.audio import decode_audio

        clip_infos = []  # (utt_id, path)
        missing_clips = []
        for row in flagged:
            utt_id = make_utt_id(row["session"], row["pass"], row["item_id"])
            clip_path = args.split_dir / f"{utt_id}.wav"
            if not clip_path.exists():
                missing_clips.append(str(clip_path))
                continue
            clip_infos.append((utt_id, clip_path))
        if missing_clips:
            raise FileNotFoundError(
                "Missing clip(s) for flagged row(s) -- run 04c_cut.py first:\n  "
                + "\n  ".join(missing_clips)
            )

        clip_arrays = {
            utt_id: decode_audio(str(path), sampling_rate=WHISPER_SAMPLE_RATE)
            for utt_id, path in clip_infos
        }
        clip_durations = [
            (utt_id, len(arr) / WHISPER_SAMPLE_RATE) for utt_id, arr in clip_arrays.items()
        ]
        batches = pack_into_batches(clip_durations, MAX_BATCH_DURATION_SEC, BATCH_GAP_SEC)

        model, actual_model_size = load_whisper_model(whisper_model)
        print(
            f"Using faster-whisper model: {actual_model_size} "
            f"({len(clip_infos)} clips packed into {len(batches)} batch(es))",
            flush=True,
        )

        n_done = 0
        progress_every = max(1, len(batches) // 15)
        for i, batch in enumerate(batches, start=1):
            whisper_texts.update(transcribe_batch(model, clip_arrays, batch))
            n_done += len(batch)
            if i % progress_every == 0 or i == len(batches):
                print(f"PROGRESS batch {i}/{len(batches)} ({n_done}/{len(clip_infos)} clips)", flush=True)

    review_rows = build_review_rows(segmentation_rows, items, whisper_texts, existing_verbatim)

    print_breakdown(flagged)

    n_carried = sum(1 for r in review_rows if r["utt_id"] in existing_verbatim)

    args.review_out.parent.mkdir(parents=True, exist_ok=True)
    with args.review_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utt_id", "script_text", "whisper_text", "match_ratio", "verbatim_text"]
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(
        f"Wrote {len(review_rows)} row(s) to {args.review_out}"
        + (f" ({n_carried} previously-filled verbatim_text value(s) carried over)" if n_carried else "")
    )


def run_finalize(args) -> None:
    if not args.review_out.exists():
        sys.exit(f"{args.review_out} not found. Run 04d_verbatim.py (without --finalize) first.")

    review_rows = load_review_sheet(args.review_out)

    gate_failures = check_finalize_gate(review_rows)
    if gate_failures:
        print(
            f"Refusing to finalize: {len(gate_failures)} row(s) have a blank verbatim_text "
            f"and match_ratio < {FINALIZE_MATCH_RATIO_GATE}:"
        )
        for row in gate_failures:
            print(f"  {row['utt_id']}: match_ratio={row['match_ratio']:.4f}")
        sys.exit(1)

    n_corrected, n_identical, fraction = check_copy_paste_warning(review_rows)
    if fraction > COPY_PASTE_WARNING_FRACTION:
        print(
            f"WARNING: verbatim_text is byte-identical to whisper_text for "
            f"{n_identical}/{n_corrected} ({fraction:.1%}) of filled-in rows. "
            f"You may have copy-pasted whisper_text instead of transcribing by ear."
        )

    segmentation_rows = load_segmentation(args.segmentation_csv)
    items = load_items(args.items_csv)
    review_by_utt_id = {row["utt_id"]: row for row in review_rows}

    references = build_references(segmentation_rows, items, review_by_utt_id)

    args.references_out.parent.mkdir(parents=True, exist_ok=True)
    with args.references_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "script_text", "verbatim_text", "was_corrected"])
        writer.writeheader()
        writer.writerows(references)

    n_total_corrected = sum(1 for r in references if r["was_corrected"])
    print(
        f"Wrote {len(references)} row(s) to {args.references_out} "
        f"({n_total_corrected} corrected from script_text)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-csv", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument("--items-csv", type=Path, default=Path("stimuli/items.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--review-out", type=Path, default=Path("results/verbatim_review.csv"))
    parser.add_argument("--references-out", type=Path, default=Path("results/references.csv"))
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        help="faster-whisper model size (default: large-v3-turbo -- see run_build for why)",
    )
    parser.add_argument("--finalize", action="store_true", help="Finalize references.csv from the filled review sheet")
    args = parser.parse_args()

    if args.finalize:
        run_finalize(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
