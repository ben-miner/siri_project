"""Whisper baseline: score faster-whisper against the same reference used
for the on-device SFSpeechRecognizer measurement.

Model is large-v3 -- this is the accuracy measurement itself, not the rough
scaffolding transcription 04b_align.py uses to help place utterance
boundaries. Every row logs model_size and compute_type so it's clear on
review what actually produced these numbers.

CRITICAL: vad_filter is explicitly False and is never exposed as a CLI
flag. Voice activity detection identifies and discards low-energy,
aperiodic stretches of audio as "non-speech" -- which is exactly what
creaky phonation looks like acoustically. Enabling VAD would systematically
strip or truncate C_creak material before Whisper ever sees it, biasing
the whole comparison in the one condition this project cares most about.
This mirrors CLAUDE.md's "no audio filtering, normalization, or noise
reduction, ever" non-negotiable -- VAD is a filtering step in the same
sense, just applied to which frames get transcribed rather than to the
waveform itself.

Reuses normalize()/wer()/format_alignment()/load_manifest() from
src/10_wer.py directly (imported, not reimplemented) so this baseline is
scored with the exact same rules as the SFSpeechRecognizer measurement.
Scored only against verbatim_text -- not script_text -- since the point of
this baseline is a same-ground-truth ASR-system comparison, not a
speaker-deviation analysis.

Runs faster-whisper concurrently across files (ThreadPoolExecutor over a
single shared WhisperModel instance -- CTranslate2's num_workers is
designed for exactly this and releases the GIL during inference; verified
safe and ~2x faster than sequential on this machine before using it here
for real). Each Whisper "file" is a fixed ~30s encoder window regardless
of the clip's actual length, so parallelizing across files benefits more
than maximizing threads within any one file -- empirically, num_workers
close to the core count with cpu_threads=1 each outperformed fewer
workers with more threads apiece on this 8-core, no-GPU machine.

Checkpointed like the CreakASR Swift tool: writes each row to --output as
soon as it's scored, and a re-run skips utt_ids already present there, so
an interrupted run resumes instead of restarting. A file that fails to
transcribe is logged and skipped (not retried with backoff -- this is
local CPU computation, not a throttled network service, so a failure is
either a bad file or a real bug, not a transient condition).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
wer_mod = import_module("10_wer")

FIELDNAMES = [
    "utt_id", "session", "pass", "verbatim_text", "hypothesis",
    "model_size", "compute_type", "elapsed_ms",
    "substitutions", "deletions", "insertions", "reference_length", "wer", "alignment",
]


def resolve_device_and_compute_type() -> tuple[str, str]:
    import ctranslate2
    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", "float16"
    return "cpu", "int8"


def already_done_utt_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8", newline="") as f:
        return {row["utt_id"] for row in csv.DictReader(f)}


def transcribe_one(model, wav_path: Path, model_size: str, compute_type: str) -> dict:
    start = time.perf_counter()
    segments, _info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=5,  # library default -- kept, not reduced, for accuracy
        vad_filter=False,  # see module docstring: never True, not a CLI flag
    )
    hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "hypothesis": hypothesis,
        "model_size": model_size,
        "compute_type": compute_type,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def score_against_verbatim(manifest_row: dict, hypothesis: str) -> dict:
    result = wer_mod.wer(wer_mod.normalize(manifest_row["verbatim_text"]), wer_mod.normalize(hypothesis))
    return {
        "substitutions": result["substitutions"],
        "deletions": result["deletions"],
        "insertions": result["insertions"],
        "reference_length": result["reference_length"],
        "wer": round(result["wer"], 4),
        "alignment": wer_mod.format_alignment(result["alignment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/wer_whisper.csv"))
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                         help="concurrent transcription workers (default: cpu count)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N manifest rows (sorted by utt_id)")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"{args.manifest} not found")
    manifest_by_utt_id = wer_mod.load_manifest(args.manifest)
    rows = [manifest_by_utt_id[utt_id] for utt_id in sorted(manifest_by_utt_id)]
    if args.limit is not None:
        rows = rows[:args.limit]

    done = already_done_utt_ids(args.output)
    todo = [row for row in rows if row["utt_id"] not in done]
    print(f"{len(rows)} row(s) selected, {len(done)} already scored, {len(todo)} to do.")
    if not todo:
        print("Nothing to do.")
        return

    device, compute_type = resolve_device_and_compute_type()
    cpu_threads = 1 if device == "cpu" else 0  # empirically fastest split on this machine; see module docstring
    print(f"Loading faster-whisper model '{args.model}' (device={device}, compute_type={compute_type}, "
          f"workers={args.workers}, cpu_threads={cpu_threads})...")
    from faster_whisper import WhisperModel
    model = WhisperModel(
        args.model, device=device, compute_type=compute_type,
        cpu_threads=cpu_threads, num_workers=args.workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    out_f = args.output.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    write_lock = threading.Lock()

    written = 0
    failed = 0

    def process(manifest_row: dict) -> None:
        nonlocal written, failed
        utt_id = manifest_row["utt_id"]
        wav_path = Path(manifest_row["wav_path"])
        try:
            transcription = transcribe_one(model, wav_path, args.model, compute_type)
            score = score_against_verbatim(manifest_row, transcription["hypothesis"])
        except Exception as exc:
            with write_lock:
                failed += 1
            print(f"  FAILED {utt_id}: {exc}", file=sys.stderr)
            return
        row = {
            "utt_id": utt_id,
            "session": manifest_row["session"],
            "pass": manifest_row["pass"],
            "verbatim_text": manifest_row["verbatim_text"],
            **transcription,
            **score,
        }
        with write_lock:
            writer.writerow(row)
            out_f.flush()
            written += 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, row) for row in todo]
        for future in as_completed(futures):
            future.result()  # re-raise any bug that wasn't already caught in process()

    out_f.close()
    print(f"Wrote {written} row(s) to {args.output} ({failed} failed -- see stderr above).")


if __name__ == "__main__":
    main()
