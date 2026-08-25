"""Build the analysis manifest by joining split clips to items and references.

Scans data/split/ for {session}_{pass}_{item_id}.wav files, parses each
filename, and joins to stimuli/items.csv (on item_id) and
results/references.csv (on utt_id == filename stem) to write
results/manifest.csv: utt_id, session, pass, item_id, item_type,
script_text, verbatim_text, was_corrected, wav_path, duration_sec.

{pass} is not a fixed-width token (e.g. "A_modal" contains its own
underscore), so filenames are parsed by anchoring on item_id's fixed
[C|D|F]DDD pattern at the end and session's no-underscore token at the
start, taking whatever falls between as the pass name -- splitting on "_"
by position would misparse "A_modal" (see 04b_align.py's parse_block_stem
and 04d_verbatim.py's print_breakdown for the same lesson learned earlier).

Validates BEFORE writing anything: every filename must parse, every item_id
must exist in items.csv, every utt_id must exist in references.csv, and
items.csv/references.csv must agree on script_text for that item_id (a
cross-source consistency check -- CLAUDE.md's fail-loudly-on-data-mismatches
rule applies project-wide, not just to the checks explicitly listed here).
Any failure produces a full report and no file is written.
"""

import argparse
import csv
import re
import sys
import wave
from pathlib import Path

FILENAME_PATTERN = re.compile(r"^(?P<session>[^_]+)_(?P<pass_name>.+)_(?P<item_id>[CDF]\d{3})$")


def find_split_wavs(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.wav"))


def parse_filename(stem: str) -> tuple[str, str, str]:
    m = FILENAME_PATTERN.match(stem)
    if not m:
        raise ValueError(
            f"'{stem}.wav' does not match the {{session}}_{{pass}}_{{item_id}} pattern"
        )
    return m.group("session"), m.group("pass_name"), m.group("item_id")


def load_items(items_csv: Path) -> dict:
    with items_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        row["item_id"]: {"item_type": row["item_type"], "text": row["text"]}
        for row in rows
    }


def load_references(references_csv: Path) -> dict:
    with references_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        row["utt_id"]: {
            "script_text": row["script_text"],
            "verbatim_text": row["verbatim_text"],
            "was_corrected": row["was_corrected"] == "True",
        }
        for row in rows
    }


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def build_manifest_rows(
    wav_paths: list[Path], items: dict, references: dict, durations: dict
) -> tuple[list[dict], list[str]]:
    errors = []
    rows = []
    for path in wav_paths:
        stem = path.stem
        try:
            session, pass_name, item_id = parse_filename(stem)
        except ValueError as e:
            errors.append(str(e))
            continue

        if item_id not in items:
            errors.append(f"{stem}.wav: item_id '{item_id}' not found in items.csv")
            continue

        if stem not in references:
            errors.append(f"{stem}.wav: utt_id '{stem}' not found in references.csv")
            continue

        item = items[item_id]
        ref = references[stem]
        if item["text"] != ref["script_text"]:
            errors.append(
                f"{stem}.wav: script_text disagrees between items.csv and references.csv "
                f"({item['text']!r} vs {ref['script_text']!r})"
            )
            continue

        rows.append({
            "utt_id": stem,
            "session": session,
            "pass": pass_name,
            "item_id": item_id,
            "item_type": item["item_type"],
            "script_text": item["text"],
            "verbatim_text": ref["verbatim_text"],
            "was_corrected": ref["was_corrected"],
            "wav_path": path.as_posix(),
            "duration_sec": durations[path],
        })
    return rows, errors


def print_grid(rows: list[dict]) -> None:
    counts: dict[tuple, int] = {}
    for row in rows:
        key = (row["session"], row["pass"])
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(counts)} session x pass cell(s):")
    for (session, pass_name), n in sorted(counts.items()):
        print(f"  {session} {pass_name}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--items-csv", type=Path, default=Path("stimuli/items.csv"))
    parser.add_argument("--references-csv", type=Path, default=Path("results/references.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/manifest.csv"))
    args = parser.parse_args()

    wav_paths = find_split_wavs(args.split_dir)
    if not wav_paths:
        sys.exit(f"No wav files found under {args.split_dir}")

    items = load_items(args.items_csv)
    references = load_references(args.references_csv)

    durations = {path: get_wav_duration(path) for path in wav_paths}

    rows, errors = build_manifest_rows(wav_paths, items, references, durations)

    if errors:
        print(f"Refusing to write {args.out}: {len(errors)} problem(s) found:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id", "session", "pass", "item_id", "item_type",
                "script_text", "verbatim_text", "was_corrected", "wav_path", "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print_grid(rows)
    print(f"Wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
