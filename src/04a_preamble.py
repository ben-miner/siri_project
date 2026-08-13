"""Extract calibration clips and the sentence block from each preamble TextGrid.

Each take in data/textgrids/preamble/ is a hand-made TextGrid (single tier
"metadata") with 5 labelled intervals: roomtone, sustained_a, glide,
creak_cal, carrier. For each take this script:

  1. Extracts each labelled interval to
     data/calibration/{session}_{pass}_{label}.wav
  2. Extracts everything from the end of the carrier interval to the end
     of the file to data/blocks/{session}_{pass}_block.wav
  3. Writes results/preamble_index.csv (take, label, start_sec, end_sec,
     sentence_block_start_sec)

{session} and {pass} come from the TextGrid stem itself (e.g.
"s01_passA_modal" -> session "s01", pass "passA_modal"), split on the
first underscore, so calibration/block filenames line up with the take
by construction rather than by re-deriving names some other way.

Audio is read from data/converted/{take}.wav (the mono, un-resampled
output of 02_convert.py) and sliced with parselmouth's extract_part
using a rectangular window (no tapering, no resampling, no filtering).
"""

import argparse
import csv
import sys
from pathlib import Path

EXPECTED_LABELS = ["roomtone", "sustained_a", "glide", "creak_cal", "carrier"]
DURATION_TOLERANCE_SEC = 0.01


def find_textgrids(tg_dir: Path) -> list[Path]:
    return sorted(tg_dir.glob("*.TextGrid"))


def parse_take(take: str) -> tuple[str, str]:
    if "_" not in take:
        raise ValueError(f"take '{take}' has no underscore to split session from pass")
    session, pass_label = take.split("_", 1)
    return session, pass_label


def validate_intervals(intervals: dict, take: str) -> None:
    labels = list(intervals.keys())
    if sorted(labels) != sorted(EXPECTED_LABELS) or len(labels) != len(EXPECTED_LABELS):
        raise ValueError(
            f"{take}: expected labelled intervals {EXPECTED_LABELS}, got {labels}"
        )


def build_index_rows(take: str, intervals: dict, sentence_block_start_sec: float) -> list[dict]:
    return [
        {
            "take": take,
            "label": label,
            "start_sec": intervals[label][0],
            "end_sec": intervals[label][1],
            "sentence_block_start_sec": sentence_block_start_sec,
        }
        for label in EXPECTED_LABELS
    ]


def read_labelled_intervals(tg_path: Path) -> dict:
    from phonpipe.align import read_phones

    phone_intervals = read_phones(tg_path, tier_name="metadata")
    intervals = {}
    for interval in phone_intervals:
        if interval.label in intervals:
            raise ValueError(
                f"{tg_path.name}: label '{interval.label}' appears more than once"
            )
        intervals[interval.label] = (interval.start, interval.end)
    return intervals


def process_take(
    tg_path: Path,
    audio_dir: Path,
    calibration_dir: Path,
    blocks_dir: Path,
) -> list[dict]:
    import parselmouth

    take = tg_path.stem
    session, pass_label = parse_take(take)

    intervals = read_labelled_intervals(tg_path)
    validate_intervals(intervals, take)

    audio_path = audio_dir / f"{take}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(
            f"{audio_path} not found. Run 02_convert.py to produce mono "
            f"converted audio before running this script."
        )

    sound = parselmouth.Sound(str(audio_path))

    calibration_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)

    for label in EXPECTED_LABELS:
        start, end = intervals[label]
        clip = sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        out_path = calibration_dir / f"{session}_{pass_label}_{label}.wav"
        clip.save(str(out_path), parselmouth.SoundFileFormat.WAV)

    sentence_block_start_sec = intervals["carrier"][1]
    if sentence_block_start_sec > sound.xmax + DURATION_TOLERANCE_SEC:
        raise ValueError(
            f"{take}: carrier ends at {sentence_block_start_sec:.3f}s but audio "
            f"is only {sound.xmax:.3f}s long"
        )
    block = sound.extract_part(
        from_time=sentence_block_start_sec, to_time=sound.xmax, preserve_times=False
    )
    block_path = blocks_dir / f"{session}_{pass_label}_block.wav"
    block.save(str(block_path), parselmouth.SoundFileFormat.WAV)

    return build_index_rows(take, intervals, sentence_block_start_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tg-dir", type=Path, default=Path("data/textgrids/preamble"),
        help="Directory of preamble TextGrids (default: data/textgrids/preamble)",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=Path("data/converted"),
        help="Directory of mono converted audio, one wav per take (default: data/converted)",
    )
    parser.add_argument(
        "--calibration-dir", type=Path, default=Path("data/calibration"),
        help="Output directory for calibration clips (default: data/calibration)",
    )
    parser.add_argument(
        "--blocks-dir", type=Path, default=Path("data/blocks"),
        help="Output directory for sentence-block wavs (default: data/blocks)",
    )
    parser.add_argument(
        "--index-out", type=Path, default=Path("results/preamble_index.csv"),
        help="Output CSV path (default: results/preamble_index.csv)",
    )
    args = parser.parse_args()

    tg_paths = find_textgrids(args.tg_dir)
    if not tg_paths:
        sys.exit(f"No TextGrids found under {args.tg_dir}")

    all_rows = []
    for tg_path in tg_paths:
        rows = process_take(tg_path, args.audio_dir, args.calibration_dir, args.blocks_dir)
        all_rows.extend(rows)

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    with args.index_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["take", "label", "start_sec", "end_sec", "sentence_block_start_sec"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    n_takes = len(tg_paths)
    print(
        f"Extracted {n_takes * len(EXPECTED_LABELS)} calibration clip(s) and "
        f"{n_takes} block(s) from {n_takes} take(s); wrote {len(all_rows)} row(s) to {args.index_out}"
    )


if __name__ == "__main__":
    main()
