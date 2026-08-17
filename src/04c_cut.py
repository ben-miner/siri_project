"""Cut each utterance out of its sentence block using hand-corrected boundaries.

Reads results/segmentation.csv (start_sec/end_sec as corrected by hand after
reviewing 04b_align.py's output) and extracts each item to
data/split/{session}_{pass}_{item_id}.wav with 150 ms of padding on each
side. Padding is clamped to the midpoint of the gap to the neighbouring item
(or to the file edges for the first/last item in a take) so padded clips
never overlap, even when a gap is smaller than 2x the pad.

Everything is validated BEFORE any audio is written:
  - every item_id in stimuli/orders.csv appears exactly once per take
    (a take is one (session, pass) pair)
  - no negative/zero durations
  - no overlapping raw boundaries within a take
  - every duration is between 0.5 and 12.0 seconds

The full report prints regardless of outcome. On failure the script refuses
to write anything unless --force is passed, in which case only the rows
involved in a failed check are skipped -- everything else is still cut.

Source audio is data/blocks/{session}_pass{pass}_block.wav (the same file
04b_align.py transcribed), since segmentation.csv's timestamps are relative
to that file, not to data/converted/. Extraction uses parselmouth's
extract_part with a rectangular window: no resampling, no filtering.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PAD_SEC = 0.150
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 12.0


def load_segmentation(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for row in rows:
        start = float(row["start_sec"]) if row["start_sec"] not in ("", None) else None
        end = float(row["end_sec"]) if row["end_sec"] not in ("", None) else None
        parsed.append({
            "session": row["session"],
            "pass": row["pass"],
            "item_id": row["item_id"],
            "position": int(row["position"]),
            "start_sec": start,
            "end_sec": end,
        })
    return parsed


def load_expected_items(orders_csv: Path) -> dict:
    with orders_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected: dict[str, set] = {}
    for row in rows:
        expected.setdefault(row["pass"], set()).add(row["item_id"])
    return expected


def group_by_take(rows: list[dict]) -> dict:
    takes: dict[tuple, list[dict]] = {}
    for row in rows:
        takes.setdefault((row["session"], row["pass"]), []).append(row)
    return takes


def check_item_coverage(take_rows: list[dict], expected_item_ids: set) -> list[dict]:
    """Each issue: {item_ids: [...], message: str}."""
    actual_counts = Counter(row["item_id"] for row in take_rows)
    issues = []
    for item_id in sorted(expected_item_ids):
        n = actual_counts.get(item_id, 0)
        if n == 0:
            issues.append({"item_ids": [item_id], "message": f"missing item_id {item_id}"})
        elif n > 1:
            issues.append({
                "item_ids": [item_id],
                "message": f"duplicate item_id {item_id} (appears {n} times)",
            })
    for item_id in sorted(set(actual_counts) - expected_item_ids):
        issues.append({
            "item_ids": [item_id],
            "message": f"unexpected item_id {item_id} (not in orders.csv for this pass)",
        })
    return issues


def check_durations(take_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    missing, non_positive, out_of_range = [], [], []
    for row in take_rows:
        item_id = row["item_id"]
        if row["start_sec"] is None or row["end_sec"] is None:
            missing.append({"item_ids": [item_id], "message": f"{item_id}: missing start_sec/end_sec"})
            continue
        duration = row["end_sec"] - row["start_sec"]
        if duration <= 0:
            non_positive.append({
                "item_ids": [item_id],
                "message": (
                    f"{item_id}: start={row['start_sec']:.3f} end={row['end_sec']:.3f} "
                    f"duration={duration:.3f}"
                ),
            })
        elif duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            out_of_range.append({
                "item_ids": [item_id], "message": f"{item_id}: duration={duration:.3f}s",
            })
    return missing, non_positive, out_of_range


def check_overlaps(take_rows: list[dict]) -> list[dict]:
    timed = [r for r in take_rows if r["start_sec"] is not None and r["end_sec"] is not None]
    ordered = sorted(timed, key=lambda r: r["start_sec"])
    issues = []
    for prev, cur in zip(ordered, ordered[1:]):
        if cur["start_sec"] < prev["end_sec"]:
            issues.append({
                "item_ids": [prev["item_id"], cur["item_id"]],
                "message": (
                    f"{prev['item_id']} (end={prev['end_sec']:.3f}) overlaps "
                    f"{cur['item_id']} (start={cur['start_sec']:.3f})"
                ),
            })
    return issues


def validate(rows: list[dict], expected_items: dict) -> tuple[list[str], bool, set]:
    report: list[str] = []
    ok = True
    bad_keys: set = set()  # (session, pass, item_id) to skip when --force is used

    takes = group_by_take(rows)
    n_sessions = len({session for session, _ in takes})
    report.append(
        f"Loaded {len(rows)} row(s) across {len(takes)} take(s) "
        f"({n_sessions} session(s) seen)"
    )

    for (session, pass_name), take_rows in sorted(takes.items()):
        expected = expected_items.get(pass_name)
        if expected is None:
            report.append(f"[FAIL] {session}/{pass_name}: pass not found in stimuli/orders.csv")
            ok = False
            for row in take_rows:
                bad_keys.add((session, pass_name, row["item_id"]))
            continue

        coverage_issues = check_item_coverage(take_rows, expected)
        missing, non_positive, out_of_range = check_durations(take_rows)
        overlap_issues = check_overlaps(take_rows)
        all_issues = [
            ("coverage", coverage_issues),
            ("missing boundary", missing),
            ("non-positive duration", non_positive),
            (f"duration out of [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}]s", out_of_range),
            ("overlap", overlap_issues),
        ]

        take_ok = not any(issues for _, issues in all_issues)
        status = "PASS" if take_ok else "FAIL"
        report.append(f"[{status}] {session}/{pass_name}: {len(take_rows)} row(s)")
        for label, issues in all_issues:
            for issue in issues:
                report.append(f"    {label}: {issue['message']}")

        if not take_ok:
            ok = False
            for _, issues in all_issues:
                for issue in issues:
                    for item_id in issue["item_ids"]:
                        bad_keys.add((session, pass_name, item_id))

    report.append(f"=== Result: {'PASS' if ok else 'FAIL'} ===")
    return report, ok, bad_keys


def block_filename(session: str, pass_name: str) -> str:
    return f"{session}_pass{pass_name}_block.wav"


def compute_padded_bounds(
    sorted_rows: list[dict], pad: float, file_duration: float
) -> list[tuple[float, float]]:
    """Pad each row by `pad` on each side, clamped to the midpoint of the gap to
    its neighbour (or to the file edges at the ends) so padded windows can
    touch but never overlap -- clamping to the neighbour's raw boundary
    instead of the midpoint would let two padded windows overlap whenever the
    gap between raw boundaries is smaller than 2*pad."""
    bounds = []
    for i, row in enumerate(sorted_rows):
        if i > 0:
            left_bound = (sorted_rows[i - 1]["end_sec"] + row["start_sec"]) / 2
        else:
            left_bound = 0.0
        if i < len(sorted_rows) - 1:
            right_bound = (row["end_sec"] + sorted_rows[i + 1]["start_sec"]) / 2
        else:
            right_bound = file_duration
        padded_start = max(row["start_sec"] - pad, left_bound, 0.0)
        padded_end = min(row["end_sec"] + pad, right_bound, file_duration)
        bounds.append((padded_start, padded_end))
    return bounds


def cut_take(
    session: str,
    pass_name: str,
    take_rows: list[dict],
    blocks_dir: Path,
    split_dir: Path,
    bad_keys: set,
) -> tuple[int, int]:
    import parselmouth

    good_rows = [
        r for r in take_rows if (session, pass_name, r["item_id"]) not in bad_keys
    ]
    skipped = len(take_rows) - len(good_rows)
    if not good_rows:
        return 0, skipped

    block_path = blocks_dir / block_filename(session, pass_name)
    if not block_path.exists():
        raise FileNotFoundError(f"{block_path} not found")
    sound = parselmouth.Sound(str(block_path))

    sorted_rows = sorted(good_rows, key=lambda r: r["start_sec"])
    bounds = compute_padded_bounds(sorted_rows, PAD_SEC, sound.xmax)

    split_dir.mkdir(parents=True, exist_ok=True)
    for row, (padded_start, padded_end) in zip(sorted_rows, bounds):
        clip = sound.extract_part(from_time=padded_start, to_time=padded_end, preserve_times=False)
        out_path = split_dir / f"{session}_{pass_name}_{row['item_id']}.wav"
        clip.save(str(out_path), parselmouth.SoundFileFormat.WAV)

    return len(sorted_rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-csv", type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument("--orders-csv", type=Path, default=Path("stimuli/orders.csv"))
    parser.add_argument("--blocks-dir", type=Path, default=Path("data/blocks"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/split"))
    parser.add_argument(
        "--force", action="store_true",
        help="Write output even if validation fails; rows involved in a failed "
             "check are skipped rather than cut.",
    )
    args = parser.parse_args()

    rows = load_segmentation(args.segmentation_csv)
    expected_items = load_expected_items(args.orders_csv)

    report, ok, bad_keys = validate(rows, expected_items)
    print("\n".join(report))

    if not ok and not args.force:
        sys.exit(
            "\nValidation failed. Refusing to write output. "
            "Re-run with --force to write everything except the flagged rows."
        )

    if not ok:
        print(f"\n--force given: skipping {len(bad_keys)} bad row(s), cutting the rest.")

    takes = group_by_take(rows)
    total_cut = 0
    total_skipped = 0
    for (session, pass_name), take_rows in sorted(takes.items()):
        n_cut, n_skipped = cut_take(
            session, pass_name, take_rows, args.blocks_dir, args.split_dir, bad_keys
        )
        total_cut += n_cut
        total_skipped += n_skipped

    print(
        f"\nCut {total_cut} utterance(s) to {args.split_dir} "
        f"({total_skipped} skipped due to validation failures)"
    )


if __name__ == "__main__":
    main()
