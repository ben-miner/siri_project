"""Prepare blind, annotation-ready TextGrids for the 60-utterance sample.

For each utt_id in results/annotation_sample.csv, copies its MFA TextGrid
(data/textgrids/mfa/{utt_id}.TextGrid) to data/textgrids/annotation/, adding
a new "creak" interval tier whose boundaries exactly mirror the phone
tier's sonorant intervals (config/sonorants.yaml -- see that file for why
it's a separate, extended copy of phonpipe's SONORANTS set, not an import
of it) with every interval labeled "" (empty), ready for the researcher to
mark. Sonorant restriction follows White, Penney, Gibson, Szakay & Cox
(2022): phonpipe applies the same restriction to its own SHR/H1-H2 creak
measures via sonorant_intervals() (phonpipe/measures/creak.py) -- this
script does not call that function, since the whole point of
config/sonorants.yaml is a project-owned phone list independent of
phonpipe's internal one.

TIER CONSTRUCTION: walking the phone tier in time order, each sonorant
phone's own (start, end) becomes its own interval in the new tier (adjacent
sonorant phones are NOT merged into one span -- each phone's boundary is
preserved exactly, matching "mirror the phone tier's sonorant intervals"
literally). Every other span (obstruents, silence, and file-edge gaps) is
merged into a surrounding blank interval, since a Praat IntervalTier must
cover its full extent with no gaps. Every interval's label is "" regardless
of whether it corresponds to a sonorant phone or a gap -- the point is that
the BOUNDARY POSITIONS mark where a sonorant is, so opening this TextGrid
next to the phones tier shows exactly which blank intervals to annotate.

BLINDING: filenames are the row's annotation_order, zero-padded to 2 digits
(01.TextGrid .. 60.TextGrid), not utt_id -- nothing in data/textgrids/annotation/
identifies which utterance, session, or pass a grid belongs to. Each wav
(from the row's wav_path, e.g. data/split/{utt_id}.wav) is ALSO copied into
the same directory renamed to match (01.wav .. 60.wav): Praat pairs a Sound
and TextGrid by matching basename in the same folder, so without this the
researcher would have to open the real, identifying wav filename from
data/split/ to get audio alongside the blind TextGrid, defeating the point.
data/split/ itself is untouched -- these are copies, not moves.
results/annotation_key.csv is the re-identification key (annotation_order,
utt_id, grid_filename, wav_filename, wav_path); it is not meant to be
consulted during annotation, only afterward to join hand-coded results
back to utt_id.

data/textgrids/mfa/ and data/split/ are both read-only here -- nothing in
either is modified.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import yaml
from praatio import textgrid as ptg
from praatio.utilities.constants import Interval


def load_sonorants(yaml_path: Path) -> set[str]:
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    labels = data.get("sonorants")
    if not labels:
        sys.exit(f"{yaml_path} has no 'sonorants' list")
    return set(labels)


def load_sample(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sonorant_spans(phones_tier, sonorants: set[str]) -> list[tuple[float, float]]:
    """(start, end) of every sonorant phone in phones_tier, time-sorted.
    Factored out of build_creak_tier so callers that need to re-derive the
    expected sonorant boundaries later (e.g. to validate a hand-annotated
    grid hasn't drifted) use the exact same logic, not a second copy of it."""
    spans = [
        (e.start, e.end) for e in phones_tier.entries
        if e.label.strip() in sonorants
    ]
    spans.sort(key=lambda span: span[0])
    return spans


def build_creak_tier(phones_tier, sonorants: set[str]) -> "ptg.IntervalTier":
    """A gapless, all-empty-label IntervalTier whose boundaries fall at
    exactly the start/end of every sonorant phone in phones_tier."""
    min_t, max_t = phones_tier.minTimestamp, phones_tier.maxTimestamp
    entries: list[Interval] = []
    cursor = min_t
    for start, end in sonorant_spans(phones_tier, sonorants):
        if start > cursor:
            entries.append(Interval(cursor, start, ""))
        elif start < cursor:
            raise ValueError(
                f"overlapping/out-of-order sonorant span ({start}, {end}) "
                f"before cursor={cursor}")
        entries.append(Interval(start, end, ""))
        cursor = end
    if cursor < max_t:
        entries.append(Interval(cursor, max_t, ""))

    return ptg.IntervalTier("creak", entries, min_t, max_t)


def prepare_grid(utt_id: str, source_dir: Path, dest_path: Path, sonorants: set[str]) -> None:
    source_path = source_dir / f"{utt_id}.TextGrid"
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))

    tg = ptg.openTextgrid(str(source_path), includeEmptyIntervals=True)
    names = tg.tierNames
    phones_tier = None
    for cand in ("phones", "phone", "segments"):
        if cand in names:
            phones_tier = tg.getTier(cand)
            break
    if phones_tier is None:
        raise ValueError(f"{source_path} has no phone tier (tiers: {names})")

    creak_tier = build_creak_tier(phones_tier, sonorants)
    tg.addTier(creak_tier)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(dest_path), format="long_textgrid", includeBlankSpaces=True,
            minTimestamp=tg.minTimestamp, maxTimestamp=tg.maxTimestamp)


def copy_audio(source_path: Path, dest_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)


def write_key_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["annotation_order", "utt_id", "grid_filename", "wav_filename", "wav_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=Path("results/annotation_sample.csv"))
    parser.add_argument("--sonorants-yaml", type=Path, default=Path("config/sonorants.yaml"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/textgrids/mfa"))
    parser.add_argument("--dest-dir", type=Path, default=Path("data/textgrids/annotation"))
    parser.add_argument("--key-out", type=Path, default=Path("results/annotation_key.csv"))
    args = parser.parse_args()

    if not args.sample_csv.exists():
        sys.exit(f"{args.sample_csv} not found -- run src/08a_select_annotation_sample.py first")
    sample_rows = load_sample(args.sample_csv)
    if not sample_rows:
        sys.exit(f"{args.sample_csv} has no rows")

    sonorants = load_sonorants(args.sonorants_yaml)

    missing_tg = [
        row["utt_id"] for row in sample_rows
        if not (args.source_dir / f"{row['utt_id']}.TextGrid").exists()
    ]
    missing_wav = [
        row["utt_id"] for row in sample_rows
        if not Path(row["wav_path"]).exists()
    ]
    if missing_tg or missing_wav:
        lines = []
        if missing_tg:
            lines.append(
                f"{len(missing_tg)} utt_id(s) have no TextGrid under {args.source_dir}: "
                + ", ".join(missing_tg))
        if missing_wav:
            lines.append(
                f"{len(missing_wav)} utt_id(s) have no wav file at their wav_path: "
                + ", ".join(missing_wav))
        sys.exit("\n".join(lines))

    key_rows = []
    for row in sample_rows:
        utt_id = row["utt_id"]
        order = int(row["annotation_order"])
        grid_filename = f"{order:02d}.TextGrid"
        wav_filename = f"{order:02d}.wav"

        prepare_grid(utt_id, args.source_dir, args.dest_dir / grid_filename, sonorants)
        copy_audio(Path(row["wav_path"]), args.dest_dir / wav_filename)

        key_rows.append({
            "annotation_order": order,
            "utt_id": utt_id,
            "grid_filename": grid_filename,
            "wav_filename": wav_filename,
            "wav_path": row["wav_path"],
        })

    key_rows.sort(key=lambda r: r["annotation_order"])
    write_key_csv(key_rows, args.key_out)
    print(f"Wrote {len(key_rows)} annotation-ready TextGrid(s) + paired wav(s) to "
          f"{args.dest_dir} and the mapping to {args.key_out}")


if __name__ == "__main__":
    main()
