"""Align with MFA on known-correct transcripts, then measure with phonpipe.

1. Writes a {utt_id}.txt transcript (verbatim_text from results/manifest.csv)
   next to each existing wav in data/split/, then runs phonpipe's own
   run_mfa() over that corpus -- NOT phonpipe's automatic Whisper+MFA path
   (align_corpus), because that would transcribe with Whisper first and any
   ASR error would propagate into the alignment. Our transcripts are already
   known-correct (hand-verified in Stage 4d), so MFA gets the real text
   directly. TextGrids land in data/textgrids/mfa/, named {utt_id}.TextGrid
   to match phonpipe's own by-stem TextGrid lookup.

   The transcript .txt files are left in data/split/ alongside the wavs
   (not copied to a separate corpus dir) -- this never touches or overwrites
   any audio, and keeping them is consistent with phonpipe's own
   --work-dir convention of keeping intermediates for inspection rather than
   deleting them after the run.

2. Calls phonpipe's own CLI entrypoint (phonpipe.cli.main, the same function
   the `phonpipe` console command runs) in-process with
   ["extract", *wavs, "--textgrid-dir", ..., "--lang", "english", "-o",
   "results/acoustics"] -- not a reimplementation of any measurement, just
   invoking the tool exactly as documented, in-process to avoid shell
   command-length limits on 600 wav paths.

3. Joins the resulting acoustics_summary.csv to manifest.csv on utt_id.
   phonpipe's own `file` column is the wav filename WITH extension
   (wav_path.name in extract.py), so utt_id is recovered as Path(file).stem
   rather than assumed to already equal utt_id verbatim.

Fails loudly (before writing acoustics_joined.csv) if:
  - MFA didn't produce a TextGrid for every utt_id in the manifest
  - any utt_id is missing from either manifest.csv or acoustics_summary.csv
  - any row's f0_quality_flag contains a flag in HEAVY_INTERVENTION_FLAGS

HEAVY_INTERVENTION_FLAGS is deliberately narrower than "any flag at all".
phonpipe's f0_quality_flag can also report things that are informational
about the *speech itself* rather than the pitch tracker failing --
"period_doubling_present" is exactly the creak signal this project is
measuring, and refusing to process the creak condition's own files on
those grounds would be self-defeating (same reasoning as 04b_align.py's
"a low match_ratio on the creak pass is not necessarily a bug"). The flags
below instead mark cases where the tracker itself needed heavy correction
or an unresolved probable octave error -- see phonpipe/measures/f0.py.

--force downgrades all of the above from "refuse to write anything" to
"exclude just the affected utt_id(s) and join the rest", printing what was
excluded and why -- mirroring 04c_cut.py's validate-then-force pattern.

Every joined row also carries f0_tracking_failed (bool, derived from
f0_quality_flag the same way the exclusion gate computes it) so a flagged
row is visible in the CSV itself, not just in this script's console output.
--keep-flagged additionally stops heavy_intervention from being treated as
an exclusion-triggering problem at all: every utt_id that joins cleanly on
both sides is written, flagged or not, with f0_quality_flag and
f0_tracking_failed as the record of which rows need caution. Missing-utt_id
problems (a join failing entirely) are unaffected by this flag and still
require --force.
"""

import argparse
import csv
import sys
from pathlib import Path

HEAVY_INTERVENTION_FLAGS = frozenset({"no_pitch", "heavy_repair", "octave_disagreement"})

MANIFEST_COLUMNS = [
    "utt_id", "session", "pass", "item_id", "item_type",
    "script_text", "verbatim_text", "was_corrected", "wav_path", "duration_sec",
]


def load_manifest(manifest_csv: Path) -> list[dict]:
    with manifest_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_mfa_corpus(manifest_rows: list[dict], corpus_dir: Path) -> None:
    """Write a {utt_id}.txt transcript next to each wav already in corpus_dir."""
    for row in manifest_rows:
        txt_path = corpus_dir / f"{row['utt_id']}.txt"
        txt_path.write_text(row["verbatim_text"], encoding="utf-8")


def find_missing_textgrids(manifest_rows: list[dict], textgrid_dir: Path) -> list[str]:
    return [
        row["utt_id"] for row in manifest_rows
        if not (textgrid_dir / f"{row['utt_id']}.TextGrid").exists()
    ]


def load_acoustics_summary(summary_csv: Path) -> dict:
    with summary_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        utt_id = Path(row["file"]).stem
        if utt_id in by_utt_id:
            raise ValueError(f"duplicate utt_id '{utt_id}' in {summary_csv} (file={row['file']!r})")
        by_utt_id[utt_id] = row
    return by_utt_id


def has_heavy_intervention(f0_quality_flag: str) -> bool:
    tokens = {t for t in f0_quality_flag.split(";") if t}
    return bool(tokens & HEAVY_INTERVENTION_FLAGS)


def find_join_problems(
    manifest_rows: list[dict], acoustics_by_utt_id: dict, keep_flagged: bool = False
) -> list[dict]:
    manifest_utt_ids = {row["utt_id"] for row in manifest_rows}
    acoustics_utt_ids = set(acoustics_by_utt_id.keys())

    problems = []
    for utt_id in sorted(manifest_utt_ids - acoustics_utt_ids):
        problems.append({"utt_id": utt_id, "category": "missing_from_acoustics", "detail": ""})
    for utt_id in sorted(acoustics_utt_ids - manifest_utt_ids):
        problems.append({"utt_id": utt_id, "category": "missing_from_manifest", "detail": ""})
    if not keep_flagged:
        for utt_id in sorted(manifest_utt_ids & acoustics_utt_ids):
            flag = acoustics_by_utt_id[utt_id].get("f0_quality_flag", "") or ""
            if has_heavy_intervention(flag):
                problems.append({"utt_id": utt_id, "category": "heavy_intervention", "detail": flag})
    return problems


def build_joined_rows(
    manifest_rows: list[dict], acoustics_by_utt_id: dict, exclude_utt_ids: set
) -> list[dict]:
    rows = []
    for row in manifest_rows:
        utt_id = row["utt_id"]
        if utt_id in exclude_utt_ids or utt_id not in acoustics_by_utt_id:
            continue
        joined = dict(row)
        joined.update(acoustics_by_utt_id[utt_id])
        joined["f0_tracking_failed"] = has_heavy_intervention(joined.get("f0_quality_flag", "") or "")
        rows.append(joined)
    return rows


def write_joined_csv(joined_rows: list[dict], out_path: Path) -> None:
    if not joined_rows:
        raise ValueError("no rows to write to " + str(out_path))
    extra_cols = [c for c in joined_rows[0].keys() if c not in MANIFEST_COLUMNS]
    fieldnames = MANIFEST_COLUMNS + extra_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=Path("results/manifest.csv"))
    parser.add_argument(
        "--corpus-dir", type=Path, default=None,
        help="MFA corpus directory (wav+txt pairs); default: same as the wavs' own directory",
    )
    parser.add_argument("--textgrid-dir", type=Path, default=Path("data/textgrids/mfa"))
    parser.add_argument("--lang", default="english", choices=["english", "japanese"])
    parser.add_argument(
        "--acoustics-out", type=Path, default=Path("results/acoustics"),
        help="output prefix passed to phonpipe extract -o (default: results/acoustics)",
    )
    parser.add_argument("--joined-out", type=Path, default=Path("results/acoustics_joined.csv"))
    parser.add_argument(
        "--skip-mfa", action="store_true",
        help="reuse existing data/textgrids/mfa/ instead of rerunning MFA",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="reuse an existing results/acoustics_summary.csv instead of rerunning phonpipe extract",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="exclude problem utt_id(s) and join the rest, instead of refusing to write anything",
    )
    parser.add_argument(
        "--keep-flagged", action="store_true",
        help="don't treat heavy_intervention as a problem at all -- write every cleanly-joined "
             "row regardless of f0_quality_flag, relying on the f0_tracking_failed column "
             "instead of exclusion. Missing-utt_id problems still require --force.",
    )
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest_csv)
    if not manifest_rows:
        sys.exit(f"{args.manifest_csv} has no rows")

    wav_paths = [Path(row["wav_path"]) for row in manifest_rows]
    missing_wavs = [p for p in wav_paths if not p.exists()]
    if missing_wavs:
        sys.exit(
            f"{len(missing_wavs)} wav path(s) from {args.manifest_csv} don't exist:\n  "
            + "\n  ".join(str(p) for p in missing_wavs)
        )

    corpus_dir = args.corpus_dir or wav_paths[0].parent

    if not args.skip_mfa:
        from phonpipe.align import run_mfa

        build_mfa_corpus(manifest_rows, corpus_dir)
        print(f"Running MFA on {corpus_dir} ({len(manifest_rows)} utterance(s))...")
        run_mfa(corpus_dir, args.lang, args.textgrid_dir)

    missing_tg = find_missing_textgrids(manifest_rows, args.textgrid_dir)
    if missing_tg:
        sys.exit(
            f"MFA did not produce a TextGrid for {len(missing_tg)} utt_id(s) in {args.textgrid_dir}:\n  "
            + "\n  ".join(missing_tg)
        )

    if not args.skip_extract:
        from phonpipe.cli import main as phonpipe_main

        argv = [
            "extract", *[str(p) for p in wav_paths],
            "--textgrid-dir", str(args.textgrid_dir),
            "--lang", args.lang,
            "-o", str(args.acoustics_out),
        ]
        print(f"Running: phonpipe {' '.join(argv)}")
        ret = phonpipe_main(argv)
        if ret != 0:
            sys.exit(f"phonpipe extract exited with code {ret}")

    summary_csv = Path(f"{args.acoustics_out}_summary.csv")
    if not summary_csv.exists():
        sys.exit(f"{summary_csv} not found (phonpipe extract may have failed)")
    acoustics_by_utt_id = load_acoustics_summary(summary_csv)

    problems = find_join_problems(manifest_rows, acoustics_by_utt_id, keep_flagged=args.keep_flagged)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            suffix = f" ({p['detail']})" if p["detail"] else ""
            print(f"  [{p['category']}] {p['utt_id']}{suffix}")
        if not args.force:
            sys.exit(
                "\nRefusing to write joined output. Inspect the row(s) above by hand, "
                "or re-run with --force to exclude them and join the rest."
            )
        print(f"\n--force given: excluding {len(problems)} row(s), joining the rest.")

    exclude = {p["utt_id"] for p in problems}
    joined_rows = build_joined_rows(manifest_rows, acoustics_by_utt_id, exclude)
    write_joined_csv(joined_rows, args.joined_out)
    print(f"Wrote {len(joined_rows)} row(s) to {args.joined_out}")


if __name__ == "__main__":
    main()
