"""Select a stratified 10-utterance sample for hand validation of the
phonpipe adaptive-range fix (branch fix/adaptive-range-period-doubling in
the sibling phonpipe repo).

Uses the FULL, unfiltered join of manifest.csv + phonpipe's raw
acoustics_summary.csv, NOT results/acoustics_joined.csv -- that file
excludes every row that tripped f0_tracking_failed (Stage 6's --force run
dropped them from the join entirely), which is exactly the population two
of the four strata below need to select from. This is v1 (pre-fix) data.

f0_tracking_failed reuses 06_measure.py's HEAVY_INTERVENTION_FLAGS
definition: no_pitch, heavy_repair, or octave_disagreement present in
phonpipe's f0_quality_flag. period_doubling_present is deliberately
excluded (it's the creak signal itself, not a tracking defect -- see
06_measure.py's docstring).

Strata (10 utterances total):
  1. 4x C_creak, highest shr_median, f0_tracking_failed=True
     (the loudest failures)
  2. 2x C_creak, shr_median > 0.40, f0_tracking_failed=False
     (silent failures -- high period-doubling that never got flagged)
  3. 2x B_natural, highest shr_median (phrase-final creak)
  4. 2x A_modal, f0_mean_hz closest to the A_modal median
     (controls -- the fix must not break what was already correct)

s02_C_creak_F011 and s02_C_creak_F023 are force-included in stratum 1
(both are already in the f0_tracking_failed=True population, so this is a
substitution within stratum 1, not an 11th/12th row).
"""

import argparse
import csv
from pathlib import Path

HEAVY_INTERVENTION_FLAGS = frozenset({"no_pitch", "heavy_repair", "octave_disagreement"})
FORCED_C_CREAK_UTT_IDS = ("s02_C_creak_F011", "s02_C_creak_F023")
SILENT_FAILURE_SHR_THRESHOLD = 0.40
BLANK_COLUMNS = (
    "window_start_sec", "window_end_sec", "n_pulses", "f0_hand_hz",
    "amplitude_alternation", "notes",
)


def load_manifest(manifest_csv: Path) -> list[dict]:
    with manifest_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_acoustics_summary(summary_csv: Path) -> dict:
    with summary_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        utt_id = Path(row["file"]).stem
        if utt_id in by_utt_id:
            raise ValueError(f"duplicate utt_id '{utt_id}' in {summary_csv}")
        by_utt_id[utt_id] = row
    return by_utt_id


def has_heavy_intervention(f0_quality_flag: str) -> bool:
    tokens = {t for t in (f0_quality_flag or "").split(";") if t}
    return bool(tokens & HEAVY_INTERVENTION_FLAGS)


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_full_dataset(manifest_rows: list[dict], acoustics_by_utt_id: dict) -> list[dict]:
    rows = []
    missing = []
    for row in manifest_rows:
        utt_id = row["utt_id"]
        if utt_id not in acoustics_by_utt_id:
            missing.append(utt_id)
            continue
        acoustics = acoustics_by_utt_id[utt_id]
        rows.append({
            "utt_id": utt_id,
            "pass": row["pass"],
            "shr_median": _float_or_none(acoustics.get("shr_median")),
            "f0_mean_hz": _float_or_none(acoustics.get("f0_mean_hz")),
            "f0_tracking_failed": has_heavy_intervention(acoustics.get("f0_quality_flag", "")),
            "wav_path": row["wav_path"],
        })
    if missing:
        raise ValueError(
            f"{len(missing)} utt_id(s) from manifest.csv missing from acoustics summary:\n  "
            + "\n  ".join(missing)
        )
    return rows


def select_loud_c_creak_failures(rows: list[dict], n: int, forced_utt_ids) -> list[dict]:
    candidates = [
        r for r in rows
        if r["pass"] == "C_creak" and r["f0_tracking_failed"] and r["shr_median"] is not None
    ]
    forced = [r for r in candidates if r["utt_id"] in forced_utt_ids]
    found_ids = {r["utt_id"] for r in forced}
    missing = set(forced_utt_ids) - found_ids
    if missing:
        raise ValueError(
            f"forced utt_id(s) not found among C_creak f0_tracking_failed=True rows: {sorted(missing)}"
        )
    remaining = sorted(
        (r for r in candidates if r["utt_id"] not in found_ids),
        key=lambda r: r["shr_median"], reverse=True,
    )
    selected = forced + remaining[: n - len(forced)]
    if len(selected) < n:
        raise ValueError(f"only {len(selected)} loud C_creak failure candidate(s), need {n}")
    return sorted(selected, key=lambda r: r["shr_median"], reverse=True)


def select_silent_c_creak_failures(
    rows: list[dict], n: int, shr_threshold: float, exclude_utt_ids
) -> list[dict]:
    candidates = sorted(
        (
            r for r in rows
            if r["pass"] == "C_creak" and not r["f0_tracking_failed"]
            and r["shr_median"] is not None and r["shr_median"] > shr_threshold
            and r["utt_id"] not in exclude_utt_ids
        ),
        key=lambda r: r["shr_median"], reverse=True,
    )
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} silent C_creak failure candidate(s), need {n}")
    return candidates[:n]


def select_highest_shr(rows: list[dict], pass_name: str, n: int) -> list[dict]:
    candidates = sorted(
        (r for r in rows if r["pass"] == pass_name and r["shr_median"] is not None),
        key=lambda r: r["shr_median"], reverse=True,
    )
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} {pass_name} candidate(s) with shr_median, need {n}")
    return candidates[:n]


def select_median_f0_controls(rows: list[dict], pass_name: str, n: int) -> list[dict]:
    candidates = [r for r in rows if r["pass"] == pass_name and r["f0_mean_hz"] is not None]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} {pass_name} candidate(s) with f0_mean_hz, need {n}")
    values = sorted(r["f0_mean_hz"] for r in candidates)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    ranked = sorted(candidates, key=lambda r: abs(r["f0_mean_hz"] - median))
    return ranked[:n]


def build_validation_sample(rows: list[dict]) -> list[dict]:
    loud = select_loud_c_creak_failures(rows, 4, FORCED_C_CREAK_UTT_IDS)
    silent = select_silent_c_creak_failures(
        rows, 2, SILENT_FAILURE_SHR_THRESHOLD, exclude_utt_ids={r["utt_id"] for r in loud}
    )
    phrase_final = select_highest_shr(rows, "B_natural", 2)
    controls = select_median_f0_controls(rows, "A_modal", 2)

    selected = loud + silent + phrase_final + controls
    if len(selected) != 10:
        raise ValueError(f"expected exactly 10 rows, got {len(selected)}")
    if len({r["utt_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate utt_id selected across strata")
    return selected


def write_validation_sample_csv(selected: list[dict], out_path: Path) -> None:
    fieldnames = [
        "utt_id", "pass", "shr_median", "f0_mean_hz", "f0_tracking_failed", "wav_path",
    ] + list(BLANK_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            out_row = {k: row.get(k, "") for k in fieldnames}
            out_row.update({col: "" for col in BLANK_COLUMNS})
            writer.writerow(out_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, default=Path("results/manifest.csv"))
    parser.add_argument(
        "--acoustics-summary-csv", type=Path, default=Path("results/acoustics_summary.csv"),
        help="phonpipe's raw, unfiltered summary (default: results/acoustics_summary.csv)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/validation_sample.csv"))
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest_csv)
    acoustics_by_utt_id = load_acoustics_summary(args.acoustics_summary_csv)
    rows = build_full_dataset(manifest_rows, acoustics_by_utt_id)

    selected = build_validation_sample(rows)
    write_validation_sample_csv(selected, args.out)

    strata = (
        ("loud C_creak failures", selected[0:4]),
        ("silent C_creak failures", selected[4:6]),
        ("B_natural phrase-final", selected[6:8]),
        ("A_modal controls", selected[8:10]),
    )
    print(f"Wrote {len(selected)} row(s) to {args.out}\n")
    print("Open these in Praat:")
    for label, group in strata:
        print(f"  -- {label} --")
        for row in group:
            print(f"  {row['wav_path']}")


if __name__ == "__main__":
    main()
