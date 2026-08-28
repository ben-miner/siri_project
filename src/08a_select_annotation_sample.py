"""Select a 60-utterance sample for blind hand annotation, stratified on
phonpipe's creak proportion.

There is no column literally named "creak_proportion" in
results/acoustics_joined.csv. This script uses creak_doubling_rate --
phonpipe's fraction of voiced frames whose SHR exceeds
CREAK_THRESHOLDS["shr_doubling"] (see phonpipe/measures/creak.py), i.e. the
proportion of the utterance classified as period-doubled/creaky -- since
that is the closest existing measure to "how much of this utterance is
creak" and has complete coverage (0 NaN across all 600 rows).

STRATIFICATION: utterances are ranked by creak_doubling_rate and split into
three equal-sized pools by rank (terciles of the full 600-utterance corpus,
not of the 60-item sample), then 20 are sampled from each pool.

PASS BALANCE WITHIN EACH TERCILE: creak_doubling_rate correlates strongly
with pass, by design (A_modal is supposed to be non-creaky, C_creak creaky).
On this corpus that correlation is total at the extremes: the low tercile
contains ZERO C_creak utterances and the high tercile contains ZERO A_modal
utterances (checked directly against acoustics_joined.csv). "All three
passes represented" is therefore not achievable in either extreme tercile
no matter how sampling is done -- this script instead takes every available
utterance from whichever pass(es) are scarce in a tercile and fills the
remainder from the dominant pass, and prints exactly what it did so the
shortfall is visible, not silently absorbed. Expect (and this was verified
against the real 2026-08 v3 data, not just theoretically): low tercile ~ 10
A_modal / 10 B_natural / 0 C_creak; mid tercile ~ 8 A_modal / 8 B_natural /
4 C_creak (all 4 available C_creak used); high tercile ~ 0 A_modal / 4
B_natural / 16 C_creak (all 4 available B_natural used).

OUTPUTS:
  results/annotation_sample.csv -- utt_id, wav_path, pass,
    creak_doubling_rate, tercile, annotation_order. Full record for
    post-annotation analysis (joining hand codes back to the stratification
    design).
  results/annotation_blind.csv -- utt_id, annotation_order ONLY. This is
    the file to annotate from. No creak_doubling_rate, no SHR, no pass
    label -- annotating with phonpipe's own estimate or the condition label
    visible would make any later validation of phonpipe against these
    annotations circular.

annotation_order is a random permutation of 1..60 (both which utterances
are drawn from each pass pool AND the final ordering use one
random.Random(ANNOTATION_SEED) instance, so a re-run reproduces the exact
same 60 utterances in the exact same order).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

ANNOTATION_SEED = 20260821
N_TOTAL = 60
N_TERCILES = 3
N_PER_TERCILE = N_TOTAL // N_TERCILES
CREAK_COLUMN = "creak_doubling_rate"
TERCILE_NAMES = ("low", "mid", "high")


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        v = float(value)
        return v if v == v else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def rows_with_creak_rate(rows: list[dict]) -> list[dict]:
    """Attach a parsed float creak_doubling_rate to each row; drop rows
    where it's missing/NaN, printing how many (fail loudly, don't silently
    stratify on a partial corpus without saying so)."""
    out = []
    dropped = []
    for row in rows:
        rate = _to_float(row.get(CREAK_COLUMN))
        if rate is None:
            dropped.append(row.get("utt_id", "<no utt_id>"))
            continue
        out.append({**row, "_creak_rate": rate})
    if dropped:
        print(f"WARNING: {len(dropped)} row(s) missing {CREAK_COLUMN}, "
              f"excluded from stratification: {dropped}")
    return out


def assign_terciles(rows: list[dict]) -> dict[str, list[dict]]:
    """Rank-based terciles over the FULL corpus (not the eventual sample):
    bottom third / middle third / top third by creak_doubling_rate."""
    ranked = sorted(rows, key=lambda r: r["_creak_rate"])
    n = len(ranked)
    t1, t2 = n // 3, 2 * n // 3
    return {
        "low": ranked[:t1],
        "mid": ranked[t1:t2],
        "high": ranked[t2:],
    }


def sample_pass_balanced(pool: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Round-robin across passes (alphabetical order: A_modal, B_natural,
    C_creak) so representation is as even as availability allows -- a pass
    with fewer utterances than its even share simply contributes all it
    has, and the round-robin naturally reallocates the remainder to
    whichever passes still have supply, without needing to precompute
    exact per-pass quotas."""
    by_pass: dict[str, list[dict]] = {}
    for row in pool:
        by_pass.setdefault(row["pass"], []).append(row)
    for p in by_pass:
        rng.shuffle(by_pass[p])

    passes = sorted(by_pass)
    idx = {p: 0 for p in passes}
    chosen: list[dict] = []
    while len(chosen) < target:
        made_progress = False
        for p in passes:
            if idx[p] < len(by_pass[p]):
                chosen.append(by_pass[p][idx[p]])
                idx[p] += 1
                made_progress = True
                if len(chosen) >= target:
                    break
        if not made_progress:
            break  # every pass in this pool is exhausted
    return chosen


def build_sample(rows: list[dict], rng: random.Random) -> list[dict]:
    with_rate = rows_with_creak_rate(rows)
    terciles = assign_terciles(with_rate)

    selected: list[dict] = []
    for name in TERCILE_NAMES:
        pool = terciles[name]
        picked = sample_pass_balanced(pool, N_PER_TERCILE, rng)
        if len(picked) < N_PER_TERCILE:
            sys.exit(
                f"Only found {len(picked)} of {N_PER_TERCILE} needed utterances "
                f"for the '{name}' tercile (pool size {len(pool)}) -- refusing "
                f"to write a short sample silently.")
        from collections import Counter
        counts = Counter(r["pass"] for r in picked)
        print(f"  {name} tercile: {len(picked)} selected -- {dict(counts)}")
        for row in picked:
            selected.append({**row, "tercile": name})

    return selected


def assign_annotation_order(selected: list[dict], rng: random.Random) -> list[dict]:
    order = list(range(1, len(selected) + 1))
    rng.shuffle(order)
    for row, pos in zip(selected, order):
        row["annotation_order"] = pos
    return sorted(selected, key=lambda r: r["annotation_order"])


def write_sample_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["utt_id", "wav_path", "pass", "creak_doubling_rate", "tercile", "annotation_order"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "utt_id": row["utt_id"],
                "wav_path": row["wav_path"],
                "pass": row["pass"],
                "creak_doubling_rate": row["_creak_rate"],
                "tercile": row["tercile"],
                "annotation_order": row["annotation_order"],
            })


def write_blind_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["utt_id", "annotation_order"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "utt_id": row["utt_id"],
                "annotation_order": row["annotation_order"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustics-joined", type=Path, default=Path("results/acoustics_joined.csv"))
    parser.add_argument("--sample-out", type=Path, default=Path("results/annotation_sample.csv"))
    parser.add_argument("--blind-out", type=Path, default=Path("results/annotation_blind.csv"))
    parser.add_argument("--seed", type=int, default=ANNOTATION_SEED)
    args = parser.parse_args()

    if not args.acoustics_joined.exists():
        sys.exit(f"{args.acoustics_joined} not found")
    rows = load_rows(args.acoustics_joined)
    if not rows:
        sys.exit(f"{args.acoustics_joined} has no rows")

    rng = random.Random(args.seed)
    selected = build_sample(rows, rng)
    selected = assign_annotation_order(selected, rng)

    write_sample_csv(selected, args.sample_out)
    write_blind_csv(selected, args.blind_out)
    print(f"Wrote {len(selected)} row(s) to {args.sample_out} and {args.blind_out} "
          f"(seed={args.seed})")


if __name__ == "__main__":
    main()
