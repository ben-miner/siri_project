"""Compute hand-annotated creak proportions from the 60 annotated TextGrids,
validate them, and compare against phonpipe's v3 estimate.

1. For each of the 60 grids in data/textgrids/annotation/, reads the
   "creak" tier. A TOKEN is one of the tier's intervals that corresponds to
   a sonorant phone -- i.e. its (start, end) matches a span in
   sonorant_spans(phones_tier, sonorants) (see 08b_prepare_grids.py). The
   tier's other intervals (obstruents, silence, file-edge gaps) are
   structural filler required for a valid Praat IntervalTier and are never
   tokens, regardless of label -- they were never meant to be annotated.
   A token is creaky if its label is non-empty ("c" is the only label the
   researcher used). hand_creak_proportion is TIME-weighted:
       sum(duration of creaky tokens) / sum(duration of all tokens)
   restricted to token (sonorant) duration only, not the whole file --
   matching phonpipe's own creak_doubling_rate, which is also computed over
   sonorant-restricted frames (extract.py passes sonorant_intervals() into
   measure_shr), so the two are comparable in step 5 below.

2. Un-blinds via results/annotation_key.csv (annotation_order -> utt_id)
   and writes results/hand_annotation.csv: utt_id, hand_creak_proportion,
   n_tokens, n_creaky, pass (joined from results/manifest.csv on utt_id,
   per this project's own non-negotiable join key).

3. VALIDATES ALL 60 GRIDS BEFORE WRITING ANYTHING. For each grid, the
   entire tier (tokens AND gap-filler intervals) is rebuilt fresh via
   08b_prepare_grids.build_creak_tier() and compared interval-by-interval
   against what's actually saved (tolerance 5ms -- well below phonetic
   significance, wide enough to absorb ordinary Praat mouse-precision
   noise on a boundary the researcher legitimately touched, but far below
   the 90-900ms+ offsets that flag a genuine structural problem). If they
   don't match,
   repair_extra_gap_boundaries() attempts one specific, safe repair before
   giving up: merge consecutive fragments that share a SPURIOUS boundary
   (one absent from the freshly-rebuilt tier -- i.e. Praat introduced it,
   08b never wrote it) but only when every fragment being merged has an
   IDENTICAL label, so no annotation information is discarded by merging.
   This is exactly the pattern found auditing the real 60-grid batch: 9
   grids had one or more extra boundaries splitting a single blank gap
   region into pieces, every piece still labeled "" -- consistent with an
   accidental stray boundary insertion in Praat, not a deliberate edit.
   Every repair performed is reported (grid, boundary time dropped) -- not
   silent. If a spurious boundary instead separates two DIFFERENT labels
   (e.g. "" next to "c"), that cannot be auto-resolved and is refused.
   Refuses to write hand_annotation.csv if, after any such repair, any of
   the following still hold for any grid:
     - fewer than 60 grids exist under data/textgrids/annotation/ (checked
       against the 60 filenames results/annotation_key.csv expects, not
       just a bare file count -- catches a wrong/renamed file too)
     - a grid has zero tokens on its creak tier
     - a grid's creak tier boundaries still don't match the sonorant
       intervals re-derived from its phone tier after the repair attempt
       (a real boundary was moved, not just split)
     - any label other than "" or "c" (after stripping whitespace) appears
       anywhere on the creak tier, token or not
   Every problem found, across every grid, is collected and reported
   together -- not just the first one.

4. Prints hand_creak_proportion's distribution overall and by pass, plus
   total token/creaky counts, and (as a bonus, cheap diagnostic for the
   "concentrated in utterance-final tokens" expectation) the mean
   normalized time-position of creaky tokens within their utterance.

5. Joins hand_creak_proportion to phonpipe's v3 creak_doubling_rate
   (results/acoustics_joined.csv, the "creak_proportion" measure used
   throughout this project since 08a_select_annotation_sample.py) by
   utt_id, reports Pearson r, and saves a scatter to
   results/annotation_vs_phonpipe.png. This is PRE-TUNING agreement -- a
   diagnostic, not a target. Threshold tuning is a separate, later step,
   not run here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from praatio import textgrid as ptg
from praatio.utilities.constants import Interval

sys.path.insert(0, str(Path(__file__).resolve().parent))
grids = import_module("08b_prepare_grids")  # sonorant_spans, build_creak_tier, load_sonorants

VALID_LABELS = {"", "c"}
# 5ms: well below phonetic significance and far below the 90-900ms+ offsets
# that correctly flagged genuine structural problems in the real 60-grid
# audit -- just wide enough to absorb ordinary Praat mouse-precision noise
# on a boundary the researcher legitimately touched while labeling a token.
BOUNDARY_TOL_S = 5e-3


def load_annotation_key(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pass_by_utt_id(manifest_csv: Path) -> dict[str, str]:
    with manifest_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in by_utt_id:
            raise ValueError(f"duplicate utt_id '{utt_id}' in {manifest_csv}")
        by_utt_id[utt_id] = row["pass"]
    return by_utt_id


def _to_float(value):
    try:
        v = float(value)
        return v if v == v else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def load_phonpipe_creak_rate(acoustics_csv: Path) -> dict[str, float]:
    with acoustics_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        rate = _to_float(row.get("creak_doubling_rate"))
        if rate is not None:
            by_utt_id[row["utt_id"]] = rate
    return by_utt_id


def _boundaries_match(expected: list, actual: list, tol: float = BOUNDARY_TOL_S) -> bool:
    if len(expected) != len(actual):
        return False
    return all(
        abs(e.start - a.start) <= tol and abs(e.end - a.end) <= tol
        for e, a in zip(expected, actual)
    )


def _is_expected_boundary(t: float, expected_times: set, tol: float = BOUNDARY_TOL_S) -> bool:
    return any(abs(t - et) <= tol for et in expected_times)


def repair_extra_gap_boundaries(actual_entries: list, expected_entries: list
                                ) -> tuple[list, list]:
    """Merge consecutive creak-tier intervals whose shared boundary is NOT
    one of the expected tier's boundary points (i.e. a boundary the
    researcher's Praat session introduced beyond what 08b_prepare_grids.py
    originally wrote, typically an accidental extra click/keystroke while
    editing). Only merges intervals with an IDENTICAL (stripped) label --
    raises ValueError if a spurious boundary separates two different
    labels, since there'd be no safe way to pick which one wins.

    Returns (repaired_entries, merges) where merges is
    [(dropped_boundary_time, label), ...] for reporting -- never silent."""
    expected_times = set()
    for e in expected_entries:
        expected_times.add(round(e.start, 6))
        expected_times.add(round(e.end, 6))

    merged: list[Interval] = []
    merges: list[tuple[float, str]] = []
    i = 0
    n = len(actual_entries)
    while i < n:
        current = actual_entries[i]
        while i + 1 < n and not _is_expected_boundary(current.end, expected_times):
            nxt = actual_entries[i + 1]
            if current.label.strip() != nxt.label.strip():
                raise ValueError(
                    f"spurious boundary at t={current.end:.3f}s separates "
                    f"differing labels {current.label!r} and {nxt.label!r} "
                    f"-- cannot auto-merge")
            merges.append((current.end, current.label.strip()))
            current = Interval(current.start, nxt.end, current.label)
            i += 1
        merged.append(current)
        i += 1
    return merged, merges


def process_grid(tg_path: Path, sonorants: set[str]) -> dict:
    """Validate one annotated grid and, if valid, extract its token data.

    Returns {"error": str} on any validation failure, or on success
    {"n_tokens": int, "n_creaky": int, "hand_creak_proportion": float,
     "creaky_relative_positions": list[float], "repairs": list} (relative
     position in [0, 1] of each creaky token's midpoint within the
     utterance; repairs is the list of spurious-boundary merges performed,
     empty if the grid needed none)."""
    if not tg_path.exists():
        return {"error": f"{tg_path.name}: file not found"}

    tg = ptg.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if "creak" not in tg.tierNames:
        return {"error": f"{tg_path.name}: no 'creak' tier (tiers: {tg.tierNames})"}
    if "phones" not in tg.tierNames:
        return {"error": f"{tg_path.name}: no 'phones' tier (tiers: {tg.tierNames})"}
    creak_tier = tg.getTier("creak")
    phones_tier = tg.getTier("phones")

    expected_tier = grids.build_creak_tier(phones_tier, sonorants)
    creak_entries = list(creak_tier.entries)
    repairs: list[tuple[float, str]] = []
    if not _boundaries_match(expected_tier.entries, creak_entries):
        try:
            creak_entries, repairs = repair_extra_gap_boundaries(
                creak_entries, expected_tier.entries)
        except ValueError as exc:
            return {"error": f"{tg_path.name}: creak tier boundaries no longer match "
                              f"the sonorant intervals re-derived from its phone tier "
                              f"({exc})"}
        if not _boundaries_match(expected_tier.entries, creak_entries):
            return {"error": f"{tg_path.name}: creak tier boundaries no longer match "
                              f"the sonorant intervals re-derived from its phone tier "
                              f"(merging spurious blank fragments was not enough to "
                              f"reconcile it -- a real boundary was likely moved, not "
                              f"just split)"}

    bad_labels = sorted({
        e.label.strip() for e in creak_entries if e.label.strip() not in VALID_LABELS
    })
    if bad_labels:
        return {"error": f"{tg_path.name}: unexpected creak-tier label(s) {bad_labels} "
                          f"(only \"\" and \"c\" are valid)"}

    token_spans = grids.sonorant_spans(phones_tier, sonorants)
    tokens = [
        e for e in creak_entries
        if any(abs(e.start - s) <= BOUNDARY_TOL_S and abs(e.end - t) <= BOUNDARY_TOL_S
               for s, t in token_spans)
    ]
    if not tokens:
        return {"error": f"{tg_path.name}: zero tokens on the creak tier"}

    n_tokens = len(tokens)
    n_creaky = sum(1 for e in tokens if e.label.strip() == "c")
    total_duration = sum(e.end - e.start for e in tokens)
    creaky_duration = sum(e.end - e.start for e in tokens if e.label.strip() == "c")

    utt_start, utt_end = phones_tier.minTimestamp, phones_tier.maxTimestamp
    utt_span = utt_end - utt_start
    creaky_relative_positions = [
        ((e.start + e.end) / 2 - utt_start) / utt_span
        for e in tokens if e.label.strip() == "c" and utt_span > 0
    ]

    return {
        "n_tokens": n_tokens,
        "n_creaky": n_creaky,
        "hand_creak_proportion": creaky_duration / total_duration,
        "creaky_relative_positions": creaky_relative_positions,
        "repairs": repairs,
        "tokens": [
            {"start": e.start, "end": e.end, "label": e.label.strip()}
            for e in tokens
        ],
    }


def _summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"n": a.size, "mean": float(np.mean(a)), "median": float(np.median(a)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def print_distribution_report(results_by_utt_id: dict, pass_by_utt_id: dict) -> None:
    all_props = [r["hand_creak_proportion"] for r in results_by_utt_id.values()]
    total_tokens = sum(r["n_tokens"] for r in results_by_utt_id.values())
    total_creaky = sum(r["n_creaky"] for r in results_by_utt_id.values())

    print(f"\n=== hand_creak_proportion distribution ({len(all_props)} utterances, "
          f"{total_tokens} tokens, {total_creaky} creaky) ===")
    s = _summarize(all_props)
    print(f"  overall: n={s['n']} mean={s['mean']:.3f} median={s['median']:.3f} "
          f"min={s['min']:.3f} max={s['max']:.3f}")

    by_pass: dict[str, list[float]] = {}
    positions_by_pass: dict[str, list[float]] = {}
    for utt_id, r in results_by_utt_id.items():
        p = pass_by_utt_id[utt_id]
        by_pass.setdefault(p, []).append(r["hand_creak_proportion"])
        positions_by_pass.setdefault(p, []).extend(r["creaky_relative_positions"])

    for p in sorted(by_pass):
        s = _summarize(by_pass[p])
        pos = positions_by_pass.get(p, [])
        pos_str = f"{np.mean(pos):.2f}" if pos else "n/a (no creaky tokens)"
        print(f"  {p}: n={s['n']} mean={s['mean']:.3f} median={s['median']:.3f} "
              f"min={s['min']:.3f} max={s['max']:.3f}  "
              f"(mean relative position of creaky tokens: {pos_str}, "
              f"0=start 1=end of utterance)")


def pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_scatter(hand: list[float], phonpipe_rate: list[float], passes: list[str],
                  r: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 7))
    colors = {"A_modal": "tab:blue", "B_natural": "tab:orange", "C_creak": "tab:red"}
    for p in sorted(set(passes)):
        xs = [x for x, pp in zip(phonpipe_rate, passes) if pp == p]
        ys = [y for y, pp in zip(hand, passes) if pp == p]
        ax.scatter(xs, ys, label=p, color=colors.get(p, "gray"), alpha=0.7)
    ax.set_xlabel("phonpipe creak_doubling_rate (v3)")
    ax.set_ylabel("hand_creak_proportion")
    ax.set_title(f"Hand annotation vs phonpipe\n(pre-tuning agreement -- diagnostic, "
                 f"not a target)\nPearson r = {r:.3f}, n={len(hand)}", fontsize=11)
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_hand_annotation_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["utt_id", "hand_creak_proportion", "n_tokens", "n_creaky", "pass"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-csv", type=Path, default=Path("results/annotation_key.csv"))
    parser.add_argument("--sonorants-yaml", type=Path, default=Path("config/sonorants.yaml"))
    parser.add_argument("--grids-dir", type=Path, default=Path("data/textgrids/annotation"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("results/manifest.csv"))
    parser.add_argument("--acoustics-joined", type=Path, default=Path("results/acoustics_joined.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/hand_annotation.csv"))
    parser.add_argument("--scatter-out", type=Path, default=Path("results/annotation_vs_phonpipe.png"))
    args = parser.parse_args()

    if not args.key_csv.exists():
        sys.exit(f"{args.key_csv} not found -- run src/08b_prepare_grids.py first")
    key_rows = load_annotation_key(args.key_csv)
    if len(key_rows) != 60:
        sys.exit(f"{args.key_csv} has {len(key_rows)} row(s), expected 60")

    expected_filenames = {row["grid_filename"] for row in key_rows}
    actual_filenames = {p.name for p in args.grids_dir.glob("*.TextGrid")}
    missing = sorted(expected_filenames - actual_filenames)
    unexpected = sorted(actual_filenames - expected_filenames)
    problems = []
    if len(actual_filenames) < 60:
        problems.append(
            f"only {len(actual_filenames)} grid(s) found under {args.grids_dir}, expected 60")
    if missing:
        problems.append(f"missing expected grid file(s): {missing}")
    if unexpected:
        problems.append(f"unexpected extra grid file(s) present: {unexpected}")

    sonorants = grids.load_sonorants(args.sonorants_yaml)

    results_by_utt_id: dict[str, dict] = {}
    all_repairs: list[tuple[str, float, str]] = []
    for row in key_rows:
        tg_path = args.grids_dir / row["grid_filename"]
        result = process_grid(tg_path, sonorants)
        if "error" in result:
            problems.append(result["error"])
        else:
            results_by_utt_id[row["utt_id"]] = result
            for t, label in result["repairs"]:
                all_repairs.append((row["grid_filename"], t, label))

    if all_repairs:
        print(f"{len(all_repairs)} spurious blank boundary(ies) merged across "
              f"{len({g for g, _, _ in all_repairs})} grid(s) (repaired in memory only -- "
              f"the TextGrid files on disk are unchanged):")
        for grid_filename, t, label in all_repairs:
            print(f"  - {grid_filename} @ t={t:.3f}s (label {label!r})")

    if problems:
        print(f"{len(problems)} problem(s) found -- refusing to write {args.out}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    pass_by_utt_id = load_pass_by_utt_id(args.manifest_csv)
    missing_pass = [u for u in results_by_utt_id if u not in pass_by_utt_id]
    if missing_pass:
        sys.exit(f"{len(missing_pass)} utt_id(s) not found in {args.manifest_csv}: {missing_pass}")

    rows = [
        {
            "utt_id": utt_id,
            "hand_creak_proportion": r["hand_creak_proportion"],
            "n_tokens": r["n_tokens"],
            "n_creaky": r["n_creaky"],
            "pass": pass_by_utt_id[utt_id],
        }
        for utt_id, r in results_by_utt_id.items()
    ]
    rows.sort(key=lambda r: r["utt_id"])
    write_hand_annotation_csv(rows, args.out)
    print(f"Wrote {len(rows)} row(s) to {args.out}")

    print_distribution_report(results_by_utt_id, pass_by_utt_id)

    phonpipe_rate_by_utt_id = load_phonpipe_creak_rate(args.acoustics_joined)
    missing_phonpipe = [u for u in results_by_utt_id if u not in phonpipe_rate_by_utt_id]
    if missing_phonpipe:
        sys.exit(f"{len(missing_phonpipe)} utt_id(s) not found in {args.acoustics_joined} "
                 f"(or have no usable creak_doubling_rate): {missing_phonpipe}")

    utt_ids = sorted(results_by_utt_id)
    hand = [results_by_utt_id[u]["hand_creak_proportion"] for u in utt_ids]
    phonpipe_rate = [phonpipe_rate_by_utt_id[u] for u in utt_ids]
    passes = [pass_by_utt_id[u] for u in utt_ids]

    r = pearson_correlation(hand, phonpipe_rate)
    print(f"\n=== hand vs phonpipe (v3 creak_doubling_rate) pre-tuning agreement ===")
    print(f"  Pearson r = {r:.3f} (n={len(hand)})")
    print(f"  This is a diagnostic, not a target -- threshold tuning not run here.")

    write_scatter(hand, phonpipe_rate, passes, r, args.scatter_out)
    print(f"  Wrote scatter to {args.scatter_out}")


if __name__ == "__main__":
    main()
