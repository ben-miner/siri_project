"""Compare phonpipe measurements before and after the adaptive-range
period-doubling fix (phonpipe commit d2ab07b -- see README's "Dependency
versions"), across every phonpipe-derived measure in acoustics_joined.csv.

v1 = results/acoustics_joined_v1.csv (archived pre-fix run, --force
    excluded the 60 rows that tripped f0_tracking_failed at the time, so
    those utt_ids are absent from v1 entirely -- not silently misread as
    "unchanged", reported as only_in_v2)
v2 = results/acoustics_joined.csv (re-measured post-fix with --keep-flagged,
    so all 600 rows are present regardless of f0_quality_flag)

The set of "measures" compared is derived from v1's own columns minus the
10 manifest columns (utt_id, session, pass, ...) and "file" (an echoed
filename, not a measurement) -- 66 columns on this project's data. Whatever
new columns v2 added (e.g. f0_tracking_failed, which didn't exist when v1
was measured) are deliberately excluded from the comparison for the same
reason "file" is: comparing a column against itself, or against a column
that has no v1 counterpart to differ from, isn't a version diff.

Each column is compared as NUMERIC if both values parse as float, else as
TEXT (covers f0_quality_flag, creak_subtype_hint, boolean-as-string columns
like creak_sonorant_restricted, etc.) -- type is detected per-value, not
hardcoded per-column, so nothing gets silently skipped if a column's type
isn't on a maintained list.

SIGN CHANGES are flagged separately from ordinary magnitude changes: this
is the specific, diagnostic signature of the bug just fixed (H1-H2 flipping
from creaky/negative to breathy/positive when the wrong floor/ceiling was
used -- see phonpipe README's "Known limitation" section). A sign change
requires strictly opposite signs on both sides (v1 or v2 exactly 0.0 is
never counted as a sign change in either direction).

Writes:
  results/version_diff.csv        -- every (utt_id, measure) pair, changed
                                      or not, with a changed/sign_changed
                                      column to filter on
  results/version_diff_signs.csv  -- just the sign-changed subset, for
                                      quick inspection without filtering
"""

import argparse
import csv
import math
from pathlib import Path

MANIFEST_COLUMNS = frozenset({
    "utt_id", "session", "pass", "item_id", "item_type",
    "script_text", "verbatim_text", "was_corrected", "wav_path", "duration_sec",
})
EXCLUDED_FROM_DIFF = frozenset({"file"})  # identifier echo, not a measurement


def load_rows(csv_path: Path) -> dict:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in by_utt_id:
            raise ValueError(f"duplicate utt_id '{utt_id}' in {csv_path}")
        by_utt_id[utt_id] = row
    return by_utt_id


def measure_columns(v1_rows: dict) -> list[str]:
    if not v1_rows:
        return []
    any_row = next(iter(v1_rows.values()))
    return [
        c for c in any_row.keys()
        if c not in MANIFEST_COLUMNS and c not in EXCLUDED_FROM_DIFF
    ]


def _try_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
        return f if f == f else None  # NaN -> None (treat as "no value")
    except (TypeError, ValueError):
        return None


def compare_value(v1_raw, v2_raw) -> dict:
    v1_num, v2_num = _try_float(v1_raw), _try_float(v2_raw)
    if v1_num is not None and v2_num is not None:
        delta = v2_num - v1_num
        if v1_num != 0:
            pct_change = delta / abs(v1_num) * 100
        else:
            pct_change = math.inf if delta != 0 else 0.0
        sign_changed = (v1_num > 0 and v2_num < 0) or (v1_num < 0 and v2_num > 0)
        return {
            "kind": "numeric", "v1": v1_num, "v2": v2_num, "delta": delta,
            "pct_change": pct_change, "sign_changed": sign_changed,
            "changed": v1_num != v2_num,
        }
    v1_norm, v2_norm = (v1_raw or "").strip(), (v2_raw or "").strip()
    if v1_norm.upper() in ("TRUE", "FALSE") and v2_norm.upper() in ("TRUE", "FALSE"):
        # Case-insensitive: results/acoustics_joined_v1.csv was archived after
        # a stage that round-tripped it through Excel, which uppercases
        # TRUE/FALSE -- comparing case-sensitively against Python's str(bool)
        # ("True"/"False") would report every single boolean column as
        # changed. Verified on real data: creak_sonorant_restricted showed
        # 540/540 "changed" before this fix, 0/540 after -- a formatting
        # artifact, not a measurement difference.
        changed = v1_norm.upper() != v2_norm.upper()
    else:
        changed = v1_norm != v2_norm
    return {
        "kind": "text", "v1": v1_raw, "v2": v2_raw, "delta": None,
        "pct_change": None, "sign_changed": False, "changed": changed,
    }


def diff_utt_id(utt_id: str, v1_row: dict, v2_row: dict, columns: list[str]) -> list[dict]:
    return [
        {"utt_id": utt_id, "measure": col, **compare_value(v1_row.get(col), v2_row.get(col))}
        for col in columns
    ]


def build_diff(v1_rows: dict, v2_rows: dict, columns: list[str]) -> dict:
    v1_ids, v2_ids = set(v1_rows), set(v2_rows)
    only_v1 = sorted(v1_ids - v2_ids)
    only_v2 = sorted(v2_ids - v1_ids)
    common = sorted(v1_ids & v2_ids)

    diffs = []
    for utt_id in common:
        diffs.extend(diff_utt_id(utt_id, v1_rows[utt_id], v2_rows[utt_id], columns))

    return {"diffs": diffs, "only_v1": only_v1, "only_v2": only_v2, "common": common}


def write_diff_csv(diffs: list[dict], out_path: Path) -> None:
    fieldnames = ["utt_id", "measure", "kind", "v1", "v2", "delta", "pct_change",
                  "sign_changed", "changed"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: d.get(k) for k in fieldnames} for d in diffs)


def print_summary(result: dict, columns: list[str]) -> None:
    diffs = result["diffs"]
    changed = [d for d in diffs if d["changed"]]
    sign_changed = [d for d in diffs if d["sign_changed"]]
    changed_utt_ids = {d["utt_id"] for d in changed}

    print(f"Compared {len(result['common'])} utt_id(s) across {len(columns)} measure(s) "
          f"({len(diffs)} total comparisons).")
    if result["only_v1"]:
        print(f"  {len(result['only_v1'])} utt_id(s) only in v1 (excluded by --force at the "
              f"time; not compared): {result['only_v1'][:10]}"
              f"{' ...' if len(result['only_v1']) > 10 else ''}")
    if result["only_v2"]:
        print(f"  {len(result['only_v2'])} utt_id(s) only in v2: {result['only_v2'][:10]}"
              f"{' ...' if len(result['only_v2']) > 10 else ''}")

    print(f"\n{len(changed)} value(s) changed across {len(changed_utt_ids)} utt_id(s).")

    by_measure: dict[str, int] = {}
    for d in changed:
        by_measure[d["measure"]] = by_measure.get(d["measure"], 0) + 1
    print("Most frequently changed measures:")
    for measure, n in sorted(by_measure.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {measure}: {n}")

    print(f"\n{len(sign_changed)} SIGN CHANGE(S) -- the fix's diagnostic signature "
          f"(e.g. H1-H2 creaky/negative <-> breathy/positive):")
    for d in sorted(sign_changed, key=lambda d: d["utt_id"]):
        print(f"  {d['utt_id']} {d['measure']}: {d['v1']:.3f} -> {d['v2']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("results/acoustics_joined_v1.csv"))
    parser.add_argument("--v2", type=Path, default=Path("results/acoustics_joined.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/version_diff.csv"))
    parser.add_argument("--signs-out", type=Path, default=Path("results/version_diff_signs.csv"))
    args = parser.parse_args()

    v1_rows = load_rows(args.v1)
    v2_rows = load_rows(args.v2)
    if not v1_rows:
        raise SystemExit(f"{args.v1} has no rows")
    if not v2_rows:
        raise SystemExit(f"{args.v2} has no rows")

    columns = measure_columns(v1_rows)
    v1_extra = set(next(iter(v1_rows.values())).keys()) - MANIFEST_COLUMNS - EXCLUDED_FROM_DIFF
    v2_extra = set(next(iter(v2_rows.values())).keys()) - MANIFEST_COLUMNS - EXCLUDED_FROM_DIFF
    schema_only_v2 = sorted(v2_extra - v1_extra)
    if schema_only_v2:
        print(f"NOTE: {len(schema_only_v2)} column(s) in v2 not present in v1's measure set "
              f"(excluded from comparison, not a version diff): {schema_only_v2}")

    result = build_diff(v1_rows, v2_rows, columns)

    write_diff_csv(result["diffs"], args.out)
    sign_changed = [d for d in result["diffs"] if d["sign_changed"]]
    write_diff_csv(sign_changed, args.signs_out)

    print_summary(result, columns)
    print(f"\nWrote {len(result['diffs'])} row(s) to {args.out}")
    print(f"Wrote {len(sign_changed)} row(s) to {args.signs_out}")


if __name__ == "__main__":
    main()
