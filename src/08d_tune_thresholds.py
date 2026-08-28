"""Grid-search per-token creak thresholds against hand annotation.

NAMING NOTE: requested as "08_tune_thresholds.py"; named 08d here to match
the existing 08a/08b/08c lettered sequence for this stage. Rename if you
actually wanted the bare "08_" prefix.

ARCHITECTURE GAP THIS SCRIPT WORKS AROUND: phonpipe's CREAK_THRESHOLDS
(phonpipe/measures/creak.py) has 3 keys, but only "shr_doubling" actually
drives a per-token-shaped decision anywhere in phonpipe's own code
(creak_components()'s "multiply_pulsed" hint). "low_f0_percentile" is
defined but never read anywhere else in phonpipe -- dead configuration.
"shr_octave_risk" drives a DIFFERENT thing (shr_octave_risk_rate / the
period_doubling_present flag, an F0-tracking-quality signal, not a creak
verdict) and isn't part of this tuning. Worse: none of phonpipe's 4 creak
components are computed at true per-TOKEN granularity except H1-H2 (and
even that's per-VOWEL-token specifically, via extract.py's phone loop --
narrower than the per-SONORANT tokens the "creak" tier / hand annotation
covers). SHR, F0, and jitter are each computed exactly once per whole
utterance (measure_shr, f0_stats["f0_p10_hz"], jitter_local_pct).

So this script computes genuine per-token values for all 4 components by
calling phonpipe's EXISTING measurement functions at each token's own
(start, end) span, not the whole-utterance span extract.py normally uses:
  - H1-H2:      measure_tilt(snd, t_start, t_end, ...) -- already takes a
                span; no new code needed, just called at every token
                instead of only vowel phones.
  - SHR:        measure_shr(snd, floor, ceiling, spans=[(t_start, t_end)])
                -- already supports span restriction.
  - low F0:     no per-span measure_f0() variant exists (and phonpipe's
                own README documents its full adaptive-range machinery
                failing on spans this short -- the same problem
                07_calibration.py's compute_constrained_f0() works around
                for creak_cal clips). Same fix here: crop the sound to the
                token's span, then a DIRECT (non-adaptive) to_pitch_ac()
                call using the utterance's own f0_floor_used/ceiling_used
                as bounds (already correctly derived per-utterance by
                phonpipe's adaptive-range fix -- reused, not re-derived).
                "Low" is inherently relative ("below this pct of speaker's
                own F0" -- CREAK_THRESHOLDS's own comment), so each
                token's F0 is compared against a percentile of this
                speaker's own A_modal utterances' f0_median_hz (the
                plainest available reference for "typical modal voice"),
                and the percentile cutoff is what's SEARCHED. What gets
                EMITTED, though, is the resulting Hz cutoff (low_f0_hz),
                not the percentile: on the real dataset F1 is tied across
                a whole plateau of percentiles that collapse to nearly the
                same Hz value (a resolution limit of a 200-value reference
                set, not an open search), so the percentile that currently
                produces the winning Hz value is an artifact of this
                run's reference-set size and would drift if that set's
                size changed, while the Hz cutoff itself would not.
  - jitter:     no per-span variant either; same crop, then
                measure_voice_quality_global() on the cropped snippet.

Each of the 4 components is tuned INDEPENDENTLY to maximise its own F1 --
matching creak_components()'s own stated philosophy of reporting separate
components rather than one combined score, since creak subtypes differ in
which components are present. No combined/ensemble classifier is computed.

Two of the four components (H1-H2, jitter) have NO existing slot in
CREAK_THRESHOLDS at all. jitter_irregularity_pct is added as a new key.
h1_h2_db is NOT emitted as an active threshold: its tuned F1 lands at the
"always predict creaky" trivial baseline (recall 1.000), and the majority
of tokens return NaN (measure_tilt() fails on short spans) -- per-token
H1-H2 does not classify with this measurement methodology. It's printed as
a commented-out line with its measured F1 and NaN rate so the negative
result stays on record instead of silently disappearing. shr_octave_risk
is excluded entirely: it doesn't drive a per-token decision anywhere in
phonpipe's current code, so there's nothing here to tune it against.

Print only -- writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import parselmouth
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
grids = import_module("08b_prepare_grids")       # load_sonorants
compute = import_module("08c_compute_proportions")  # process_grid, load_annotation_key

PASSES = ("A_modal", "B_natural", "C_creak")


def _to_float(value):
    try:
        v = float(value)
        return v if v == v else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def load_acoustics_by_utt_id(csv_path: Path) -> dict[str, dict]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_utt_id = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in by_utt_id:
            raise ValueError(f"duplicate utt_id '{utt_id}' in {csv_path}")
        by_utt_id[utt_id] = row
    return by_utt_id


def nan_rate(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v != v) / len(values)


def load_speaker_modal_f0(acoustics_by_utt_id: dict[str, dict]) -> list[float]:
    """A_modal utterances' f0_median_hz -- the reference distribution for
    'speaker's own F0', used to define what counts as a LOW F0 token."""
    values = []
    for row in acoustics_by_utt_id.values():
        if row.get("pass") != "A_modal":
            continue
        v = _to_float(row.get("f0_median_hz"))
        if v is not None:
            values.append(v)
    return values


# -- per-token measurement -----------------------------------------------------------

def measure_token_h1_h2(snd, t_start: float, t_end: float, floor: float, ceiling: float,
                        max_formant: float) -> float:
    from phonpipe.measures.voice_quality import measure_tilt
    try:
        return measure_tilt(snd, t_start, t_end, floor, ceiling,
                            max_formant=max_formant)["h1_h2_db"]
    except Exception:
        return float("nan")


def measure_token_shr(snd, t_start: float, t_end: float, floor: float, ceiling: float) -> float:
    from phonpipe.measures.creak import measure_shr
    try:
        return measure_shr(snd, floor, ceiling, spans=[(t_start, t_end)])["shr_median"]
    except Exception:
        return float("nan")


def measure_token_f0(snd, t_start: float, t_end: float, floor: float, ceiling: float) -> float:
    try:
        cropped = snd.extract_part(t_start, t_end)
        pitch = cropped.to_pitch_ac(pitch_floor=floor, pitch_ceiling=ceiling)
        freqs = pitch.selected_array["frequency"]
        voiced = freqs[freqs > 0]
        if voiced.size == 0:
            return float("nan")
        return float(np.median(voiced))
    except Exception:
        return float("nan")


def measure_token_jitter(snd, t_start: float, t_end: float, floor: float, ceiling: float) -> float:
    from phonpipe.measures.voice_quality import measure_voice_quality_global
    try:
        cropped = snd.extract_part(t_start, t_end)
        return measure_voice_quality_global(cropped, floor, ceiling)["jitter_local_pct"]
    except Exception:
        return float("nan")


def build_token_dataset(key_rows: list[dict], acoustics_by_utt_id: dict[str, dict],
                        sonorants: set[str], grids_dir: Path) -> tuple[list[dict], list[str]]:
    """Returns (token_rows, problems). token_rows is empty if problems is
    non-empty -- mirrors 08c's fail-loud pattern (collect every problem,
    refuse to proceed with a partial/silently-shrunk dataset)."""
    problems = []
    per_grid: dict[str, dict] = {}
    for row in key_rows:
        utt_id = row["utt_id"]
        tg_path = grids_dir / row["grid_filename"]
        result = compute.process_grid(tg_path, sonorants)
        if "error" in result:
            problems.append(result["error"])
            continue
        acc = acoustics_by_utt_id.get(utt_id)
        if acc is None:
            problems.append(f"{utt_id}: not found in acoustics_joined.csv")
            continue
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            problems.append(f"{utt_id}: wav not found at {wav_path}")
            continue
        per_grid[utt_id] = {"result": result, "acc": acc, "wav_path": wav_path}

    if problems:
        return [], problems

    token_rows = []
    for utt_id, info in per_grid.items():
        acc = info["acc"]
        floor = _to_float(acc.get("f0_floor_used"))
        ceiling = _to_float(acc.get("f0_ceiling_used"))
        max_formant = _to_float(acc.get("formant_ceiling_hz")) or 5500.0
        if floor is None or ceiling is None:
            problems.append(f"{utt_id}: missing f0_floor_used/f0_ceiling_used")
            continue
        snd = parselmouth.Sound(str(info["wav_path"]))
        pass_name = acc["pass"]
        for tok in info["result"]["tokens"]:
            t_start, t_end = tok["start"], tok["end"]
            token_rows.append({
                "utt_id": utt_id,
                "pass": pass_name,
                "hand_creaky": 1 if tok["label"] == "c" else 0,
                "h1_h2_db": measure_token_h1_h2(snd, t_start, t_end, floor, ceiling, max_formant),
                "shr": measure_token_shr(snd, t_start, t_end, floor, ceiling),
                "f0_hz": measure_token_f0(snd, t_start, t_end, floor, ceiling),
                "jitter_pct": measure_token_jitter(snd, t_start, t_end, floor, ceiling),
            })

    return token_rows, problems


# -- evaluation -----------------------------------------------------------

def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def trivial_classifier_f1(labels: list[int]) -> float:
    """F1 of the classifier that ignores the signal and always predicts
    creaky -- the best a totally uninformative component can score. A
    tuned component whose F1 lands at or near this isn't discriminating
    at all, whatever its threshold looks like (this catches the h1_h2_db
    case found on the real 60-grid dataset: F1 climbs monotonically all
    the way to the always-creaky limit, well before hitting either grid
    edge, so grid-edge detection alone misses it)."""
    if not labels:
        return 0.0
    precision = sum(labels) / len(labels)  # recall is 1.0 by construction
    return 2 * precision / (1 + precision) if precision > 0 else 0.0


def grid_search_threshold(values: list[float], labels: list[int], grid, direction: str):
    """direction '>' predicts creaky when value > threshold, '<' when value
    < threshold. Returns None if no non-NaN values exist, else a dict with
    the F1-maximising threshold and its precision/recall/F1/n."""
    paired = [(v, l) for v, l in zip(values, labels) if v == v]
    if not paired:
        return None
    vs, ls = zip(*paired)
    best = None
    for t in grid:
        preds = [1 if (v > t if direction == ">" else v < t) else 0 for v in vs]
        p, r, f1 = precision_recall_f1(list(ls), preds)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "precision": p, "recall": r, "f1": f1, "n": len(vs)}
    return best


def grid_search_low_f0_percentile(token_f0s: list[float], labels: list[int],
                                  reference_f0s: list[float], percentile_grid):
    """'Low F0' is relative: creaky if token F0 <= the Nth percentile of
    the speaker's own A_modal F0 distribution. N is what's SEARCHED here,
    matching CREAK_THRESHOLDS's low_f0_percentile -- but see
    print_winning_config_yaml: what's ultimately EMITTED as the frozen
    threshold is the resulting Hz cutoff, not N, because N is unstable
    under changes to the reference set's size/composition while the Hz
    cutoff a given plateau of N values maps to is not.

    Also returns plateau_pct_max/plateau_hz_max/n_reference: the
    contiguous run of grid points starting at the winning (lowest)
    percentile whose F1 ties the winner. A wide plateau means the
    reference distribution doesn't have enough resolution at this
    percentile range to distinguish those cutoffs -- not that the search
    is still climbing."""
    paired = [(v, l) for v, l in zip(token_f0s, labels) if v == v]
    if not paired or not reference_f0s:
        return None
    vs, ls = zip(*paired)
    scored = []
    best = None
    for pct in percentile_grid:
        cutoff_hz = float(np.percentile(reference_f0s, pct))
        preds = [1 if v <= cutoff_hz else 0 for v in vs]
        p, r, f1 = precision_recall_f1(list(ls), preds)
        row = {"pct": float(pct), "cutoff_hz": cutoff_hz, "precision": p, "recall": r, "f1": f1}
        scored.append(row)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(pct), "cutoff_hz": cutoff_hz,
                    "precision": p, "recall": r, "f1": f1, "n": len(vs)}

    scored.sort(key=lambda row: row["pct"])
    plateau = []
    for row in scored:
        if row["pct"] < best["threshold"]:
            continue
        if abs(row["f1"] - best["f1"]) < 1e-9:
            plateau.append(row)
        elif plateau:
            break
    best["plateau_pct_max"] = plateau[-1]["pct"]
    best["plateau_hz_max"] = plateau[-1]["cutoff_hz"]
    best["n_reference"] = len(reference_f0s)
    return best


# Grids widened beyond what any of these signals plausibly need, verified
# by hand against the real 60-grid dataset before settling on these ranges:
# SHR (0.38) and jitter (2.3) both land on a genuine interior peak with a
# clear precision/recall tradeoff on either side. h1_h2_db does NOT --
# every widening tried (checked out to -30/+40) kept climbing toward the
# trivial "always predict creaky" limit, meaning per-token H1-H2 (unlike
# its utterance-level MEAN, which correlates r=0.772 with phonpipe's
# creak_doubling_rate -- see 08c) has ~no discriminative power on its own,
# most likely because a single token's spectral tilt estimate is much
# noisier than an utterance-level average across many tokens, compounded
# by measure_tilt() failing outright (NaN) on the majority of tokens under
# ~60ms. It is excluded from the emitted config entirely (see
# print_winning_config_yaml) rather than tuned to a number that looks like
# a real threshold but isn't.
#
# low_f0_percentile: widened down to 0.01% (from an original 1% floor).
# F1 still registers at that new minimum, but hand-checking percentiles
# down to 0.001% shows F1 flat at 0.470 across 0.001%-0.02%, then
# declining as the percentile grows past that -- it HAS converged, it just
# converged onto a plateau that happens to sit at the grid's edge. The
# plateau itself is a resolution limit, not an open-ended optimum:
# reference_f0s has only 200 A_modal values, so below the ~0.5th
# percentile (1-in-200) there's no additional data for np.percentile to
# resolve a finer cutoff_hz (108.58Hz at 0.001% vs. 108.62Hz at 0.01%).
# Recall barely moves across the whole percentile range (0.60-0.63),
# meaning a substantial share of hand-labeled creaky tokens simply aren't
# low-F0 by any cutoff, consistent with Keating et al. (2023): not every
# creak subtype shows every component. The "AT GRID EDGE" warning below
# still fires on this (technically true -- the threshold IS the grid
# minimum) even though it's converged.
#
# Because the plateau is a reference-resolution artifact rather than a
# real optimum, the PERCENTILE itself is not a meaningful thing to freeze
# -- it would land somewhere else on the same plateau if the A_modal
# reference set changed size, without the underlying Hz cutoff moving.
# So low_f0_hz (the Hz cutoff), not low_f0_percentile, is what
# print_winning_config_yaml emits as the active threshold; the percentile
# search is kept only as the mechanism that produced it, documented in a
# comment alongside the emitted value.
COMPONENT_GRIDS = {
    "shr_doubling": (np.arange(0.10, 0.71, 0.01), ">"),
    "h1_h2_db": (np.arange(-30.0, 40.01, 0.5), "<"),
    "jitter_irregularity_pct": (np.arange(0.0, 30.01, 0.1), ">"),
}
COMPONENT_VALUE_KEY = {
    "shr_doubling": "shr",
    "h1_h2_db": "h1_h2_db",
    "jitter_irregularity_pct": "jitter_pct",
}
LOW_F0_PERCENTILE_GRID = [round(p, 2) for p in np.arange(0.01, 2.0, 0.01)] + list(range(2, 51))


def _is_at_grid_edge(threshold: float, grid, tol: float = 1e-6) -> bool:
    values = np.asarray(list(grid), dtype=float)
    return bool(abs(threshold - values.min()) < tol or abs(threshold - values.max()) < tol)


def tune_all_components(token_rows: list[dict], reference_f0s: list[float]) -> dict:
    labels = [r["hand_creaky"] for r in token_rows]
    results = {}
    for name, (grid, direction) in COMPONENT_GRIDS.items():
        values = [r[COMPONENT_VALUE_KEY[name]] for r in token_rows]
        best = grid_search_threshold(values, labels, grid, direction)
        if best is not None:
            valid_labels = [l for v, l in zip(values, labels) if v == v]
            best["trivial_f1"] = trivial_classifier_f1(valid_labels)
        results[name] = best
    low_f0_values = [r["f0_hz"] for r in token_rows]
    best = grid_search_low_f0_percentile(low_f0_values, labels, reference_f0s,
                                          LOW_F0_PERCENTILE_GRID)
    if best is not None:
        valid_labels = [l for v, l in zip(low_f0_values, labels) if v == v]
        best["trivial_f1"] = trivial_classifier_f1(valid_labels)
    results["low_f0_percentile"] = best
    return results


def predict_component(name: str, value: float, best: dict, reference_f0s: list[float]) -> int | None:
    if value != value:  # NaN
        return None
    if name == "low_f0_percentile":
        return 1 if value <= best["cutoff_hz"] else 0
    direction = COMPONENT_GRIDS[name][1]
    t = best["threshold"]
    return 1 if (value > t if direction == ">" else value < t) else 0


def evaluate_by_pass(token_rows: list[dict], results: dict, reference_f0s: list[float]) -> dict:
    value_key = {**COMPONENT_VALUE_KEY, "low_f0_percentile": "f0_hz"}
    by_pass_component: dict[tuple[str, str], dict] = {}
    for pass_name in PASSES:
        subset = [r for r in token_rows if r["pass"] == pass_name]
        for name, best in results.items():
            if best is None:
                continue
            vs = [r[value_key[name]] for r in subset]
            ls = [r["hand_creaky"] for r in subset]
            paired = [
                (predict_component(name, v, best, reference_f0s), l)
                for v, l in zip(vs, ls) if v == v
            ]
            if not paired:
                by_pass_component[(pass_name, name)] = None
                continue
            preds, ls2 = zip(*paired)
            p, r, f1 = precision_recall_f1(list(ls2), list(preds))
            by_pass_component[(pass_name, name)] = {
                "precision": p, "recall": r, "f1": f1, "n": len(preds)}
    return by_pass_component


def print_winning_config_yaml(results: dict, h1_h2_nan_rate: float) -> None:
    low_f0 = results.get("low_f0_percentile")
    config = {
        "shr_doubling": round(results["shr_doubling"]["threshold"], 4)
                        if results["shr_doubling"] else None,
        "low_f0_hz": round(low_f0["cutoff_hz"], 1) if low_f0 else None,
        "jitter_irregularity_pct": round(results["jitter_irregularity_pct"]["threshold"], 2)
                                   if results["jitter_irregularity_pct"] else None,
    }
    print("\n=== Winning config (paste into config/thresholds.yaml) ===")
    print("# jitter_irregularity_pct is a NEW key -- phonpipe's CREAK_THRESHOLDS")
    print("# currently has no slot for it. shr_octave_risk is not included: it")
    print("# doesn't drive a per-token decision anywhere in phonpipe's current")
    print("# code, so there's nothing here to tune it against.")
    if low_f0 is not None:
        print(f"# low_f0_hz replaces CREAK_THRESHOLDS's low_f0_percentile. The search still")
        print(f"#   sweeps a percentile of this speaker's A_modal f0_median_hz distribution")
        print(f"#   (n={low_f0['n_reference']}) -- winner {low_f0['threshold']:.2f}%ile -- but the percentile is a")
        print(f"#   derived artifact of the Hz cutoff, not the reverse: F1 is tied (={low_f0['f1']:.3f})")
        print(f"#   for every grid point from {low_f0['threshold']:.2f}%ile to {low_f0['plateau_pct_max']:.2f}%ile "
              f"({low_f0['cutoff_hz']:.2f}-{low_f0['plateau_hz_max']:.2f}Hz),")
        print(f"#   and hand-checking below the grid floor (down to 0.001%ile) held flat at the")
        print(f"#   same ~{low_f0['cutoff_hz']:.1f}Hz too -- {low_f0['n_reference']} reference values just can't resolve a finer")
        print(f"#   percentile than that. The Hz cutoff is stable under this; the percentile that")
        print(f"#   produces it would shift if the reference set's size changed. Freezing the Hz value.")
    h1_h2 = results.get("h1_h2_db")
    if h1_h2 is not None:
        print(f"# h1_h2_db: {h1_h2['threshold']:.2f}  # NOT emitted as an active threshold.")
        print(f"#   F1 {h1_h2['f1']:.3f} == trivial 'always predict creaky' baseline "
              f"{h1_h2['trivial_f1']:.3f} (recall {h1_h2['recall']:.3f}, precision "
              f"{h1_h2['precision']:.3f}); {h1_h2_nan_rate:.1%} of tokens returned NaN")
        print(f"#   (measure_tilt fails on short spans). Per-token H1-H2 does not classify")
        print(f"#   with this methodology -- kept here as a negative result, not a threshold.")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))


DEGENERATE_F1_GAP = 0.01  # winning F1 within this of the trivial baseline -> not discriminating


def print_component_report(results: dict) -> None:
    print("=== Per-component token-level F1 (independently tuned; no combined score) ===")
    header = (f"{'component':28} {'threshold':>16} {'precision':>10} {'recall':>10} "
              f"{'f1':>8} {'trivial_f1':>10} {'n':>6}")
    print(header)
    edge_components, degenerate_components = [], []
    for name, best in results.items():
        if best is None:
            print(f"{name:28} {'(no valid tokens)':>16}")
            continue
        if name == "low_f0_percentile":
            thr_str = f"{best['threshold']:.1f}%ile ({best['cutoff_hz']:.1f}Hz)"
            grid = LOW_F0_PERCENTILE_GRID
        else:
            thr_str = f"{best['threshold']:.2f}"
            grid = COMPONENT_GRIDS[name][0]
        flags = []
        if _is_at_grid_edge(best["threshold"], grid):
            edge_components.append(name)
            flags.append("AT GRID EDGE")
        if best["f1"] - best["trivial_f1"] < DEGENERATE_F1_GAP:
            degenerate_components.append(name)
            flags.append("NOT DISCRIMINATING")
        flag_str = f" <-- {', '.join(flags)}" if flags else ""
        print(f"{name:28} {thr_str:>16} {best['precision']:>10.3f} {best['recall']:>10.3f} "
              f"{best['f1']:>8.3f} {best['trivial_f1']:>10.3f} {best['n']:>6}{flag_str}")

    if edge_components:
        print(f"\nWARNING: {edge_components} landed at the edge of their search grid -- "
              f"the true F1-maximising threshold may lie beyond what was searched.")
    if degenerate_components:
        print(f"\nWARNING: {degenerate_components} scored within {DEGENERATE_F1_GAP} F1 of "
              f"'trivial_f1' -- the F1 a classifier gets by ignoring the signal entirely and "
              f"always predicting creaky. Their 'winning' threshold is not meaningfully "
              f"discriminating; don't treat it as a real finding (h1_h2_db does this on the "
              f"real 60-grid dataset -- see module docstring for why).")


def print_pass_breakdown(by_pass_component: dict) -> None:
    print("\n=== Recall by pass (using each component's overall-tuned threshold) ===")
    print("# Recall, not F1: F1 on A_modal's small positive-token count is mostly")
    print("# prevalence noise. Recall -- what fraction of hand-labeled creaky tokens")
    print("# each component catches -- stays comparable across passes with very")
    print("# different creaky-token base rates.")
    header = f"{'component':28} {'A_modal':>12} {'B_natural':>12} {'C_creak':>12}"
    print(header)
    names = sorted({name for _, name in by_pass_component})
    for name in names:
        cells = []
        for pass_name in PASSES:
            r = by_pass_component.get((pass_name, name))
            cells.append(f"{r['recall']:.3f} (n={r['n']})" if r else "n/a")
        print(f"{name:28} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-csv", type=Path, default=Path("results/annotation_key.csv"))
    parser.add_argument("--sonorants-yaml", type=Path, default=Path("config/sonorants.yaml"))
    parser.add_argument("--grids-dir", type=Path, default=Path("data/textgrids/annotation"))
    parser.add_argument("--acoustics-joined", type=Path, default=Path("results/acoustics_joined.csv"))
    args = parser.parse_args()

    if not args.key_csv.exists():
        sys.exit(f"{args.key_csv} not found -- run src/08b_prepare_grids.py first")
    key_rows = compute.load_annotation_key(args.key_csv)
    if len(key_rows) != 60:
        sys.exit(f"{args.key_csv} has {len(key_rows)} row(s), expected 60")

    sonorants = grids.load_sonorants(args.sonorants_yaml)
    acoustics_by_utt_id = load_acoustics_by_utt_id(args.acoustics_joined)

    print(f"Measuring {len(key_rows)} grids' tokens against phonpipe primitives "
          f"(this calls Praat per token, may take a few minutes)...")
    token_rows, problems = build_token_dataset(key_rows, acoustics_by_utt_id, sonorants,
                                               args.grids_dir)
    if problems:
        print(f"{len(problems)} problem(s) found -- refusing to tune:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print(f"Built {len(token_rows)} token(s) across {len(key_rows)} utterances "
          f"({sum(r['hand_creaky'] for r in token_rows)} hand-labeled creaky).")

    reference_f0s = load_speaker_modal_f0(acoustics_by_utt_id)
    print(f"Speaker reference F0 distribution: {len(reference_f0s)} A_modal utterances.")

    results = tune_all_components(token_rows, reference_f0s)
    h1_h2_nan_rate = nan_rate([r["h1_h2_db"] for r in token_rows])
    print_winning_config_yaml(results, h1_h2_nan_rate)
    print_component_report(results)

    by_pass_component = evaluate_by_pass(token_rows, results, reference_f0s)
    print_pass_breakdown(by_pass_component)


if __name__ == "__main__":
    main()
