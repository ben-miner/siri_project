"""Primary analysis: does creaky phonation predict ASR word error rate?

Joins results/acoustics_joined.csv (v3, per-utterance acoustic measures),
results/scored.csv (Apple SFSpeechRecognizer WER), and results/wer_whisper.csv
(Whisper large-v3 WER) on utt_id into one wide dataframe, then:

  1. Checks whether roomtone RMS differs between sessions (results/
     calibration.csv) BEFORE modeling -- if recording gain/room conditions
     differ by session, intensity is not directly comparable across
     sessions, and that has to be reported, not modeled over silently.
  2. Computes two per-utterance quantities not present in any existing
     results file: intensity (via 07_calibration.py's compute_rms(), reused
     rather than reimplemented -- see that module for why RMS isn't one of
     phonpipe's own measures) and speech rate (reference word count /
     duration, i.e. words per second -- purely derived from columns already
     on hand, no new acoustic measurement needed).
  3. Fits the primary mixed-effects model: errors-per-reference-word (Apple)
     ~ shr_median + jitter_local_pct + intensity + speech_rate +
     f0_tracking_failed, with CROSSED random intercepts for item_id and
     session (statsmodels MixedLM handles crossed, non-nested random effects
     by putting both grouping factors in vc_formula under a single dummy
     `groups` -- verified against a synthetic dataset with known random-
     effect variances before trusting it on the real data; see the module's
     git history / commit message for that check).
  4. Refits the same structure with Whisper WER as the outcome and compares
     coefficients between recognizers.
  5. Runs the remaining secondary analyses (error-type breakdown, verbatim-
     vs-script WER gap, correction rate by pass, sentence-type effects
     within B_natural) and saves the shr_median-vs-WER scatter (figure1.png).

h1_h2_db_mean is EXCLUDED from every model. It failed five independent
validation checks over the course of this project: a sign inversion
relative to the physiologically expected direction, 63 physiological-
plausibility violations against the shr_median>0.45 => h1_h2<=+2dB
constraint (README Limitations), degenerate token-level threshold tuning
(F1 == the trivial "always predict creaky" baseline, Stage 8d), a 63%
per-token NaN rate from measure_tilt() failing on short spans (also Stage
8d), and the same shr_median>0.45 pattern surfacing independently in that
per-token investigation. Continuing to include it as a predictor here would
launder a measure this project has already found isn't trustworthy.

Predictors are z-scored (shr_median, jitter_local_pct, intensity, speech
rate) before fitting, so fixed-effect coefficients are on a per-1-SD-change
scale and comparable to each other despite very different native units;
f0_tracking_failed is left as 0/1 (already meaningful on its own scale).

Print-only for the results table; writes results/figure1.png (300 dpi) as
its one file output.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
calibration_mod = import_module("07_calibration")

ROOMTONE_FLAG_THRESHOLD_PCT = 15.0  # matches 07_calibration.py's own FLAG_THRESHOLD_PCT


# -- data loading and joining -----------------------------------------------------------

def load_joined_data(acoustics_path: Path, scored_path: Path, whisper_path: Path) -> pd.DataFrame:
    """Joins the three per-utterance result files on utt_id. session/pass/
    verbatim_text/script_text are duplicated across all three (all three
    ultimately derive from manifest.csv) -- kept once, from acoustics_joined.
    Whisper's un-prefixed error columns are renamed with a whisper_ prefix
    so they don't collide with scored.csv's own verbatim_*/script_* columns.
    validate="one_to_one" fails loudly on any utt_id duplicated or missing
    on either side instead of silently producing a partial/inflated join.
    """
    acoustics = pd.read_csv(acoustics_path)
    scored = pd.read_csv(scored_path).drop(
        columns=["session", "pass", "verbatim_text", "script_text", "was_corrected"])
    scored = scored.rename(columns={"hypothesis": "apple_hypothesis"})
    whisper = pd.read_csv(whisper_path).drop(columns=["session", "pass", "verbatim_text"])
    whisper = whisper.rename(columns={
        "hypothesis": "whisper_hypothesis", "model_size": "whisper_model_size",
        "compute_type": "whisper_compute_type", "elapsed_ms": "whisper_elapsed_ms",
        "substitutions": "whisper_substitutions", "deletions": "whisper_deletions",
        "insertions": "whisper_insertions", "reference_length": "whisper_reference_length",
        "wer": "whisper_wer", "alignment": "whisper_alignment",
    })

    df = acoustics.merge(scored, on="utt_id", how="inner", validate="one_to_one")
    df = df.merge(whisper, on="utt_id", how="inner", validate="one_to_one")

    expected = len(acoustics)
    if len(df) != expected or len(df) != len(scored) or len(df) != len(whisper):
        raise ValueError(
            f"join produced {len(df)} rows; expected {expected} "
            f"(acoustics={len(acoustics)}, scored={len(scored)}, whisper={len(whisper)}) "
            f"-- refusing to proceed with a mismatched join"
        )
    return df


def compute_intensity_db(wav_path: Path) -> float:
    """Per-utterance intensity: 20*log10(RMS), reusing 07_calibration.py's
    compute_rms() (a direct Praat "Get root-mean-square" call -- RMS is not
    one of phonpipe's own measures, so this isn't a phonpipe-measure
    reimplementation). Not calibrated to absolute SPL, but that's fine for
    a within-recording-setup covariate; relative dB is what matters here."""
    import parselmouth
    snd = parselmouth.Sound(str(wav_path))
    rms = calibration_mod.compute_rms(snd)
    return 20 * math.log10(rms) if rms and rms == rms and rms > 0 else float("nan")


def add_intensity_and_speech_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["intensity_db"] = [compute_intensity_db(Path(p)) for p in df["wav_path"]]
    df["speech_rate_wps"] = df["verbatim_reference_length"] / df["duration_sec"]
    return df


def add_wer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recomputes WER from the raw integer error counts + reference_length
    rather than reusing scored.csv's/wer_whisper.csv's already-rounded (4dp)
    wer columns, so the model fits on full precision, not rounded display
    values."""
    df = df.copy()
    df["apple_wer"] = (
        df["verbatim_substitutions"] + df["verbatim_deletions"] + df["verbatim_insertions"]
    ) / df["verbatim_reference_length"]
    df["whisper_wer_full"] = (
        df["whisper_substitutions"] + df["whisper_deletions"] + df["whisper_insertions"]
    ) / df["whisper_reference_length"]
    return df


def add_zscored_predictors(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


# -- pre-modeling check -----------------------------------------------------------

def check_roomtone_by_session(calibration_path: Path) -> dict:
    cal = pd.read_csv(calibration_path, encoding="utf-8-sig")
    room = cal[cal["label"] == "roomtone"]
    sessions = sorted(room["session"].unique())
    if len(sessions) != 2:
        raise ValueError(f"expected exactly 2 sessions of roomtone clips, found {sessions}")
    s1, s2 = sessions
    v1 = room.loc[room["session"] == s1, "rms_level"]
    v2 = room.loc[room["session"] == s2, "rms_level"]
    m1, m2 = v1.mean(), v2.mean()
    pct_diff = abs(m1 - m2) / ((abs(m1) + abs(m2)) / 2) * 100
    t_stat, p_value = stats.ttest_ind(v1, v2)
    return {
        "session_1": s1, "session_2": s2,
        "mean_1": m1, "mean_2": m2, "n_1": len(v1), "n_2": len(v2),
        "pct_diff": pct_diff, "t_stat": t_stat, "p_value": p_value,
        "flagged": pct_diff > ROOMTONE_FLAG_THRESHOLD_PCT,
    }


# -- primary + secondary-1 model -----------------------------------------------------------

PRIMARY_FORMULA = "{outcome} ~ shr_median_z + jitter_local_pct_z + intensity_db_z + speech_rate_wps_z + f0_tracking_failed"


def fit_mixed_model(df: pd.DataFrame, outcome: str) -> tuple[object, list[str]]:
    """Returns (result, warning_messages). Warnings are captured rather than
    left to print to stderr on their own, so a convergence problem is
    reported right next to the model it happened in, not easy to miss in
    scrollback."""
    data = df.copy()
    data["_group"] = "all"  # crossed random effects: single dummy group, both
    # factors as variance components -- see module docstring.
    model = smf.mixedlm(
        PRIMARY_FORMULA.format(outcome=outcome),
        data=data,
        groups="_group",
        vc_formula={"item_id": "0 + C(item_id)", "session": "0 + C(session)"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit()
    messages = [str(w.message) for w in caught]
    return result, messages


# -- secondary analyses -----------------------------------------------------------

def error_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled substitution/deletion/insertion rates (errors of that type /
    total reference words) by pass, for each recognizer separately. Excess
    deletions relative to substitutions would point at a segmentation/VAD
    problem rather than an acoustic-model one."""
    specs = {
        "Apple": ("verbatim_substitutions", "verbatim_deletions", "verbatim_insertions", "verbatim_reference_length"),
        "Whisper": ("whisper_substitutions", "whisper_deletions", "whisper_insertions", "whisper_reference_length"),
    }
    rows = []
    for pass_name, subset in df.groupby("pass"):
        for recognizer, (sub_c, del_c, ins_c, ref_c) in specs.items():
            ref_total = subset[ref_c].sum()
            rows.append({
                "pass": pass_name,
                "recognizer": recognizer,
                "substitution_rate": subset[sub_c].sum() / ref_total,
                "deletion_rate": subset[del_c].sum() / ref_total,
                "insertion_rate": subset[ins_c].sum() / ref_total,
                "n": len(subset),
            })
    return pd.DataFrame(rows)


def verbatim_vs_script_gap(df: pd.DataFrame) -> dict:
    """Apple-only: script_text WER conflates speaker deviation from the
    script with recognition error, so a gap here is expected; this reports
    its size."""
    verb_errs = (df["verbatim_substitutions"] + df["verbatim_deletions"] + df["verbatim_insertions"]).sum()
    verb_ref = df["verbatim_reference_length"].sum()
    script_errs = (df["script_substitutions"] + df["script_deletions"] + df["script_insertions"]).sum()
    script_ref = df["script_reference_length"].sum()
    verb_wer = verb_errs / verb_ref
    script_wer = script_errs / script_ref
    return {"verbatim_wer": verb_wer, "script_wer": script_wer, "gap": script_wer - verb_wer}


def correction_rate_by_pass(df: pd.DataFrame) -> pd.Series:
    return df.groupby("pass")["was_corrected"].mean().rename("correction_rate")


def sentence_type_effect_within_b_natural(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """item_type (command/declarative/final_fall, from item_id's C/D/F
    prefix) effects on Apple WER, restricted to pass == B_natural. Reports
    per-type descriptive stats plus a one-way ANOVA across the three types
    (low power at n~65-70/group within one pass, but directionally
    informative)."""
    subset = df[df["pass"] == "B_natural"]
    groups = subset.groupby("item_type")["apple_wer"]
    summary = groups.agg(["mean", "std", "count"]).rename(columns={"count": "n"})
    type_names = sorted(subset["item_type"].unique())
    f_stat, p_value = stats.f_oneway(*[subset.loc[subset["item_type"] == t, "apple_wer"] for t in type_names])
    return summary, f_stat, p_value


# -- figure -----------------------------------------------------------

def make_figure1(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    long_df = pd.concat([
        pd.DataFrame({"shr_median": df["shr_median"], "wer": df["apple_wer"], "recognizer": "Apple"}),
        pd.DataFrame({"shr_median": df["shr_median"], "wer": df["whisper_wer_full"], "recognizer": "Whisper"}),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"Apple": "#1f77b4", "Whisper": "#d62728"}
    for recognizer, color in colors.items():
        sub = long_df[long_df["recognizer"] == recognizer]
        ax.scatter(sub["shr_median"], sub["wer"], s=14, alpha=0.4, color=color, label=recognizer)

        ols = smf.ols("wer ~ shr_median", data=sub).fit()
        x_grid = np.linspace(sub["shr_median"].min(), sub["shr_median"].max(), 100)
        pred = ols.get_prediction(pd.DataFrame({"shr_median": x_grid})).summary_frame(alpha=0.05)
        ax.plot(x_grid, pred["mean"], color=color, linewidth=2)
        ax.fill_between(x_grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color=color, alpha=0.2)

    ax.set_xlabel("SHR (median)")
    ax.set_ylabel("Word error rate")
    ax.set_title("WER vs. subharmonic-to-harmonic ratio, by recognizer")
    ax.legend(title="Recognizer")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -- reporting -----------------------------------------------------------

def print_mixedlm_summary_table(result, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'term':28} {'coef':>10} {'std err':>10} {'z':>8} {'P>|z|':>8} {'[0.025':>9} {'0.975]':>9}")
    for term in result.params.index:
        coef = result.params[term]
        se = result.bse[term]
        z = result.tvalues[term]
        p = result.pvalues[term]
        ci_low, ci_high = result.conf_int().loc[term]
        print(f"{term:28} {coef:>10.4f} {se:>10.4f} {z:>8.3f} {p:>8.4f} {ci_low:>9.4f} {ci_high:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustics", type=Path, default=Path("results/acoustics_joined.csv"))
    parser.add_argument("--scored", type=Path, default=Path("results/scored.csv"))
    parser.add_argument("--whisper", type=Path, default=Path("results/wer_whisper.csv"))
    parser.add_argument("--calibration", type=Path, default=Path("results/calibration.csv"))
    parser.add_argument("--figure-out", type=Path, default=Path("results/figure1.png"))
    args = parser.parse_args()

    for path in (args.acoustics, args.scored, args.whisper, args.calibration):
        if not path.exists():
            sys.exit(f"{path} not found")

    print("=== h1_h2_db_mean: EXCLUDED from every model below ===")
    print("Failed 5 independent checks over this project's history: sign inversion")
    print("vs. physiological expectation; 63 shr_median>0.45=>h1_h2<=+2dB violations")
    print("(README Limitations); degenerate Stage 8d token-level threshold tuning")
    print("(F1 == trivial always-predict-creaky baseline); 63% per-token NaN rate")
    print("(Stage 8d, measure_tilt() failing on short spans); same shr_median>0.45")
    print("pattern recurring independently in that per-token investigation.")

    print("\n=== Pre-modeling check: roomtone RMS by session ===")
    roomtone = check_roomtone_by_session(args.calibration)
    print(f"{roomtone['session_1']}: mean={roomtone['mean_1']:.6f} (n={roomtone['n_1']}); "
          f"{roomtone['session_2']}: mean={roomtone['mean_2']:.6f} (n={roomtone['n_2']})")
    print(f"symmetric % difference: {roomtone['pct_diff']:.1f}%  "
          f"(t={roomtone['t_stat']:.2f}, p={roomtone['p_value']:.3f}, n=3/session -- very low power)")
    if roomtone["flagged"]:
        print(f"FLAGGED: >{ROOMTONE_FLAG_THRESHOLD_PCT:.0f}% difference between sessions.")
        print("Intensity is NOT directly comparable across sessions on this evidence.")
        print("The model's random intercept by session absorbs a session-level mean")
        print("shift in the OUTCOME, but does not fully resolve this for interpreting")
        print("the intensity coefficient itself -- treat it with that caveat.")
    else:
        print("Not flagged at the 15% threshold used elsewhere in this project (07_calibration.py).")

    print("\nLoading and joining acoustics_joined.csv, scored.csv, wer_whisper.csv...")
    df = load_joined_data(args.acoustics, args.scored, args.whisper)
    print(f"Joined {len(df)} rows.")

    print("Computing per-utterance intensity (Praat RMS) and speech rate...")
    df = add_intensity_and_speech_rate(df)
    df = add_wer_columns(df)
    df = add_zscored_predictors(df, ["shr_median", "jitter_local_pct", "intensity_db", "speech_rate_wps"])

    print("\nFitting primary model (Apple WER)...")
    primary_result, primary_warnings = fit_mixed_model(df, "apple_wer")
    print_mixedlm_summary_table(primary_result, "PRIMARY MODEL: Apple WER ~ acoustic predictors + covariates")
    print(f"Random effects variance: item_id={primary_result.vcomp[0]:.5f}, "
          f"session={primary_result.vcomp[1]:.5f} (session has only 2 levels -- "
          f"this variance component is very imprecisely estimated)")
    if primary_warnings:
        print("Fitting warnings (report these -- don't silently trust a model that raised them):")
        for w in primary_warnings:
            print(f"  - {w}")

    print("\nFitting secondary model 1 (Whisper WER, same structure)...")
    whisper_result, whisper_warnings = fit_mixed_model(df, "whisper_wer_full")
    print_mixedlm_summary_table(whisper_result, "SECONDARY 1: Whisper WER ~ acoustic predictors + covariates")
    if whisper_warnings:
        print("Fitting warnings (report these -- don't silently trust a model that raised them):")
        for w in whisper_warnings:
            print(f"  - {w}")

    print("\n=== SECONDARY 2: Error-type breakdown by pass x recognizer (pooled rates) ===")
    breakdown = error_type_breakdown(df)
    print(breakdown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== SECONDARY 3: Apple WER, verbatim_text vs script_text ===")
    gap = verbatim_vs_script_gap(df)
    print(f"verbatim WER={gap['verbatim_wer']:.4f}  script WER={gap['script_wer']:.4f}  "
          f"gap (script - verbatim)={gap['gap']:.4f}")

    print("\n=== SECONDARY 4: Correction rate (was_corrected) by pass ===")
    print(correction_rate_by_pass(df).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== SECONDARY 5: Sentence type (item_type) effects within B_natural, Apple WER ===")
    sentence_summary, f_stat, p_value = sentence_type_effect_within_b_natural(df)
    print(sentence_summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"one-way ANOVA across item_type: F={f_stat:.3f}, p={p_value:.4f}")

    print(f"\nWriting {args.figure_out}...")
    make_figure1(df, args.figure_out)

    print(f"\nDone. Wrote {args.figure_out}. Nothing else written -- results above are print-only.")


if __name__ == "__main__":
    main()
