"""Build the empirical calibration reference from data/calibration/ clips.

Runs `phonpipe extract --no-align` over all 30 calibration clips (5 labels
x 3 passes x 2 sessions) and writes results/calibration.csv, one row per
clip, with phonpipe's global (file-level) measures plus two things phonpipe
doesn't provide at all: RMS level (not one of its measures) and H1-H2 for
creak_cal clips (see below for why that needs a workaround).

WHY H1-H2 NEEDS SPECIAL HANDLING
---------------------------------
H1-H2 (phonpipe/measures/voice_quality.py: measure_tilt) is a PER-VOWEL-TOKEN
measure -- extract.py only calls it inside the per-phone loop, which only
runs when a TextGrid supplies phone intervals. --no-align mode sets
phones=None, so in a plain `phonpipe extract --no-align` run, H1-H2 is never
computed at all: measure_voice_quality_global (the function that actually
runs in --no-align mode) returns HNR/jitter/shimmer/CPPS only, no H1-H2. And
there's no known-good transcript for calibration clips the way there is for
the sentence-reading utterances (Stage 4d's verbatim_text), so MFA alignment
isn't an available fix either.

Instead of skipping H1-H2 or reimplementing spectral-tilt math, this script
calls phonpipe's own measure_tilt() function directly on each creak_cal
clip, passing the clip's full extent as the "vowel interval" -- measure_tilt
already trims to the middle 50% internally (dur*0.25 padding each side),
which is exactly the right thing to do on an isolated sustained-creak token.
This is reuse, not reimplementation: the actual H1-H2/H1*-H2*/H1-A3 math is
100% phonpipe's.

WHY F0/SHR/H1-H2 ON creak_cal NEED A CONSTRAINED PITCH RANGE
---------------------------------------------------------------
phonpipe's adaptive floor/ceiling search (measures/f0.py: _adaptive_range)
re-derives the speaker's F0 range from voiced frames in the SAME file. That
works for ordinary speech, which mixes modal and creaky stretches the search
can bootstrap from -- but creak_cal clips are deliberately, continuously
creak with no modal frames present at all, and on this corpus the search
diverges: it converges on an ~350-369 Hz floor (implausible for anyone's
voice, let alone deliberate creak) and phonpipe's own SHS anchor -- the
octave-safety check the README describes -- reports "no_anchor" on 5 of 6
creak_cal clips, i.e. it found no reliable independent confirmation at all.

Note that _adaptive_range() hardcodes its own starting point (50.0, 700.0)
and never reads measure_f0()'s floor/ceiling parameters -- despite the
docstring calling them "starting hints", they are unused dead code. Passing
different values through measure_f0() therefore would not have helped.
extract.py then feeds that same bad floor/ceiling into measure_shr() and
measure_tilt() too (see extract.py's process_file), so SHR and H1-H2 for
creak_cal inherit the same contamination, not just the raw F0 numbers.

The fix used here bypasses the adaptive search entirely for creak_cal only:
a fixed CREAK_F0_FLOOR_HZ/CREAK_F0_CEILING_HZ (40-150 Hz, chosen below this
speaker's own ~109-173 Hz modal range from Stage 6's A_modal utterances) is
used for (1) a direct, non-adaptive parselmouth pitch extraction for the F0
percentiles, and (2) phonpipe's own measure_shr()/measure_tilt(), called
directly with these bounds instead of the file's f0_floor_used/
f0_ceiling_used. (1) is a simple direct Praat call, not a reimplementation
of phonpipe's octave-repair machinery; (2) is straight reuse of phonpipe's
own SHR/tilt math, just given the right search range. The original
unconstrained phonpipe f0/shr columns are kept in calibration.csv alongside
the new creak_f0_*/creak_shr_* columns for comparison, not overwritten.

CALIBRATION REPORT
-------------------
Prints per-session aggregates (mean across that session's 3 passes):
  - creak_cal:   F0 median + [p10, p90] range, SHR, H1-H2 (constrained range)
  - sustained_a: jitter, shimmer, HNR
  - roomtone:    RMS level, and estimated SNR (dB) against the mean RMS of
                 that session's actual read-sentence utterances in
                 data/split/

Any measure differing by more than 15% (symmetric: |a-b| / mean(|a|,|b|))
between session 1 and session 2 is flagged -- gain drift or a vocal state
change the researcher needs to know about, not something to silently average
over.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

CALIB_LABELS = ("roomtone", "sustained_a", "glide", "creak_cal", "carrier")
FILENAME_PATTERN = re.compile(
    r"^(?P<session>[^_]+)_(?P<pass_label>.+)_(?P<label>" + "|".join(CALIB_LABELS) + r")$"
)
FLAG_THRESHOLD_PCT = 15.0

# Below this speaker's ~109-173 Hz modal F0 range (Stage 6 A_modal utterances),
# used to bypass phonpipe's adaptive search for creak_cal only -- see module
# docstring for why the adaptive search fails on isolated sustained creak.
CREAK_F0_FLOOR_HZ = 40.0
CREAK_F0_CEILING_HZ = 150.0


def find_calibration_wavs(calibration_dir: Path) -> list[Path]:
    return sorted(calibration_dir.glob("*.wav"))


def parse_calibration_filename(stem: str) -> tuple[str, str, str]:
    m = FILENAME_PATTERN.match(stem)
    if not m:
        raise ValueError(
            f"'{stem}.wav' does not match {{session}}_{{pass}}_{{label}} "
            f"(label must be one of {CALIB_LABELS})"
        )
    return m.group("session"), m.group("pass_label"), m.group("label")


def canonical_pass(pass_label: str) -> str:
    if not pass_label.startswith("pass"):
        raise ValueError(f"pass label '{pass_label}' does not start with 'pass'")
    return pass_label[len("pass"):]


def load_phonpipe_summary(summary_csv: Path) -> dict:
    with summary_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_stem = {}
    for row in rows:
        stem = Path(row["file"]).stem
        if stem in by_stem:
            raise ValueError(f"duplicate file '{stem}' in {summary_csv}")
        by_stem[stem] = row
    return by_stem


def compute_rms(snd) -> float:
    from parselmouth.praat import call
    return call(snd, "Get root-mean-square", 0, 0)


def compute_h1_h2(snd, f0_floor: float, f0_ceiling: float, max_formant: float) -> dict:
    from phonpipe.measures.voice_quality import measure_tilt
    return measure_tilt(snd, 0.0, snd.duration, f0_floor, f0_ceiling, max_formant=max_formant)


def compute_constrained_f0(
    snd, floor: float = CREAK_F0_FLOOR_HZ, ceiling: float = CREAK_F0_CEILING_HZ
) -> dict:
    """Direct, non-adaptive pitch extraction -- see module docstring for why
    creak_cal needs this instead of phonpipe's own measure_f0()."""
    import numpy as np

    pitch = snd.to_pitch_ac(pitch_floor=floor, pitch_ceiling=ceiling)
    freqs = pitch.selected_array["frequency"]
    voiced = freqs[freqs > 0]
    if voiced.size == 0:
        return {
            "creak_f0_median_hz": math.nan, "creak_f0_p10_hz": math.nan,
            "creak_f0_p90_hz": math.nan, "creak_f0_pct_voiced": 0.0,
        }
    p10, p90 = np.percentile(voiced, [10, 90])
    return {
        "creak_f0_median_hz": float(np.median(voiced)),
        "creak_f0_p10_hz": float(p10),
        "creak_f0_p90_hz": float(p90),
        "creak_f0_pct_voiced": 100.0 * voiced.size / freqs.size,
    }


def compute_constrained_shr(
    snd, floor: float = CREAK_F0_FLOOR_HZ, ceiling: float = CREAK_F0_CEILING_HZ
) -> dict:
    """phonpipe's own measure_shr(), reused directly with the constrained
    range instead of the file's (contaminated) f0_floor_used/ceiling_used."""
    from phonpipe.measures.creak import measure_shr
    stats = measure_shr(snd, floor, ceiling)
    return {"creak_shr_median": stats["shr_median"], "creak_shr_mean": stats["shr_mean"]}


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_calibration_rows(wav_paths: list[Path], summary_by_stem: dict) -> list[dict]:
    import parselmouth

    rows = []
    for path in wav_paths:
        stem = path.stem
        session, pass_label, label = parse_calibration_filename(stem)
        if stem not in summary_by_stem:
            raise ValueError(f"'{stem}' not found in phonpipe's summary output")
        measures = dict(summary_by_stem[stem])

        snd = parselmouth.Sound(str(path))
        rms = compute_rms(snd)

        h1_h2 = {"h1_h2_db": math.nan, "h1s_h2s_db": math.nan, "h1_a3_db": math.nan}
        creak_f0 = {
            "creak_f0_median_hz": math.nan, "creak_f0_p10_hz": math.nan,
            "creak_f0_p90_hz": math.nan, "creak_f0_pct_voiced": math.nan,
        }
        creak_shr = {"creak_shr_median": math.nan, "creak_shr_mean": math.nan}
        if label == "creak_cal":
            max_formant = _float_or_nan(measures.get("formant_ceiling_hz"))
            if math.isnan(max_formant):
                max_formant = 5500.0
            tilt = compute_h1_h2(snd, CREAK_F0_FLOOR_HZ, CREAK_F0_CEILING_HZ, max_formant)
            h1_h2 = {k: tilt[k] for k in ("h1_h2_db", "h1s_h2s_db", "h1_a3_db")}
            creak_f0 = compute_constrained_f0(snd)
            creak_shr = compute_constrained_shr(snd)

        row = {
            "session": session,
            "pass": canonical_pass(pass_label),
            "label": label,
            "wav_path": path.as_posix(),
        }
        row.update(measures)
        row.update(h1_h2)
        row.update(creak_f0)
        row.update(creak_shr)
        row["rms_level"] = rms
        rows.append(row)
    return rows


def compute_session_speech_rms(split_dir: Path) -> dict:
    import parselmouth

    totals: dict[str, list[float]] = {}
    for path in sorted(split_dir.glob("*.wav")):
        session = path.stem.split("_", 1)[0]
        snd = parselmouth.Sound(str(path))
        totals.setdefault(session, []).append(compute_rms(snd))
    return {session: sum(vals) / len(vals) for session, vals in totals.items()}


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return sum(clean) / len(clean) if clean else math.nan


def aggregate_session_report(rows: list[dict], session_speech_rms: dict) -> dict:
    sessions = sorted({row["session"] for row in rows})
    report: dict = {}
    for session in sessions:
        by_label = {
            label: [r for r in rows if r["session"] == session and r["label"] == label]
            for label in CALIB_LABELS
        }
        creak = by_label["creak_cal"]
        sustained = by_label["sustained_a"]
        room = by_label["roomtone"]

        roomtone_rms = _mean([r["rms_level"] for r in room])
        speech_rms = session_speech_rms.get(session, math.nan)
        snr_db = (
            20 * math.log10(speech_rms / roomtone_rms)
            if roomtone_rms and roomtone_rms == roomtone_rms and roomtone_rms > 0
            and speech_rms == speech_rms
            else math.nan
        )

        report[session] = {
            # Constrained measurements (see module docstring) -- NOT phonpipe's
            # raw f0_median_hz/f0_p10_hz/.../shr_median, which are contaminated
            # by the adaptive search's bad floor/ceiling on isolated creak_cal.
            "creak_f0_median_hz": _mean([_float_or_nan(r.get("creak_f0_median_hz")) for r in creak]),
            "creak_f0_p10_hz": _mean([_float_or_nan(r.get("creak_f0_p10_hz")) for r in creak]),
            "creak_f0_p90_hz": _mean([_float_or_nan(r.get("creak_f0_p90_hz")) for r in creak]),
            "creak_shr_median": _mean([_float_or_nan(r.get("creak_shr_median")) for r in creak]),
            "creak_h1_h2_db": _mean([_float_or_nan(r.get("h1_h2_db")) for r in creak]),
            "sustained_jitter_pct": _mean([_float_or_nan(r.get("jitter_local_pct")) for r in sustained]),
            "sustained_shimmer_pct": _mean([_float_or_nan(r.get("shimmer_local_pct")) for r in sustained]),
            "sustained_hnr_db": _mean([_float_or_nan(r.get("hnr_db")) for r in sustained]),
            "roomtone_rms_level": roomtone_rms,
            "roomtone_snr_db": snr_db,
        }
    return report


def symmetric_pct_diff(v1: float, v2: float) -> float:
    if v1 != v1 or v2 != v2:  # either NaN
        return math.nan
    denom = (abs(v1) + abs(v2)) / 2
    if denom == 0:
        return 0.0
    return abs(v1 - v2) / denom * 100


def find_session_flags(report: dict, threshold_pct: float = FLAG_THRESHOLD_PCT) -> list[dict]:
    sessions = sorted(report.keys())
    if len(sessions) != 2:
        return []
    s1, s2 = sessions
    flags = []
    for measure in report[s1]:
        v1, v2 = report[s1][measure], report[s2][measure]
        pct = symmetric_pct_diff(v1, v2)
        if pct == pct and pct > threshold_pct:
            flags.append({"measure": measure, s1: v1, s2: v2, "pct_diff": pct})
    return flags


def print_report(report: dict, flags: list[dict]) -> None:
    print("\n=== Per-session calibration report ===")
    print(f"(creak_cal F0/SHR/H1-H2 use a constrained "
          f"{CREAK_F0_FLOOR_HZ:.0f}-{CREAK_F0_CEILING_HZ:.0f} Hz range, "
          f"not phonpipe's adaptive search -- see module docstring)")
    for session in sorted(report):
        m = report[session]
        print(f"\n{session}:")
        print(f"  creak_cal:   F0 median={m['creak_f0_median_hz']:.1f} Hz, "
              f"range=[{m['creak_f0_p10_hz']:.1f}, {m['creak_f0_p90_hz']:.1f}] Hz, "
              f"SHR median={m['creak_shr_median']:.3f}, H1-H2={m['creak_h1_h2_db']:.1f} dB")
        print(f"  sustained_a: jitter={m['sustained_jitter_pct']:.2f}%, "
              f"shimmer={m['sustained_shimmer_pct']:.2f}%, HNR={m['sustained_hnr_db']:.1f} dB")
        print(f"  roomtone:    RMS={m['roomtone_rms_level']:.5f}, "
              f"estimated SNR={m['roomtone_snr_db']:.1f} dB")

    print(f"\n=== Flags (>{FLAG_THRESHOLD_PCT}% symmetric difference between sessions) ===")
    if not flags:
        print("  none")
    else:
        for f in flags:
            sessions = [k for k in f if k not in ("measure", "pct_diff")]
            vals = ", ".join(f"{s}={f[s]:.3f}" for s in sessions)
            print(f"  {f['measure']}: {vals}  ({f['pct_diff']:.1f}% difference)")


def write_calibration_csv(rows: list[dict], out_path: Path) -> None:
    lead_cols = ["session", "pass", "label", "wav_path"]
    extra_cols = [c for c in rows[0].keys() if c not in lead_cols]
    fieldnames = lead_cols + extra_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=Path("data/calibration"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/split"))
    parser.add_argument("--lang", default="english", choices=["english", "japanese"])
    parser.add_argument(
        "--phonpipe-out", type=Path, default=Path("results/calibration_phonpipe"),
        help="output prefix passed to phonpipe extract -o (default: results/calibration_phonpipe)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/calibration.csv"))
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="reuse an existing <phonpipe-out>_summary.csv instead of rerunning phonpipe extract",
    )
    args = parser.parse_args()

    wav_paths = find_calibration_wavs(args.calibration_dir)
    if not wav_paths:
        sys.exit(f"No wav files found under {args.calibration_dir}")

    if not args.skip_extract:
        from phonpipe.cli import main as phonpipe_main

        argv = [
            "extract", *[str(p) for p in wav_paths],
            "--no-align", "--lang", args.lang, "-o", str(args.phonpipe_out),
        ]
        print(f"Running: phonpipe {' '.join(argv)}")
        ret = phonpipe_main(argv)
        if ret != 0:
            sys.exit(f"phonpipe extract exited with code {ret}")

    summary_csv = Path(f"{args.phonpipe_out}_summary.csv")
    if not summary_csv.exists():
        sys.exit(f"{summary_csv} not found (phonpipe extract may have failed)")
    summary_by_stem = load_phonpipe_summary(summary_csv)

    rows = build_calibration_rows(wav_paths, summary_by_stem)
    write_calibration_csv(rows, args.out)
    print(f"Wrote {len(rows)} row(s) to {args.out}")

    session_speech_rms = compute_session_speech_rms(args.split_dir)
    report = aggregate_session_report(rows, session_speech_rms)
    flags = find_session_flags(report)
    print_report(report, flags)


if __name__ == "__main__":
    main()
