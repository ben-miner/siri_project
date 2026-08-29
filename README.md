# siri_project

Does creaky phonation increase word error rate in real ASR systems? This
project measures it directly: one speaker reads the same 100 sentences
under three phonation conditions (modal, natural, and deliberately creaky
voice), across two recording sessions, and the resulting 600 utterances
are scored against two speech recognizers — Apple's on-device
`SFSpeechRecognizer` and OpenAI's Whisper (`large-v3`, via faster-whisper)
— to test whether acoustic creak measures predict recognition errors
independent of loudness and speaking rate.

Single-speaker, exploratory study — not a general claim about creak and
ASR, but a specific, carefully-instrumented test of the mechanism in one
controlled corpus. See `CLAUDE.md` for the project's binding conventions
and non-negotiables (join keys, no audio filtering, ever, etc.) if you're
working on this code rather than just reading the results.

## Design

- **Speaker**: one speaker (the repository author), recorded twice
  (`s01`, `s02`) on separate days.
- **Stimuli**: 100 sentences (`stimuli/items.csv`), built deterministically
  by `03_build_stimuli.py`: 40 commands (`C001`-`C040`, Siri-style voice
  commands), 35 declaratives (`D001`-`D035`), and 25 sentences with a
  final falling intonation contour (`F001`-`F025`).
- **Conditions** (`pass` in every results file): each of the 100 sentences
  is read once under each condition, in three deterministically-shuffled
  read orders (`stimuli/orders.csv`) so condition isn't confounded with
  recording order:
  - `A_modal` — modal (baseline) voice.
  - `B_natural` — the speaker's ordinary, natural reading style.
  - `C_creak` — deliberately creaky phonation throughout.
- **Total corpus**: 100 sentences × 3 conditions × 2 sessions = 600
  utterances (`results/manifest.csv`).
- **Reference transcripts**: every utterance has both `script_text` (what
  was supposed to be read) and `verbatim_text` (what was actually said,
  hand-corrected by ear against the recording — never derived from any
  ASR output, see `CLAUDE.md`). 64 of the 600 utterances needed a
  verbatim correction (`results/references.csv`).
- **Recognizers compared**:
  - Apple `SFSpeechRecognizer`, on-device
    (`requiresOnDeviceRecognition = true`), via `swift/CreakASR` — see
    that package's README for why `SFSpeechRecognizer` rather than the
    newer `SpeechAnalyzer` (hardware constraint: no macOS 26 Mac
    available for this project).
  - Whisper `large-v3` via faster-whisper, CPU inference, VAD filtering
    explicitly disabled (`10b_whisper_baseline.py`) — VAD would discard
    the low-energy, aperiodic stretches that creak actually looks like
    acoustically, which would bias the exact comparison this project is
    testing.
- **Acoustic measurement**: via `phonpipe` (a separate tool by the same
  author — see Setup), MFA-aligned per utterance, plus this project's own
  blind hand-annotation validation (Stage 8) and recording-session
  calibration reference (Stage 7).

## Setup

This project depends on
[`phonpipe`](https://github.com/ben-miner/acoustic-phonetic-master-data-extractor)
(the author's own acoustic-measurement tool — GPL v3, embeds Praat via
`parselmouth`) and Montreal Forced Aligner (MFA), both installed into one
conda environment.

```bash
# 1. Clone both repos
git clone https://github.com/ben-miner/siri_project.git
git clone https://github.com/ben-miner/acoustic-phonetic-master-data-extractor.git

# 2. Create the conda env from phonpipe's environment.yml -- this installs
#    MFA and its conda-only dependencies ("MFA is conda-only; everything
#    else is pip" per phonpipe's own README) and phonpipe itself.
cd acoustic-phonetic-master-data-extractor
conda env create -f environment.yml
conda activate phonpipe

# If `import phonpipe` doesn't work after that (environment.yml SHOULD
# install it, but if it only installed phonpipe's dependencies):
#   pip install -e .

# 3. Download MFA's English acoustic model + dictionary
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa

# 4. Install this project's own dependencies into the same env
cd ../siri_project
pip install -r requirements.txt
```

Developed on Windows 11 with Python 3.11; the conda env should work
identically on macOS/Linux for everything except Stage 9 (below).

**Windows gotcha**: a bare `python.exe` invocation doesn't inherit the
PATH/DLL setup conda's activation provides, so `06_measure.py` (the only
script that shells out to `mfa`) has to be run via
`conda run -n phonpipe python src\06_measure.py` — see Running scripts.

**macOS only, separate toolchain**: `swift/CreakASR` (Stage 9, the
on-device ASR measurement) is a Swift package that must be built and run
on a Mac — see `swift/README.md` for exact commands and prerequisites.
Everything else in this repo is pure Python (plus the one Swift package)
and doesn't need a Mac.

## Data

Raw and processed audio (`data/raw/`, `data/converted/`, `data/split/`,
`data/calibration/`) is **not included** in this repository —
`.gitignore` excludes all `.wav`/`.m4a`/`.mp3`/`.aiff` files. What *is*
included, and sufficient to inspect or extend the analysis without
re-recording anything:

- Every derived measurement and result: `results/*.csv` (the 600-row
  `manifest.csv`/`acoustics_joined.csv`/`scored.csv`/`wer_whisper.csv`,
  the 60-utterance hand-annotation files, calibration data — everything
  `11_analysis.py`'s models actually run on) and `results/figure1.png`.
- All 600 utterances' reference transcripts, both the intended script and
  the hand-corrected verbatim text (`results/references.csv`,
  `manifest.csv`).
- The 60 hand-annotated TextGrids used to validate and tune the creak
  detection thresholds (`data/textgrids/annotation/`).
- The stimulus sentences and read orders (`stimuli/`), and the recording
  session logs (`docs/`).
- All pipeline/analysis code (`src/`, `swift/`) and its tests (`tests/`).

Reproducing Stages 1-9 from scratch would need your own matching audio
recordings (same script, comparable setup) — the pipeline is written
generically enough that it should run on a different speaker's recordings
of the same `stimuli/items.csv` script, but that's untested. Stages 10
onward (scoring, analysis) run directly on the checked-in `results/`
files with no audio required at all.

## Pipeline

Each stage is a standalone script in `src/`, run in order; each writes its
output to `results/` (CSV) or `data/` and prints a one-line summary.

| stage | script | does |
|---|---|---|
| 1 | `01_inventory.py` | Inventory audio files under `data/raw/` via ffprobe. |
| 2 | `02_convert.py` | Extract channel 0 of each raw WAV to a mono file in `data/converted/`. |
| 3 | `03_build_stimuli.py` | Regenerate the stimulus item list and read orders deterministically. |
| 4a | `04a_preamble.py` | Extract calibration clips and the sentence block from each preamble TextGrid. |
| 4b | `04b_align.py` | Word-align Whisper transcripts of each sentence block to the expected script. |
| 4c | `04c_cut.py` | Cut each utterance out of its sentence block using hand-corrected boundaries. |
| 4d | `04d_verbatim.py` | Build and finalize the verbatim-correction review queue. |
| 5 | `05_manifest.py` | Build the analysis manifest by joining split clips to items and references. |
| 6 | `06_measure.py` | MFA-align on known-correct transcripts, then measure with phonpipe; joins to `manifest.csv`. |
| 6d | `06d_version_diff.py` | Diff two `acoustics_joined.csv` runs (e.g. before/after a phonpipe fix) across every measure, flagging sign changes separately. |
| 6e | `06e_select_validation_sample.py` | Select a stratified 10-utterance sample for hand validation of phonpipe's F0. |
| 7 | `07_calibration.py` | Build the empirical calibration reference from `data/calibration/` clips (creak_cal, sustained_a, roomtone, etc). |
| 8a | `08a_select_annotation_sample.py` | Select a 60-utterance sample stratified on phonpipe's creak proportion (terciles × pass), blind to condition/estimate. |
| 8b | `08b_prepare_grids.py` | Copy each sampled TextGrid, add an empty sonorant-boundary `creak` tier, and pair blind-named grids/audio for hand annotation. |
| 8c | `08c_compute_proportions.py` | Validate the completed hand annotation against the phone tier, compute token-weighted `hand_creak_proportion`, correlate against phonpipe. |
| 8d | `08d_tune_thresholds.py` | Grid-search per-token creak thresholds against the hand annotation; prints (never writes) the winning config. |
| 9 | `swift/CreakASR` | macOS command-line Swift package (not a `src/*.py` script — see `swift/README.md`). Transcribes with `SFSpeechRecognizer` (`requiresOnDeviceRecognition = true`), writes `hypotheses.csv`. |
| 10 | `10_wer.py` | Normalizes text (case, punctuation, numeral/symbol expansion) and computes a from-scratch Levenshtein WER; joins `hypotheses.csv` to `manifest.csv`, writes `scored.csv` against both `verbatim_text` and `script_text`. |
| 10b | `10b_whisper_baseline.py` | Same normalizer/scorer (imported from `10_wer.py`), run against faster-whisper `large-v3` instead of the on-device recognizer — VAD filtering explicitly off, since it would systematically discard creak. Writes `wer_whisper.csv`. |
| 11 | `11_analysis.py` | Joins `acoustics_joined.csv`/`scored.csv`/`wer_whisper.csv`; mixed-effects models of WER on acoustic predictors, plus the secondary analyses below. Print-only; writes `figure1.png`. |

`manifest.csv`/`acoustics_joined.csv` are keyed on `utt_id` throughout —
see CLAUDE.md's non-negotiables. `config/thresholds.yaml` is Stage 8d's
output, pasted and committed by hand per that script's print-only
convention; see the file's own header for the frozen values and the
negative results (H1-H2) behind them.

## Running scripts

Most scripts only import Python packages and run fine invoked directly:

```
"C:\Users\<you>\miniconda3\envs\phonpipe\python.exe" src\01_inventory.py
```

**`06_measure.py` is an exception.** It shells out to the `mfa` (Montreal
Forced Aligner) executable via `phonpipe.align.run_mfa`. `mfa` is installed
in the `phonpipe` env (`envs\phonpipe\Scripts\mfa.exe`), but a bare
`python.exe` invocation does not inherit the PATH/DLL setup conda's
activation normally provides, so `shutil.which("mfa")` fails to find it and
the script exits with `RuntimeError: MFA not found` even though it's
installed. Run it via `conda run` instead, which sets up the environment
correctly:

```
conda run -n phonpipe python src\06_measure.py
```

(Confirmed: `shutil.which("mfa")` returns `None` under a direct `python.exe`
call and the full path under `conda run -n phonpipe python`.)

## Dependency versions

- **phonpipe**: pinned to commit
  [`f74fe22`](https://github.com/ben-miner/acoustic-phonetic-master-data-extractor/commit/f74fe22)
  (`main`). Two commits back-to-back:
  - [`d2ab07b`](https://github.com/ben-miner/acoustic-phonetic-master-data-extractor/commit/d2ab07b)
    (merged from `fix/adaptive-range-period-doubling`) fixes a
    period-doubling self-reinforcement bug in `_adaptive_range` that caused
    wrong (harmonic-locked) F0/SHR/H1-H2 on sustained, isolated creak with
    no modal frames to anchor against — e.g. the
    `data/calibration/*creak_cal.wav` tokens and some `C_creak`-pass
    utterances.
  - `f74fe22` makes that fix's detection response proportional to the
    evidence instead of all-or-nothing. The first version discarded the
    bootstrap to a full 40–750 Hz fallback any time either detection signal
    fired, which over-penalised borderline cases: auditing the v2 re-run
    found a plausible A_modal bootstrap (`s02_A_modal_C010`, floor
    110.6 Hz) discarded over a wide-SHR margin of 0.0036, flipping
    `h1_h2_db_mean` from -5.33 to +11.35 dB, and the same signature on all
    22 `C_creak` `neg→pos` sign changes from that re-run. Now a plausible
    floor with only the SHR signal firing gets the range extended downward
    (floor/2) instead of discarded.

  See phonpipe's own README ("Known limitation: sustained, isolated creak
  can defeat the adaptive range") for the full root cause and validation of
  both commits. Stage 6/7 acoustic measurements: `results/acoustics_summary_v1.csv`
  (pre-fix, archived) and `_v2.csv` (post-`d2ab07b`, archived) are affected
  on flagged/sign-changed rows; current results in `results/acoustics_joined.csv`
  are v3 (post-`f74fe22`). `results/version_diff_signs.csv` compares v3
  against v1 directly (`06d_version_diff.py --v1 acoustics_joined_v1.csv`).

## Results

From `11_analysis.py` (print-only; run it to regenerate these numbers).

### Excluded predictor

**`h1_h2_db_mean` is excluded from every model below.** It failed five
independent checks over this project's history: a sign inversion relative
to the physiologically expected direction; 63 `shr_median>0.45 =>
h1_h2<=+2dB` violations (see Limitations below); degenerate Stage 8d
token-level threshold tuning (F1 equal to the trivial "always predict
creaky" baseline); a 63% per-token NaN rate from `measure_tilt()` failing
on short spans (Stage 8d); and the same `shr_median>0.45` pattern
recurring independently in that per-token investigation.

### Pre-modeling check: roomtone RMS by session

| session | mean RMS | n |
|---|---|---|
| s01 | 0.000909 | 3 |
| s02 | 0.001192 | 3 |

27.0% symmetric difference (t=-1.64, p=0.176, n=3/session — very low
power). **Flagged** at the 15% threshold used elsewhere in this project
(`07_calibration.py`). Intensity is not directly comparable across
sessions on this evidence; the model's random intercept by session
absorbs a session-level mean shift in the *outcome* but does not fully
resolve this for interpreting the intensity *coefficient* itself.

### Primary model: Apple WER ~ acoustic predictors + covariates

Errors-per-reference-word, mixed-effects, crossed random intercepts for
`item_id` and `session`. Continuous predictors z-scored (coefficients are
per-1-SD).

| term | coef | std err | z | p | [0.025, 0.975] |
|---|---|---|---|---|---|
| Intercept | 0.0764 | 0.0079 | 9.70 | <0.0001 | [0.0610, 0.0919] |
| f0_tracking_failed | 0.0011 | 0.0120 | 0.09 | 0.929 | [-0.0225, 0.0246] |
| **shr_median (z)** | **0.0161** | 0.0068 | 2.36 | **0.018** | [0.0027, 0.0295] |
| jitter_local_pct (z) | 0.0075 | 0.0071 | 1.06 | 0.292 | [-0.0064, 0.0214] |
| intensity_db (z) | 0.0187 | 0.0039 | 4.78 | <0.0001 | [0.0110, 0.0264] |
| speech_rate_wps (z) | -0.0114 | 0.0049 | -2.35 | 0.019 | [-0.0209, -0.0019] |
| item_id variance | 0.5152 | 0.1065 | — | — | — |
| session variance | 0.0000 | — | — | — | — |

`shr_median` predicts WER independent of intensity, speech rate, and f0
tracking failure — the core result. Note: session variance converged to
the parameter-space boundary (0), and the fit raised convergence warnings
("Maximum Likelihood optimization failed to converge", "MLE may be on the
boundary of the parameter space"). This is a plausible, expected outcome
given only 2 session levels, not a code defect — but it means the session
random effect is not meaningfully estimated here, only nominally
included. `jitter_local_pct` is not significant in this model, though it
is in the Whisper model below.

### Secondary 1: Whisper WER, same model structure

| term | coef | std err | z | p |
|---|---|---|---|---|
| Intercept | 0.0269 | 0.0045 | 5.99 | <0.0001 |
| f0_tracking_failed | 0.0086 | 0.0091 | 0.95 | 0.345 |
| shr_median (z) | -0.0016 | 0.0059 | -0.27 | 0.789 |
| **jitter_local_pct (z)** | **0.0141** | 0.0052 | 2.74 | **0.006** |
| intensity_db (z) | 0.0053 | 0.0056 | 0.96 | 0.338 |
| speech_rate_wps (z) | -0.0092 | 0.0036 | -2.51 | 0.012 |
| item_id variance | 0.2637 | 0.0674 | — | — |

**Coefficient pattern differs by recognizer**: `shr_median` predicts
Apple's errors but not Whisper's; `jitter_local_pct` predicts Whisper's
errors but not Apple's. Intensity is significant for Apple only. This
model's fit is less stable than the primary one — repeated "Random
effects covariance is singular" and non-convergence warnings — plausibly
because Whisper's much lower, more zero-inflated WER leaves less variance
for the random effects to explain; take the point estimates as
indicative, not as precise as the primary model's.

### Secondary 2: Error-type breakdown by pass × recognizer (pooled rates)

| pass | recognizer | substitution | deletion | insertion |
|---|---|---|---|---|
| A_modal | Apple | 0.0367 | 0.0152 | 0.0101 |
| A_modal | Whisper | 0.0076 | 0.0025 | 0.0070 |
| B_natural | Apple | 0.0411 | 0.0329 | 0.0057 |
| B_natural | Whisper | 0.0089 | 0.0038 | 0.0101 |
| C_creak | Apple | 0.0517 | 0.0240 | 0.0107 |
| C_creak | Whisper | 0.0196 | 0.0019 | 0.0177 |

No evidence of a segmentation/VAD problem: deletions don't scale with
creak severity for either recognizer — Apple's deletion rate actually
peaks in B_natural, not C_creak, and Whisper's deletion rate is lowest in
C_creak. What scales with creak is substitutions (both recognizers) and
insertions (Whisper especially: 0.0070→0.0101→0.0177). That pattern
points at an acoustic-model confusion, not lost/discarded audio.

### Secondary 3: Apple WER, verbatim_text vs script_text

verbatim WER = 0.0761, script WER = 0.0842, **gap = 0.0081**.

### Secondary 4: Correction rate (was_corrected) by pass

| pass | correction rate |
|---|---|
| A_modal | 0.120 |
| B_natural | 0.085 |
| C_creak | 0.115 |

### Secondary 5: Sentence type effects within B_natural (Apple WER)

| item_type | mean WER | std | n |
|---|---|---|---|
| command | 0.0639 | 0.0908 | 80 |
| **declarative** | **0.1185** | 0.1267 | 70 |
| final_fall | 0.0533 | 0.0805 | 50 |

One-way ANOVA: F=7.596, **p=0.0007**. Declarative sentences are
substantially harder within the natural-speech condition specifically.

### Figure

`results/figure1.png` (300 dpi): WER vs. `shr_median`, points colored by
recognizer, OLS fit line + 95% CI band per recognizer.

## Limitations

- **H1-H2 on strongly period-doubled speech is unresolved.** Physiological
  expectation (from this project's own `creak_cal` calibration clips,
  deliberate maximal creak): strong period doubling should read negative
  H1-H2 (creaky), not positive (breathy). Checking that constraint
  (`tests/test_creak_h1_h2_plausibility.py`: `shr_median > 0.45` implies
  `h1_h2_db_mean <= +2 dB`) against the current `C_creak` data finds 63
  violations out of 126 qualifying utterances — roughly half of all
  strongly period-doubled `C_creak` tokens read breathier than modal voice
  while showing strong subharmonic energy. 49 of the 63 predate every
  phonpipe fix made in this project's history (already present in
  `acoustics_joined_v1.csv`), so this is not a consequence of the
  adaptive-range work above — it looks like `measure_tilt()`'s harmonic
  search behaving differently on aperiodic/irregular voicing than on
  periodic voicing, independent of which floor/ceiling it's given. Not
  yet investigated further. **Affects H1-H2's reliability as a creak
  predictor in the WER analysis** — treat it as a noisier signal than SHR
  for roughly this subset of tokens until this is understood. The +2 dB
  threshold itself is provisional (derived from sustained deliberate
  creak, which is more extreme than creak within running speech), so some
  fraction of the 63 may reflect real milder tokens near a fuzzy boundary
  rather than a defect — that per-file judgment has not been made.

- **Per-token H1-H2 has no discriminative power for creak.** Stage 8d's
  grid search against 60 hand-annotated utterances (1077 sonorant tokens)
  found H1-H2's best token-level F1 (0.349) exactly equals the trivial
  "always predict creaky" baseline, confirmed non-degenerate under
  widened grids (-30 to +40 dB) and unrelated to the point above — 63% of
  tokens returned NaN outright (`measure_tilt()` failing on spans under
  ~60ms), and even where it succeeds, a single token's spectral tilt
  estimate is far noisier than an utterance-level mean across many
  tokens. `h1_h2_db` is excluded from `config/thresholds.yaml` as a
  result. Whatever signal H1-H2 carries about creak in this corpus is
  only recoverable as an utterance-level aggregate — it correlates
  r=0.772 with phonpipe's `creak_doubling_rate` (Stage 8c,
  `results/annotation_vs_phonpipe.png`) — not from a single token's own
  span.

- **Single speaker.** Every result above describes this one speaker's
  voice and recording setup. Random intercepts by `item_id` and `session`
  account for item- and session-level variation within that speaker's
  data; they say nothing about how creak affects ASR accuracy for anyone
  else.

- **Session variance in the mixed models is not meaningfully estimated**
  (see Results) — there are only 2 sessions, which is too few to
  precisely estimate a variance component from regardless of model
  specification.

## License

GPL v3 (see `LICENSE`) — required because this project's acoustic
measurement code depends on `phonpipe`, which embeds Praat via
`parselmouth` and is itself GPL v3.

```
siri_project: measuring creaky-phonation effects on ASR word error rate
Copyright (C) 2026  ben-miner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```
