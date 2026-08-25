# siri_project

Measuring whether creaky phonation increases word error rate in Apple's
on-device SpeechTranscriber. See `CLAUDE.md` for full project context,
non-negotiables, and style conventions.

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

`manifest.csv`/`acoustics_joined.csv` are keyed on `utt_id` throughout —
see CLAUDE.md's non-negotiables.

## Environment

Windows 11, Python 3.11, conda env `phonpipe` (shared with the sibling
`../acoustic-phonetic-master-data-extractor` project).

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
