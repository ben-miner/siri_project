"""Standing physiological-plausibility check on results/acoustics_joined.csv:
strong period doubling should not read breathier than modal voice.

CONSTRAINT: this project's own creak_cal calibration clips (deliberate,
sustained, maximal creak) read h1_h2_db_mean = -4.8 to -5.2 dB (see
phonpipe's tests/test_creak_cal_regression.py). A C_creak-pass utterance
with strong period doubling (shr_median > 0.45) reading POSITIVE H1-H2 is
claiming to be breathier than modal voice while simultaneously showing
strong subharmonic energy -- physiologically incoherent. Values near zero
are plausible (creak within running speech is milder than sustained
deliberate creak); the line is drawn at +2 dB, matching the margin used
when auditing the phonpipe adaptive-range proportional-response redesign
(commit f74fe22) against this exact constraint.

THIS IS A PRE-EXISTING PATTERN, NOT A REGRESSION FROM TODAY'S FIX. Running
this check against the full C_creak set (not just the utterances that
changed sign relative to v1) finds 63 violations out of 126 qualifying
rows -- roughly half of all strongly period-doubled C_creak utterances.
Of those 63: 49 were ALREADY positive above +2 dB in v1 (measured before
either phonpipe fix in this project's history), 10 have no v1 baseline
(excluded by v1's heavy_intervention gate, so nothing to compare against),
and only 4 newly crossed +2 dB as a direct consequence of today's
proportional-response redesign (s01_C_creak_C009, s01_C_creak_C029,
s02_C_creak_C038, s02_C_creak_D009). The overwhelming majority of this
pattern predates the adaptive-range work entirely and points at something
else -- most likely measure_tilt()'s harmonic search behaving differently
on aperiodic/irregular voicing than on periodic voicing, independent of
which floor/ceiling it's given. That is an open question, not something
this branch's fix could or did address.

THE +2 dB THRESHOLD ITSELF IS PROVISIONAL. It was derived from 6
creak_cal clips of sustained, deliberate, maximal creak -- more extreme
than creak occurring within running speech, where H1-H2 nearer zero is
plausible. It is entirely possible some fraction of these 63 reflect real
milder creak tokens near the boundary rather than a measurement defect.
That judgment call has not been made per-file; every violation below is
listed, not adjudicated.

PURPOSE: this test is a tripwire for NEW violations on future re-runs, not
a claim that the 63 currently-known cases are individually wrong. Every
one is recorded in KNOWN_RESIDUAL_FAILURES with its category
(pre_existing / no_v1_baseline / newly_crossed) and marked
xfail(strict=False) -- if a future Stage 6 re-run fixes one, that shows up
as an unexpected XPASS; if a future re-run introduces a NEW violation not
in this dict, that fails loudly instead of silently disappearing into a
sea of already-known failures.

See siri_project README's "Limitations" section for the project-level
framing of this as an open issue affecting H1-H2 as a creak predictor.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "acoustics_joined.csv"
SHR_THRESHOLD = 0.45
H1_H2_MAX_DB = 2.0

# utt_id -> category. Category is informational only (goes into the xfail
# reason); it does not change how the row is evaluated.
KNOWN_RESIDUAL_FAILURES = {
    "s01_C_creak_C005": "pre_existing",
    "s01_C_creak_C009": "newly_crossed",
    "s01_C_creak_C015": "no_v1_baseline",
    "s01_C_creak_C024": "pre_existing",
    "s01_C_creak_C025": "pre_existing",
    "s01_C_creak_C027": "pre_existing",
    "s01_C_creak_C029": "newly_crossed",
    "s01_C_creak_C031": "pre_existing",
    "s01_C_creak_C039": "pre_existing",
    "s01_C_creak_D010": "pre_existing",
    "s01_C_creak_D011": "pre_existing",
    "s01_C_creak_D015": "pre_existing",
    "s01_C_creak_D019": "pre_existing",
    "s01_C_creak_D022": "pre_existing",
    "s01_C_creak_D023": "no_v1_baseline",
    "s01_C_creak_D025": "pre_existing",
    "s01_C_creak_D027": "no_v1_baseline",
    "s01_C_creak_D031": "pre_existing",
    "s01_C_creak_D035": "no_v1_baseline",
    "s01_C_creak_F007": "pre_existing",
    "s01_C_creak_F010": "pre_existing",
    "s01_C_creak_F013": "pre_existing",
    "s01_C_creak_F017": "pre_existing",
    "s01_C_creak_F021": "pre_existing",
    "s01_C_creak_F023": "pre_existing",
    "s02_C_creak_C001": "no_v1_baseline",
    "s02_C_creak_C002": "pre_existing",
    "s02_C_creak_C005": "pre_existing",
    "s02_C_creak_C010": "pre_existing",
    "s02_C_creak_C012": "pre_existing",
    "s02_C_creak_C018": "no_v1_baseline",
    "s02_C_creak_C025": "pre_existing",
    "s02_C_creak_C026": "pre_existing",
    "s02_C_creak_C027": "pre_existing",
    "s02_C_creak_C029": "pre_existing",
    "s02_C_creak_C031": "pre_existing",
    "s02_C_creak_C033": "pre_existing",
    "s02_C_creak_C038": "newly_crossed",
    "s02_C_creak_D001": "pre_existing",
    "s02_C_creak_D002": "pre_existing",
    "s02_C_creak_D005": "pre_existing",
    "s02_C_creak_D006": "no_v1_baseline",
    "s02_C_creak_D007": "pre_existing",
    "s02_C_creak_D009": "newly_crossed",
    "s02_C_creak_D011": "no_v1_baseline",
    "s02_C_creak_D015": "pre_existing",
    "s02_C_creak_D017": "pre_existing",
    "s02_C_creak_D022": "pre_existing",
    "s02_C_creak_D024": "pre_existing",
    "s02_C_creak_D025": "pre_existing",
    "s02_C_creak_D026": "pre_existing",
    "s02_C_creak_D031": "pre_existing",
    "s02_C_creak_F001": "pre_existing",
    "s02_C_creak_F005": "pre_existing",
    "s02_C_creak_F009": "pre_existing",
    "s02_C_creak_F013": "pre_existing",
    "s02_C_creak_F015": "no_v1_baseline",
    "s02_C_creak_F016": "pre_existing",
    "s02_C_creak_F018": "pre_existing",
    "s02_C_creak_F019": "pre_existing",
    "s02_C_creak_F020": "pre_existing",
    "s02_C_creak_F021": "pre_existing",
    "s02_C_creak_F023": "no_v1_baseline",
}

pytestmark = pytest.mark.skipif(
    not RESULTS_CSV.exists(),
    reason=f"{RESULTS_CSV} not found -- run src/06_measure.py first",
)


def _to_float(value):
    try:
        v = float(value)
        return v if v == v else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def _load_rows() -> list[dict]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _qualifying_rows() -> list[dict]:
    """C_creak rows with shr_median > SHR_THRESHOLD and a usable h1_h2_db_mean."""
    out = []
    for row in _load_rows():
        if row.get("pass") != "C_creak":
            continue
        shr = _to_float(row.get("shr_median"))
        if shr is None or shr <= SHR_THRESHOLD:
            continue
        if _to_float(row.get("h1_h2_db_mean")) is None:
            continue
        out.append(row)
    return out


_QUALIFYING = _qualifying_rows() if RESULTS_CSV.exists() else []


def test_qualifying_rows_found():
    """Guards against silent 0-item parametrization below: if the C_creak
    pass name or shr_median/h1_h2_db_mean columns get renamed, the
    parametrized test would silently collect zero cases and this
    physiological check would stop running without anyone noticing."""
    assert _QUALIFYING, (
        "no C_creak rows with shr_median > 0.45 found -- check that the "
        "'pass', 'shr_median', and 'h1_h2_db_mean' columns still exist "
        f"and are named as expected in {RESULTS_CSV}")


@pytest.mark.parametrize("row", _QUALIFYING, ids=lambda r: r["utt_id"])
def test_strong_period_doubling_does_not_read_breathy(row):
    utt_id = row["utt_id"]
    h1_h2 = _to_float(row["h1_h2_db_mean"])
    shr = _to_float(row["shr_median"])

    if utt_id in KNOWN_RESIDUAL_FAILURES and h1_h2 > H1_H2_MAX_DB:
        pytest.xfail(
            f"{utt_id}: h1_h2_db_mean={h1_h2:.2f} dB, shr_median={shr:.3f} "
            f"-- known residual failure ({KNOWN_RESIDUAL_FAILURES[utt_id]}), "
            f"see module docstring")

    assert h1_h2 <= H1_H2_MAX_DB, (
        f"{utt_id}: h1_h2_db_mean={h1_h2:.2f} dB with shr_median={shr:.3f} "
        f"(> {SHR_THRESHOLD}) -- strong period doubling reading breathier "
        f"than modal voice is physiologically incoherent (creak_cal "
        f"calibration clips read -4.8 to -5.2 dB for deliberate maximal "
        f"creak); this utt_id is not in KNOWN_RESIDUAL_FAILURES, so this "
        f"is a NEW violation -- investigate before adding it there.")


if __name__ == "__main__":
    if not RESULTS_CSV.exists():
        print(f"SKIP: {RESULTS_CSV} not found")
        raise SystemExit(0)
    failures = []
    for row in _QUALIFYING:
        h1_h2 = _to_float(row["h1_h2_db_mean"])
        if h1_h2 > H1_H2_MAX_DB and row["utt_id"] not in KNOWN_RESIDUAL_FAILURES:
            failures.append(row["utt_id"])
    print(f"{len(_QUALIFYING)} qualifying row(s), "
          f"{len(KNOWN_RESIDUAL_FAILURES)} known residual failure(s), "
          f"{len(failures)} NEW/unexplained violation(s)")
    if failures:
        print("NEW violations:", failures)
    raise SystemExit(1 if failures else 0)
