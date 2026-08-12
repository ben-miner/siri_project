# Project
Measuring whether creaky phonation increases word error rate in Apple's
on-device SpeechTranscriber. Single speaker (the author), scripted stimuli,
100 sentences × 3 conditions (modal/natural/creak) × 2 sessions.

# Environment
- Windows 11, Python 3.11 (conda env `phonpipe`), VSCode.
- Use pathlib. Never hardcode backslashes.
- swift/ targets macOS 26 and is built elsewhere. NEVER try to build or run
  it from Windows.
- ../- "C:\Users\benmi\PhonPhon Squib" (package `phonpipe`, installed editable
  into this conda env) is an existing validated tool of mine. Read its
  README and phonpipe/measures/creak.py before writing any acoustic
  measurement code. Do not reimplement what it provides.
- NOTE: that path contains a space. Always quote it in shell commands and
  use pathlib for any path construction.

# Non-negotiables
- utt_id is the join key everywhere. Never join on list position.
- No audio filtering, normalization, or noise reduction. Ever. It destroys
  the voice-quality measures this project depends on.
- Reference transcripts come from the read script, corrected by ear. NEVER
  from any ASR output (not Voice Memos, not Whisper).
- config/thresholds.yaml is frozen once committed. Do not modify it.
- Write to results/ as CSV. Never overwrite raw or converted audio.
- This project is GPL v3 (phonpipe dependency embeds Praat).

# Style
Small scripts with a main() and argparse. No notebooks in src/.
Every script prints a one-line summary of what it wrote.
pytest tests for anything with parsing or arithmetic.
Fail loudly on data mismatches. Never silently proceed.