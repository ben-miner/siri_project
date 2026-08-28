# CreakASR

Transcribes the project's WAV corpus with Apple's Speech framework and
writes `hypotheses.csv` for word-error-rate analysis against the
hand-corrected reference transcripts in `results/manifest.csv`.

## Why SFSpeechRecognizer, not SpeechAnalyzer

The original plan was `SpeechAnalyzer`/`SpeechTranscriber` (macOS 26+).
This project's only available Mac is an Intel iMac (2019) on macOS
Sequoia 15.7.9, which cannot run macOS 26 and therefore cannot run
`SpeechAnalyzer`. `CreakASR` uses `SFSpeechRecognizer` instead, with
`requiresOnDeviceRecognition = true` so recognition never leaves the
device.

The recognizer sits behind a `Transcribing` protocol
(`Sources/CreakASR/Transcribing.swift`) specifically so this substitution
doesn't have to be permanent: `SFSpeechTranscriber`
(`Sources/CreakASR/SFSpeechTranscriber.swift`) is the only file that knows
about `SFSpeechRecognizer`. If a Mac that can run `SpeechAnalyzer` becomes
available later, add a `SpeechAnalyzerTranscriber` conforming to the same
protocol in its own file, then change the one line in `main.swift` that
constructs the transcriber (`let transcriber: Transcribing = try
SFSpeechTranscriber(...)`). Argument parsing, checkpointing, retry/backoff,
and CSV writing in `main.swift` don't need to change.

## Contract

- Input: a manifest CSV with `utt_id` and `wav_path` columns (other
  columns are ignored, so `results/manifest.csv` works as-is). `wav_path`
  is resolved relative to `--audio-root`.
- Output: `<output-dir>/hypotheses.csv` with columns `utt_id, hypothesis,
  os_version, recognizer, elapsed_ms`, plus `<output-dir>/failures.log`
  recording any file that failed after retries (not every failure ends up
  as a row in `hypotheses.csv` -- that's intentional, see Resume below).

## Prerequisites (one-time, on the Mac)

1. **On-device dictation must be downloaded for the target locale**
   (default `en-US`). System Settings > Keyboard > Dictation > turn on
   Dictation, and enable "Offline"/on-device recognition for the
   language. `SFSpeechTranscriber.prepare()` checks
   `supportsOnDeviceRecognition` at startup and fails fast with a message
   pointing here if it's not set up.
2. **Grant Speech Recognition permission when prompted** the first time
   you run the tool. If no prompt appears, see Troubleshooting below.
3. Xcode command-line tools (for `swift build`/`swift run`). Any recent
   Xcode on Sequoia includes a new enough Swift toolchain for this
   package (`swift-tools-version:5.9`, deployment target macOS 13+).

## Build and run

Use absolute paths for `--manifest`/`--output`/`--audio-root` so behavior
doesn't depend on which directory you happen to `cd` into. Substitute
your actual checkout location for `~/siri_project` below.

```sh
cd ~/siri_project/swift/CreakASR
swift build -c release
```

Test on 5 files first:

```sh
swift run -c release CreakASR \
  --manifest ~/siri_project/results/manifest.csv \
  --output ~/siri_project/results/hypotheses_test \
  --audio-root ~/siri_project \
  --limit 5
```

Check `~/siri_project/results/hypotheses_test/hypotheses.csv` looks right
(5 rows, plausible transcripts), then run the full 600:

```sh
swift run -c release CreakASR \
  --manifest ~/siri_project/results/manifest.csv \
  --output ~/siri_project/results/hypotheses \
  --audio-root ~/siri_project
```

If a run gets interrupted (crash, killed terminal, sleep), just re-run the
same command -- see Resume below.

### Flags

| flag | required | default | meaning |
|---|---|---|---|
| `--manifest <path>` | yes | -- | Manifest CSV with `utt_id`, `wav_path`. |
| `--output <dir>` | yes | -- | Output directory (created if missing). |
| `--audio-root <dir>` | no | current directory | Base dir `wav_path` is resolved against. |
| `--limit <N>` | no | (all rows) | Only process the first N manifest rows. |
| `--delay <seconds>` | no | `0` | Sleep this long between files. |
| `--locale <id>` | no | `en-US` | BCP-47 locale passed to `SFSpeechRecognizer`. |

## Resume / checkpointing

Every successfully transcribed file is written to `hypotheses.csv`
immediately (append + the file handle is never buffered past a single
row) -- that's the checkpoint. On startup, `CreakASR` reads whatever
`utt_id`s are already in `hypotheses.csv` at `--output` and skips them.
Re-running the exact same command after a crash resumes from wherever it
stopped instead of re-transcribing everything.

Failed files (audio missing, or recognition failed after all retries) are
**not** written to `hypotheses.csv` -- they're logged to `failures.log`
instead and the row stays absent. That means a failed file is
automatically retried on the next full re-run (it's not in the checkpoint
set), which is what you want for transient throttling-style failures.
If a file fails for a structural reason (corrupt WAV, wrong path), it'll
just keep failing and logging on every re-run until you fix the
underlying problem -- check `failures.log` if a run finishes with fewer
than 600 rows in `hypotheses.csv`.

## Retry / backoff / throttling

`SFSpeechRecognizer` has an undocumented-in-detail per-request duration
limit and can throttle under sustained batch use. `CreakASR` handles this
defensively rather than trying to special-case exact error codes Apple
doesn't fully document:

- Each file gets up to 5 attempts with exponential backoff (2s, 4s, 8s,
  16s between attempts, capped at 30s).
- Each individual attempt is bounded to 90s (generous for this project's
  few-second clips); a stalled request is treated as a failure and
  retried rather than hanging the batch loop forever.
- Worst case, a single truly-stuck file costs roughly 8 minutes before
  it's logged as failed and the loop moves on to the next file -- it
  cannot stall the whole 600-file run indefinitely.
- If throttling shows up on the full run (failures clustering together,
  or retries consistently needed), pass `--delay 2` (or higher) to space
  requests out. `--delay` is separate from the retry backoff -- it's a
  fixed pause after every file, success or failure.

## Troubleshooting

**`requestAuthorization` never prompts, or immediately returns
`denied`.** Apple's Speech framework requires
`NSSpeechRecognitionUsageDescription` in an `Info.plist` even for a
command-line tool with no `.app` bundle -- without it, authorization
tends to fail closed instead of asking the user. `Package.swift` embeds
`Sources/CreakASR/Info.plist` into the built binary via a linker
`-sectcreate` flag to work around this. This is the one part of this
package I could not verify by actually running it (no Mac access from
here), so if authorization still misbehaves:

1. Confirm the plist actually landed in the binary:
   ```sh
   otool -s __TEXT __info_plist .build/release/CreakASR
   ```
   If this prints nothing (or errors), the embed didn't work -- check
   that `swift build` was run from `swift/CreakASR` (so `Package.swift`'s
   `#filePath`-derived path resolves correctly).
2. Check System Settings > Privacy & Security > Speech Recognition --
   the permission may be attributed to your terminal app (Terminal.app,
   iTerm, etc.) rather than to `CreakASR` itself; enable it there if
   present but off.
3. As a last resort, wrapping `CreakASR` in a minimal signed `.app`
   bundle via Xcode (rather than running the bare SwiftPM executable)
   sidesteps the whole issue, since a real bundle gets its
   `Info.plist`/TCC identity for free.

## Structure

- `main.swift` -- argument parsing, checkpoint/resume, retry/backoff, the
  batch loop, and CSV I/O.
- `Manifest.swift` -- reads the input manifest and the existing
  `hypotheses.csv` (for resume), both by CSV header name.
- `CSV.swift` -- shared quote-aware CSV parse/escape helpers (transcript
  text can contain commas and quotes, so this isn't optional).
- `Transcribing.swift` -- the protocol described above.
- `SFSpeechTranscriber.swift` -- the current `Transcribing` implementation.
