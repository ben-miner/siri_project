"""Extract channel 0 of each raw WAV to a mono file in data/converted/.

Source recordings are iPhone stereo where the two channels differ in
content (-36.6 dB difference signal) but not level (c0 -26.8 dB, c1
-27.3 dB). Averaging the channels risks phase cancellation that would
affect H1-H2, so a single channel is taken instead. Channel 0 is used
because it has the marginally higher level, and the same channel index
is used for all six takes (3 conditions x 2 sessions) so that channel
choice cannot masquerade as a condition effect.

No resampling, no loudness normalization, no denoising, and no other
filters are applied: source is 48000 Hz and must stay 48000 Hz, and
these operations would destroy the voice-quality measures this project
depends on.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_SAMPLE_RATE = "48000"
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_FMT = "s16"
EXCLUDED_DIR_NAME = "originals_m4a"


def find_source_wavs(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.rglob("*.wav")
        if p.is_file() and EXCLUDED_DIR_NAME not in p.relative_to(raw_dir).parts[:-1]
    )


def needs_conversion(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime


def convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", "pan=mono|c0=c0",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {src}: {result.stderr.strip()}")


def probe_stream(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a:0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no audio stream in {path}")
    return streams[0]


def verify_output(path: Path) -> None:
    stream = probe_stream(path)
    sample_rate = stream.get("sample_rate", "")
    channels = stream.get("channels")
    sample_fmt = stream.get("sample_fmt", "")
    if (
        sample_rate != EXPECTED_SAMPLE_RATE
        or channels != EXPECTED_CHANNELS
        or sample_fmt != EXPECTED_SAMPLE_FMT
    ):
        raise RuntimeError(
            f"{path} is not {EXPECTED_SAMPLE_RATE} Hz / mono / 16-bit: "
            f"sample_rate={sample_rate}, channels={channels}, sample_fmt={sample_fmt}"
        )


def find_output_collisions(sources: list[Path]) -> dict:
    by_stem: dict[str, list[Path]] = {}
    for src in sources:
        by_stem.setdefault(src.stem, []).append(src)
    return {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Directory to walk recursively for source WAVs (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/converted"),
        help="Directory to write mono WAVs to (default: data/converted)",
    )
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH. Install ffmpeg and ensure {tool} is available.")

    raw_dir = args.raw_dir
    if not raw_dir.is_dir():
        sys.exit(f"Raw audio directory not found: {raw_dir}")

    sources = find_source_wavs(raw_dir)
    if not sources:
        sys.exit(f"No source WAVs found under {raw_dir} (excluding {EXCLUDED_DIR_NAME}/)")

    collisions = find_output_collisions(sources)
    if collisions:
        lines = "\n".join(
            f"  {stem}: {[str(p) for p in paths]}" for stem, paths in collisions.items()
        )
        sys.exit(f"Filename stem collisions would overwrite outputs in {args.out_dir}:\n{lines}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for src in sources:
        dst = args.out_dir / f"{src.stem}.wav"
        if needs_conversion(src, dst):
            convert(src, dst)
            converted += 1
        else:
            skipped += 1

    for src in sources:
        dst = args.out_dir / f"{src.stem}.wav"
        verify_output(dst)

    print(f"Converted {converted}, skipped {skipped} (up to date), verified {len(sources)} output(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
