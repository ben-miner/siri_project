"""Inventory audio files under data/raw/ via ffprobe."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".m4a", ".aiff", ".mp3"}
LOSSY_CODECS = {"aac", "mp3", "opus"}
EXPECTED_SAMPLE_RATE = "44100"
EXPECTED_CHANNELS = 1
EXPECTED_BITS_PER_SAMPLE = 16
EXPECTED_SAMPLE_FMT = "s16"

FIELDNAMES = [
    "filename",
    "relative_path",
    "codec_name",
    "sample_rate",
    "bit_depth",
    "channels",
    "duration_sec",
    "bit_rate",
    "size_bytes",
]


def probe_audio(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "a:0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no audio stream in {path}")
    stream = streams[0]
    fmt = info.get("format", {})

    bit_depth = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or ""

    return {
        "codec_name": stream.get("codec_name", ""),
        "sample_fmt": stream.get("sample_fmt", ""),
        "sample_rate": stream.get("sample_rate", ""),
        "bit_depth": bit_depth,
        "channels": stream.get("channels", ""),
        "duration_sec": stream.get("duration") or fmt.get("duration", ""),
        "bit_rate": stream.get("bit_rate") or fmt.get("bit_rate", ""),
        "size_bytes": fmt.get("size", ""),
    }


def find_audio_files(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def is_conforming(sample_rate: str, channels: int | None, sample_fmt: str) -> bool:
    return (
        sample_rate == EXPECTED_SAMPLE_RATE
        and channels == EXPECTED_CHANNELS
        and sample_fmt == EXPECTED_SAMPLE_FMT
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Directory to walk recursively for audio files (default: data/raw)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/audio_inventory.csv"),
        help="Output CSV path (default: results/audio_inventory.csv)",
    )
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        sys.exit("ffprobe not found on PATH. Install ffmpeg and ensure ffprobe is available.")

    raw_dir = args.raw_dir
    if not raw_dir.is_dir():
        sys.exit(f"Raw audio directory not found: {raw_dir}")

    files = find_audio_files(raw_dir)
    if not files:
        sys.exit(f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found under {raw_dir}")

    rows = []
    nonconforming = []
    lossy = []

    for path in files:
        info = probe_audio(path)
        row = {
            "filename": path.name,
            "relative_path": path.relative_to(raw_dir).as_posix(),
            "codec_name": info["codec_name"],
            "sample_rate": info["sample_rate"],
            "bit_depth": info["bit_depth"],
            "channels": info["channels"],
            "duration_sec": info["duration_sec"],
            "bit_rate": info["bit_rate"],
            "size_bytes": info["size_bytes"],
        }
        rows.append(row)

        try:
            channels = int(info["channels"])
        except (TypeError, ValueError):
            channels = None

        if not is_conforming(info["sample_rate"], channels, info["sample_fmt"]):
            nonconforming.append(row)

        if info["codec_name"] in LOSSY_CODECS:
            lossy.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    for row in lossy:
        print(f"WARNING: lossy codec ({row['codec_name']}) in {row['relative_path']}")

    if nonconforming:
        print(f"WARNING: {len(nonconforming)} of {len(rows)} file(s) are NOT 44100 Hz / mono / 16-bit signed PCM:")
        for row in nonconforming:
            print(
                f"  {row['relative_path']}: "
                f"{row['sample_rate']} Hz, {row['channels']} ch, "
                f"codec={row['codec_name']}, bit_depth={row['bit_depth']}"
            )

    print(f"Wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
