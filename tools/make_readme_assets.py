from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GitHub-friendly preview assets")
    parser.add_argument("--video", default="Figs/Demonstration.mov")
    parser.add_argument("--preview", default="Figs/Demonstration_preview.gif")
    parser.add_argument("--mp4", default="Figs/Demonstration.mp4")
    parser.add_argument("--seconds", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    source = Path(args.video)
    if not source.exists():
        raise FileNotFoundError(source)

    preview = Path(args.preview)
    mp4 = Path(args.mp4)
    preview.parent.mkdir(parents=True, exist_ok=True)

    palette = preview.with_suffix(".palette.png")
    filter_base = f"fps=10,scale=960:-1:flags=lanczos"
    run([
        "ffmpeg", "-y", "-t", str(args.seconds), "-i", str(source),
        "-vf", f"{filter_base},palettegen=stats_mode=diff", str(palette),
    ])
    run([
        "ffmpeg", "-y", "-t", str(args.seconds), "-i", str(source),
        "-i", str(palette), "-lavfi",
        f"{filter_base}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(preview),
    ])
    palette.unlink(missing_ok=True)

    # H.264 MP4 is more broadly compatible across browsers than arbitrary MOV
    # codecs. The MOV remains the original downloadable demonstration.
    run([
        "ffmpeg", "-y", "-i", str(source),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", str(mp4),
    ])
    print(f"Created {preview}")
    print(f"Created {mp4}")


if __name__ == "__main__":
    main()
