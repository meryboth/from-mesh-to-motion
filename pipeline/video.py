"""Sampling keyframes out of the animated triptych.

The video stage returns a clip; the sheet needs N stills. Which N matters: a
keyframe sheet is only useful to a rigging tool if the frames are evenly spaced
in time and the first one is the rest pose it already knows.

ffmpeg is used when present because it handles every container the cloud models
emit. imageio-ffmpeg is the fallback so the repo still runs on a machine without
a system ffmpeg.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List


def ffprobe_meta(path: str) -> dict:
    """Frame count, fps and duration, or {} if ffprobe is unavailable."""
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    cmd = [
        exe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,width,height",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return {}
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fps = _parse_rate(stream.get("avg_frame_rate", "0/0"))
    nb = stream.get("nb_frames")
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    if nb in (None, "N/A") and fps and duration:
        nb = int(round(fps * duration))
    return {
        "fps": fps,
        "frames": int(nb) if nb not in (None, "N/A") else None,
        "duration_s": duration,
        "width": stream.get("width"),
        "height": stream.get("height"),
    }


def _parse_rate(rate: str) -> float:
    try:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def extract_all(path: str, out_dir: str) -> List[str]:
    """Decode every frame to PNG. Sampling happens afterwards, on the list."""
    os.makedirs(out_dir, exist_ok=True)
    exe = shutil.which("ffmpeg")
    if exe:
        pattern = os.path.join(out_dir, "frame_%05d.png")
        subprocess.run(
            [exe, "-y", "-v", "error", "-i", path, "-vsync", "0", pattern],
            check=True,
        )
    else:
        _extract_imageio(path, out_dir)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )


def _extract_imageio(path: str, out_dir: str) -> None:
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Neither ffmpeg nor imageio is available. Install one:\n"
            "  pip install imageio imageio-ffmpeg"
        ) from exc
    from PIL import Image

    for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
        Image.fromarray(frame).save(
            os.path.join(out_dir, "frame_" + str(i + 1).zfill(5) + ".png")
        )


def pick_keyframes(frames: List[str], n: int, include_last: bool = True) -> List[int]:
    """Indices of N evenly spaced frames.

    Index 0 is always included: it is the rest pose the rig already matches, so
    it anchors the whole sequence. `include_last=False` is for a looping motion,
    where the final frame would duplicate the first.
    """
    total = len(frames)
    if total == 0:
        raise ValueError("No frames to sample")
    if n >= total:
        return list(range(total))
    if n == 1:
        return [0]
    divisor = (n - 1) if include_last else n
    return [min(total - 1, round(i * (total - 1) / divisor)) for i in range(n)]
