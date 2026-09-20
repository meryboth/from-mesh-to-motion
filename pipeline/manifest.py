"""The handoff artifact.

A folder of PNGs is not a deliverable. What the rigging tool -- Astra, Meshy, or
a human animator -- actually needs is: which pose happens *when*, which view is
which, and what the motion was supposed to be. That is this file.

The manifest is also the provenance record: every model, seed and prompt that
touched the sheet, so a run can be re-read months later and argued with.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Sequence

SCHEMA_VERSION = "1.0"


def build(
    *,
    subject: dict,
    motion: dict,
    views: Sequence[str],
    keyframes: Sequence[dict],
    provenance: dict,
    sheet_path: str | None = None,
) -> dict:
    """Assemble the manifest dict.

    `keyframes` entries are {"index": int, "t": float, "views": {view: path}}.
    Paths are rewritten relative to the manifest so the folder stays portable.
    """
    fps = float(motion.get("fps", 24))
    duration = float(motion.get("duration_s", 0)) or (len(keyframes) / fps)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "subject": subject,
        "motion": {
            "description": motion.get("description", ""),
            "loop": bool(motion.get("loop", False)),
            "fps": fps,
            "duration_s": round(duration, 4),
            "keyframe_count": len(keyframes),
        },
        "views": list(views),
        "projection": "orthographic",
        "sheet": sheet_path,
        "keyframes": [
            {
                "index": kf["index"],
                "t": round(float(kf["t"]), 4),
                "frame": kf.get("frame"),
                "views": kf.get("views", {}),
            }
            for kf in keyframes
        ],
        "provenance": provenance,
    }


def write(manifest: dict, out_path: str) -> str:
    """Write the manifest, with every path made relative to its own folder."""
    out_path = os.path.abspath(out_path)
    base = os.path.dirname(out_path)
    os.makedirs(base, exist_ok=True)

    doc = json.loads(json.dumps(manifest))  # deep copy
    if doc.get("sheet"):
        doc["sheet"] = _rel(doc["sheet"], base)
    for kf in doc.get("keyframes", []):
        kf["views"] = {v: _rel(p, base) for v, p in kf.get("views", {}).items()}

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return out_path


def _rel(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:
        return path


HANDOFF_TEMPLATE = """\
Animate the rig you built from this mesh using the attached keyframe sheet.

The sheet has {n} keyframes down the page and {m} orthographic views across it
({views}). Every view shares one camera scale and one pivot, so a limb at a
given height in the front view is at that same height in the side and back
views. Read the pose from all {m} views together, not from the front alone.

Motion: {description}
Timing: {duration}s total at {fps} fps, keyframe k{first} at t=0.
Loop: {loop}

Do not invent poses between keyframes beyond smooth interpolation, and do not
change the character's proportions -- the mesh is the authority on those, the
sheet is the authority only on pose.
"""


def handoff_text(manifest: dict) -> str:
    """The prompt to paste alongside the sheet in Astra (or any rigging tool)."""
    motion = manifest["motion"]
    views = manifest["views"]
    kfs = manifest["keyframes"]
    return HANDOFF_TEMPLATE.format(
        n=len(kfs),
        m=len(views),
        views=", ".join(views),
        description=motion["description"] or "(unspecified)",
        duration=motion["duration_s"],
        fps=int(motion["fps"]),
        first=str(kfs[0]["index"]).zfill(2) if kfs else "00",
        loop="yes, k00 and the last keyframe should meet" if motion["loop"] else "no",
    )
