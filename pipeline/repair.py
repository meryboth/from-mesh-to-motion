"""Measuring identity drift, and whether a repair pass would actually help.

Getting this metric wrong is easy and it cost this repo a wrong conclusion once
already, so the reasoning is written down rather than assumed.

**The trap.** The obvious measure is the distance between the colour histogram of
a posed panel and that of the rest pose. It does not work. Raising the character's
arms exposes more body and less backpack, which shifts the histogram hard without
anything about the character having changed. Measured on a clip whose identity
visibly holds, that metric moved by 0.298 on the back panel purely from the pose
-- larger than any real drift in the run, and pointing at the wrong panel.

**What is measured instead.** The rest pose is reduced to a handful of dominant
colours. Each panel's subject pixels are assigned to the nearest of those colours,
and the drift is how far each colour has *moved*, weighted by how much of it the
**anchor** contained. Weighting by the anchor rather than the panel is the whole
trick: a raised arm changes how much teal is visible, not what teal looks like.
Pose leak drops to 0.006-0.016, while an 8/255 red tint -- barely visible -- still
reads three times baseline.

**The pose guard.** A repair pass has a failure mode that looks like success: an
edit model asked to restore the character can restore it into a different pose.
The panel then matches the anchor beautifully and is worthless, because pose is
the only thing the sheet is authoritative about. `silhouette_iou` catches that and
nothing else does, so a repair is an improvement only if drift falls *and* iou
holds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

DEFAULT_COLORS = 6
IOU_FLOOR = 0.90


def subject_mask(img: Image.Image, bg: Sequence[int], tol: int = 18) -> np.ndarray:
    """True where the image is not the flat background colour."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    diff = np.abs(arr - np.asarray(bg, dtype=np.int16)[None, None, :])
    return diff.max(axis=2) > tol


def subject_pixels(img: Image.Image, bg: Sequence[int]) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float64)[subject_mask(img, bg)]


def anchor_palette(
    img: Image.Image,
    bg: Sequence[int],
    colors: int = DEFAULT_COLORS,
    iters: int = 12,
    seed: int = 0,
) -> tuple:
    """Reduce the rest pose to (centres, weights) by k-means on subject pixels.

    Deterministic for a given seed, because a QA number that moves between runs
    of the same data is not a QA number.
    """
    pixels = subject_pixels(img, bg)
    if len(pixels) < colors:
        raise ValueError("Anchor has too few subject pixels to build a palette.")
    rng = np.random.default_rng(seed)
    centres = pixels[rng.choice(len(pixels), colors, replace=False)].copy()

    for _ in range(iters):
        assign = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(2).argmin(1)
        for j in range(colors):
            if (assign == j).any():
                centres[j] = pixels[assign == j].mean(0)

    weights = np.array([(assign == j).sum() for j in range(colors)], dtype=np.float64)
    total = weights.sum()
    return centres, (weights / total if total else weights)


def drift(
    panel: Image.Image,
    palette: tuple,
    bg: Sequence[int],
    min_pixels: int = 40,
) -> float:
    """Mean displacement of the anchor's colours, normalised to [0, 1].

    Roughly: 0.01 is codec noise, 0.03 is a tint you would notice if you looked
    for it, 0.07 is a character that has changed colour.
    """
    centres, weights = palette
    pixels = subject_pixels(panel, bg)
    if len(pixels) == 0:
        return 1.0

    assign = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(2).argmin(1)
    total = 0.0
    weight_sum = 0.0
    for j in range(len(centres)):
        selected = pixels[assign == j]
        # A colour the pose has hidden contributes nothing rather than a spurious
        # distance measured from a handful of edge pixels.
        if len(selected) < min_pixels:
            continue
        total += weights[j] * float(np.linalg.norm(selected.mean(0) - centres[j]))
        weight_sum += weights[j]
    return float(total / weight_sum / 255.0) if weight_sum else 1.0


def silhouette_iou(before: Image.Image, after: Image.Image, bg: Sequence[int]) -> float:
    """Intersection over union of two subject silhouettes, in [0, 1].

    Images must match in size; a repair pass that returns a different resolution
    has already lost the alignment the sheet depends on, so this raises rather
    than resizing quietly and hiding it.
    """
    if before.size != after.size:
        raise ValueError(
            "Cannot compare silhouettes at different sizes "
            + repr(before.size) + " vs " + repr(after.size)
            + ": a repair pass must preserve the panel resolution."
        )
    a = subject_mask(before, bg)
    b = subject_mask(after, bg)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def assess_panel(panel, palette, bg, repaired=None) -> dict:
    """One panel's row: drift, and what a repair did to it if one was run."""
    row = {"drift": round(drift(panel, palette, bg), 4)}
    if repaired is not None:
        row["drift_repaired"] = round(drift(repaired, palette, bg), 4)
        row["pose_iou"] = round(silhouette_iou(panel, repaired, bg), 4)
        row["improved"] = bool(
            row["drift_repaired"] < row["drift"] and row["pose_iou"] >= IOU_FLOOR
        )
    return row


def summarise(rows: Sequence[dict], iou_floor: float = IOU_FLOOR) -> dict:
    """Aggregate a run into numbers that can be quoted in a write-up."""
    before = [r["drift"] for r in rows if "drift" in r]
    after = [r["drift_repaired"] for r in rows if "drift_repaired" in r]
    ious = [r["pose_iou"] for r in rows if "pose_iou" in r]

    out = {
        "panels": len(rows),
        "drift_mean": round(float(np.mean(before)), 4) if before else None,
        "drift_max": round(float(np.max(before)), 4) if before else None,
    }
    if after:
        out["drift_mean_repaired"] = round(float(np.mean(after)), 4)
        out["drift_delta"] = round(out["drift_mean_repaired"] - out["drift_mean"], 4)
        out["pose_iou_mean"] = round(float(np.mean(ious)), 4)
        out["pose_iou_min"] = round(float(np.min(ious)), 4)
        out["pose_broken"] = int(sum(1 for i in ious if i < iou_floor))
        out["helped"] = int(sum(1 for r in rows if r.get("improved")))
    return out


def gate(summary: dict, drift_ceiling: float = 0.03) -> dict:
    """Turn the numbers into a pass/fail a person can act on.

    `drift_ceiling` is set where a tint starts being visible, not at zero: the
    point of a gate is to catch sheets that are not worth rigging from, and a
    threshold nothing can clear is the same as having no gate.
    """
    verdict = {"pass": True, "notes": []}
    if summary.get("drift_max") is None:
        return {"pass": False, "notes": ["nothing measured"]}

    if summary["drift_max"] > drift_ceiling:
        verdict["pass"] = False
        verdict["notes"].append(
            "identity drift peaks at " + format(summary["drift_max"], ".3f")
            + " (ceiling " + format(drift_ceiling, ".3f") + ") -- a repair pass may help"
        )
    else:
        verdict["notes"].append(
            "identity holds: drift peaks at " + format(summary["drift_max"], ".3f")
            + ", under the " + format(drift_ceiling, ".3f") + " ceiling"
        )

    if summary.get("pose_broken"):
        verdict["pass"] = False
        verdict["notes"].append(
            str(summary["pose_broken"]) + " panel(s) lost pose during repair "
            "(IoU below " + format(IOU_FLOOR, ".2f") + ") -- discard the repair"
        )
    return verdict
