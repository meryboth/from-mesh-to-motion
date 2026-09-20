"""Building the motion prompt.

The prompt handed to the video model is 140-odd words, of which about 20 are the
animation someone actually wants. The rest is scaffolding that locks the panels
down, and it is the reason the three views stay in agreement. Leaving a user to
hand-edit one sentence out of the middle of that block is a trap: break the
scaffolding and nothing errors, you just get a sheet whose panels quietly
disagree -- the exact failure this pipeline exists to prevent.

So the scaffolding lives here and the user writes only the action.

Two things in the template are load-bearing and were arrived at by running it:

1. **The action comes first.** With the layout constraints in front, an early
   "crouch and jump" prompt produced arms going overhead and feet that never left
   the ground. Moving the action to the top got a real jump. The constraints are
   necessary but they crowd out the motion when they lead.
2. **The panel roles are named by position, not just by view.** "left panel front
   view" holds; "three views of the character" does not -- the model reorders them.
"""

from __future__ import annotations

from typing import Sequence

# How each view reads in a sentence. A view not listed falls back to "<name> view".
VIEW_PHRASES = {
    "front": "front view",
    "back": "back view",
    "side": "side profile",
    "left": "left side profile",
    "right": "right side profile",
}

# Positional words for up to five panels; beyond that they are numbered.
POSITIONS = {
    2: ["left panel", "right panel"],
    3: ["left panel", "centre panel", "right panel"],
    4: ["first panel", "second panel", "third panel", "fourth panel"],
    5: ["first panel", "second panel", "third panel", "fourth panel", "fifth panel"],
}

TEMPLATE = """\
{action}

This is a {count}-panel split-screen character turnaround reference sheet on a \
{background} background. The same {subject} appears {count} times side by side, \
each panel locked to its own fixed camera angle: {panels}. The camera is \
completely static in every panel: no pan, no zoom, no dolly, no rotation, no \
cuts. The panels never move, never resize and never change their viewing angle. \
All {count} copies perform that one action in perfect unison, frame for frame, \
each seen from its own fixed angle. Each copy stays centred inside its own panel \
and never drifts out of frame. Nothing is added to the scene.{loop}{extra}"""

LOOP_CLAUSE = (
    " The motion ends in exactly the pose it started in, so the sequence loops "
    "cleanly."
)


def describe_background(rgb: Sequence[int] | None) -> str:
    """A plain-language name for the canvas colour.

    The prompt has to describe the background the model is actually looking at.
    Saying "light grey" over a dark canvas invites the model to repaint it, and a
    repainted background breaks `content_box`, which the whole split step needs.
    """
    if not rgb:
        return "flat light grey"
    r, g, b = (int(c) for c in rgb[:3])
    value = (r + g + b) / 3.0
    spread = max(r, g, b) - min(r, g, b)

    if spread > 30:
        hue = "warm" if r >= b else "cool"
        return "flat " + hue + (" light" if value > 140 else " dark") + " toned"
    # The default canvas is #E4E4E6, which averages 228 and reads as light grey to
    # a person. Calling it white invites the model to lighten the background, and
    # a repainted background breaks content_box and with it the whole split step.
    if value > 244:
        return "flat white"
    if value > 170:
        return "flat light grey"
    if value > 90:
        return "flat mid grey"
    return "flat dark grey"


def panel_clause(views: Sequence[str]) -> str:
    """'left panel front view, centre panel side profile, right panel back view'."""
    count = len(views)
    positions = POSITIONS.get(count) or [
        "panel " + str(i + 1) for i in range(count)
    ]
    parts = []
    for position, view in zip(positions, views):
        phrase = VIEW_PHRASES.get(view.lower(), view + " view")
        parts.append(position + " " + phrase)
    return ", ".join(parts)


def build(
    action: str,
    views: Sequence[str] = ("front", "side", "back"),
    subject: str = "character",
    background=None,
    loop: bool = False,
    extra: str = "",
) -> str:
    """Compose the full prompt from the one thing the user should have to write."""
    action = (action or "").strip()
    if not action:
        raise ValueError(
            "No action given. Describe what the character does, e.g. "
            '"raises its right arm and waves twice, then lowers it".'
        )
    # The template reads "...perform that one action...", which only parses if the
    # action is its own sentence.
    if not action.endswith((".", "!", "?")):
        action += "."

    extra = (extra or "").strip()
    return TEMPLATE.format(
        action=action,
        count=len(views),
        background=background if isinstance(background, str)
        else describe_background(background),
        subject=subject.strip() or "character",
        panels=panel_clause(views),
        loop=LOOP_CLAUSE if loop else "",
        extra=(" " + extra) if extra else "",
    )


def summarise(action: str) -> str:
    """The short human phrase for the manifest and the Astra handoff.

    Deliberately the same string the prompt was built from, so the sheet's
    provenance cannot claim one motion while the clip shows another.
    """
    action = (action or "").strip().rstrip(".")
    return action[:1].upper() + action[1:] if action else ""
