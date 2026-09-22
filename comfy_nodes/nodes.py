"""ComfyUI nodes for the mesh -> keyframe-sheet pipeline.

Eight nodes, in the order a graph uses them:

  MotionPrompt      the action alone   -> the full, panel-locking video prompt
  TriptychCompose   three ortho views  -> one canvas + its layout
  TriptychRegister  canvas + frame 0   -> the layout corrected for what came back
  SampleKeyframes   video frames       -> N evenly spaced stills
  TriptychSplit     stills + layout    -> three per-view batches
  KeyframeSheet     three batches      -> the labelled contact sheet
  SaveKeyframeManifest                 -> keyframes.json + handoff.txt
  IdentityQA        three batches      -> drift report and a pass/fail

The layout travels between them as a JSON string (FMM_LAYOUT) so it survives a
save/reload of the graph and can be read by a human when a run looks wrong.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from ..pipeline import manifest as manifest_mod
from ..pipeline import prompt as prompt_mod
from ..pipeline import repair as repair_mod
from ..pipeline import sheet as sheet_mod
from ..pipeline import video as video_mod

CATEGORY = "from-mesh-to-motion"


# ----------------------------------------------------------------- helpers


def to_pil(image: torch.Tensor, index: int = 0) -> Image.Image:
    arr = image[index].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode)


def to_tensor(images) -> torch.Tensor:
    if isinstance(images, Image.Image):
        images = [images]
    arrays = []
    for im in images:
        arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
        arrays.append(arr)
    return torch.from_numpy(np.stack(arrays))


def parse_color(text: str, fallback=(228, 228, 230)) -> tuple:
    text = (text or "").strip()
    if not text:
        return fallback
    if text.startswith("#"):
        text = text[1:]
        if len(text) == 6:
            return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) == 3:
        return tuple(max(0, min(255, int(float(p)))) for p in parts)
    return fallback


# --------------------------------------------------------- 0. motion prompt


class MotionPrompt:
    """Write only the action; this builds the rest of the video prompt.

    The prompt the video model needs is ~140 words, of which ~20 describe the
    animation. The other 120 lock the panels down, and they are the reason the
    three views stay in agreement. Hand-editing one sentence out of the middle of
    that block is a trap: break the scaffolding and nothing errors, the sheet just
    comes back with panels that quietly disagree.

    Wire `layout` in and the view names and background description are taken from
    the triptych that was actually built, so the prompt cannot drift away from the
    image it describes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ("STRING", {
                    "multiline": True,
                    "default": "raises both arms straight overhead, holds for a "
                               "moment, then lowers them back down",
                    "tooltip": "Only what the character does. No camera or layout "
                               "instructions -- those are added for you, in the "
                               "order that was measured to work.",
                }),
                "subject": ("STRING", {
                    "default": "clay toy character",
                    "tooltip": "How to refer to the character, e.g. 'clay toy "
                               "character', 'four-legged creature'.",
                }),
                "loop": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "layout": ("FMM_LAYOUT",),
                "view_names": ("STRING", {
                    "default": "front,side,back",
                    "tooltip": "Used only when no layout is connected.",
                }),
                "extra": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Any further constraint, appended at the end.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "summary")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, action, subject, loop, layout=None, view_names="front,side,back",
            extra=""):
        background = None
        views = [v.strip() for v in view_names.split(",") if v.strip()]
        if layout:
            doc = json.loads(layout)
            views = [p["view"] for p in doc.get("panels", [])] or views
            background = doc.get("background")

        text = prompt_mod.build(
            action=action, views=views, subject=subject,
            background=background, loop=bool(loop), extra=extra,
        )
        return (text, prompt_mod.summarise(action))


# ------------------------------------------------------------ 1. compose


class TriptychCompose:
    """Lay three orthographic views onto one canvas.

    Everything downstream depends on the three views having been rendered with
    one camera scale and one pivot -- this node concatenates, it does not fit.
    If the views disagree in size it raises rather than silently rescaling, since
    a rescale here is invisible and poisons every pose read off the sheet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view_a": ("IMAGE",),
                "view_b": ("IMAGE",),
                "view_c": ("IMAGE",),
                "names": ("STRING", {"default": "front,side,back"}),
                "gutter": ("INT", {"default": 20, "min": 0, "max": 256}),
                "pad": ("INT", {"default": 20, "min": 0, "max": 256}),
                "background": ("STRING", {"default": "#E4E4E6"}),
                "target_ratio": ("STRING", {
                    "default": "21:9,16:9,4:3,1:1",
                    "tooltip": "Menu of aspect ratios; the closest to the strip is "
                               "used and the canvas is padded to it. Partner video "
                               "APIs centre-crop an off-menu canvas, which eats the "
                               "outer panels.",
                }),
                "crop_margin": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FMM_LAYOUT")
    RETURN_NAMES = ("triptych", "layout")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, view_a, view_b, view_c, names, gutter, pad, background,
            target_ratio, crop_margin):
        order = [n.strip() for n in names.split(",") if n.strip()][:3]
        while len(order) < 3:
            order.append("view" + str(len(order)))

        pils = [to_pil(view_a), to_pil(view_b), to_pil(view_c)]
        sizes = {p.size for p in pils}
        if len(sizes) != 1:
            raise ValueError(
                "The three views differ in size " + repr(sizes) + ". They must come "
                "from one orthographic camera at one scale; re-render them together."
            )

        bg = parse_color(background)
        crop = None
        if crop_margin >= 0:
            try:
                crop = sheet_mod.common_crop_on_bg(pils, bg, margin=crop_margin)
            except ValueError:
                # A view that is entirely background (an empty render) is not
                # worth guessing about -- fall through and keep the full frame.
                crop = None

        tmp = _scratch_dir()
        paths = {}
        for name, pil in zip(order, pils):
            p = os.path.join(tmp, "view_" + name + ".png")
            pil.save(p)
            paths[name] = p

        out_path = os.path.join(tmp, "triptych.png")
        layout = sheet_mod.compose(
            paths, order, out_path, gutter=gutter, pad=pad,
            background=bg, crop=crop, target_ratio=sheet_mod.parse_ratio(target_ratio),
        )
        return (to_tensor(Image.open(out_path)), json.dumps(layout.to_json()))






def _scratch_dir() -> str:
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "_tmp")
    path = os.path.join(base, "from_mesh_to_motion")
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------- 2. register


class TriptychRegister:
    """Correct the layout for however the video model reshaped the canvas.

    Video models do not return what they were given. A 1904x816 canvas came back
    as 1536x672 in testing -- a 2% non-uniform rescale -- and providers that
    letterbox or crop move the panels far more. Splitting on the original pixel
    columns then cuts the wrong part of the frame.

    The first frame of an image-to-video clip is very nearly the submitted image,
    so the mapping can be read off the two content bounding boxes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "triptych": ("IMAGE",),
                "first_frame": ("IMAGE",),
                "layout": ("FMM_LAYOUT",),
            }
        }

    RETURN_TYPES = ("FMM_LAYOUT", "STRING")
    RETURN_NAMES = ("layout", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, triptych, first_frame, layout):
        lay = sheet_mod.TriptychLayout.from_json(json.loads(layout))
        tmp = _scratch_dir()
        ref_p = os.path.join(tmp, "_reg_ref.png")
        frm_p = os.path.join(tmp, "_reg_frame.png")
        to_pil(triptych).save(ref_p)
        to_pil(first_frame).save(frm_p)

        sx, sy, ox, oy = sheet_mod.register(ref_p, frm_p, lay.background)
        skew = abs(sx - sy) / max(sx, sy)
        report = (
            "scale " + format(sx, ".4f") + " x " + format(sy, ".4f")
            + "   offset " + format(ox, ".1f") + ", " + format(oy, ".1f")
            + "   skew " + format(skew * 100, ".2f") + "%"
        )
        if skew > 0.02:
            report += "\nWARNING: over 2% non-uniform scale; check the first split."

        doc = lay.to_json()
        doc["transform"] = [sx, sy, ox, oy]
        return (json.dumps(doc), report)


# -------------------------------------------------------------- 3. sample


class SampleKeyframes:
    """Pick N evenly spaced frames out of the clip.

    Index 0 is always kept: it is the rest pose the rig already matches, so it
    anchors the sequence. A looping motion drops the final frame instead, since
    it would duplicate the first.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "count": ("INT", {"default": 16, "min": 2, "max": 64}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "loop": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "fps_in": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "Wire Get Video Components' frame-rate output here; it "
                               "overrides the widget so the timings match the clip "
                               "that actually came back.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("keyframes", "timings")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, count, fps, loop, fps_in=None):
        if fps_in:
            fps = float(fps_in)
        total = frames.shape[0]
        picks = video_mod.pick_keyframes(list(range(total)), count, include_last=not loop)
        timings = [
            {"index": i, "frame": int(f), "t": round(float(f) / fps, 4)}
            for i, f in enumerate(picks)
        ]
        return (frames[picks], json.dumps(timings))


# --------------------------------------------------------------- 4. split


class TriptychSplit:
    """Cut every frame back into three per-view batches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "layout": ("FMM_LAYOUT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("view_a", "view_b", "view_c", "names")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, layout):
        doc = json.loads(layout)
        lay = sheet_mod.TriptychLayout.from_json(doc)
        transform = doc.get("transform")

        per_view = {p.view: [] for p in lay.panels}
        for i in range(frames.shape[0]):
            img = to_pil(frames, i)
            if transform:
                img = sheet_mod.apply_registration(img, tuple(transform), lay)
            elif img.size != (lay.width, lay.height):
                img = img.resize((lay.width, lay.height), Image.LANCZOS)
            for p in lay.panels:
                per_view[p.view].append(img.crop((p.x, p.y, p.x + p.w, p.y + p.h)))

        order = [p.view for p in lay.panels]
        batches = [to_tensor(per_view[v]) for v in order]
        while len(batches) < 3:
            batches.append(batches[-1])
        return (batches[0], batches[1], batches[2], ",".join(order))


# --------------------------------------------------------------- 5. sheet


class KeyframeSheet:
    """The labelled N x 3 contact sheet.

    Labels are added here and nowhere earlier. Text inside an image handed to a
    video or edit model gets garbled and then regenerated as nonsense, so the
    sheet stays clean until the last step.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view_a": ("IMAGE",),
                "view_b": ("IMAGE",),
                "view_c": ("IMAGE",),
                "names": ("STRING", {"default": "front,side,back"}),
                "timings": ("STRING", {"default": "", "multiline": True}),
                "thumb": ("INT", {"default": 256, "min": 96, "max": 768}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, view_a, view_b, view_c, names, timings, thumb):
        order = [n.strip() for n in names.split(",") if n.strip()][:3]
        while len(order) < 3:
            order.append("view" + str(len(order)))

        try:
            times = json.loads(timings) if timings.strip() else []
        except json.JSONDecodeError:
            times = []

        tmp = _scratch_dir()
        batches = [view_a, view_b, view_c]
        n = min(b.shape[0] for b in batches)

        rows = []
        for i in range(n):
            views = {}
            for view, batch in zip(order, batches):
                p = os.path.join(tmp, "sheet_" + str(i).zfill(2) + "_" + view + ".png")
                to_pil(batch, i).save(p)
                views[view] = p
            row = {"index": i, "views": views}
            if i < len(times):
                row["t"] = times[i].get("t")
            rows.append(row)

        out_path = os.path.join(tmp, "keyframe_sheet.png")
        sheet_mod.contact_sheet(rows, order, out_path, thumb=thumb)
        return (to_tensor(Image.open(out_path)),)


# ------------------------------------------------------------ 6. manifest


class SaveKeyframeManifest:
    """Write keyframes.json and the paste-into-Astra handoff text.

    A folder of PNGs is not a deliverable: the rigging tool needs which pose
    happens when, which view is which, and what the motion was meant to be.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view_a": ("IMAGE",),
                "view_b": ("IMAGE",),
                "view_c": ("IMAGE",),
                "names": ("STRING", {"default": "front,side,back"}),
                "timings": ("STRING", {"default": "", "multiline": True}),
                "subject_name": ("STRING", {"default": "character"}),
                "motion_description": ("STRING", {"default": "", "multiline": True}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "loop": ("BOOLEAN", {"default": False}),
                "output_dir": ("STRING", {"default": "from_mesh_to_motion"}),
            },
            "optional": {
                "sheet": ("IMAGE",),
                "fps_in": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "handoff_text")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, view_a, view_b, view_c, names, timings, subject_name,
            motion_description, fps, loop, output_dir, sheet=None, fps_in=None):
        if fps_in:
            fps = float(fps_in)
        order = [n.strip() for n in names.split(",") if n.strip()][:3]
        while len(order) < 3:
            order.append("view" + str(len(order)))

        try:
            import folder_paths
            base = os.path.join(folder_paths.get_output_directory(), output_dir)
        except Exception:
            base = os.path.abspath(output_dir)
        views_dir = os.path.join(base, "views")
        os.makedirs(views_dir, exist_ok=True)

        try:
            times = json.loads(timings) if timings.strip() else []
        except json.JSONDecodeError:
            times = []

        batches = [view_a, view_b, view_c]
        n = min(b.shape[0] for b in batches)

        # Timing is the one thing the rigging tool cannot recover from the images,
        # so there is no fallback. An earlier version guessed t = index / fps when
        # this input was left unwired, which put k01 at 0.04 s instead of 0.33 s
        # and handed Astra a sheet whose timing was eight times off -- plausibly, silently.
        if len(times) < n:
            raise ValueError(
                "Save Keyframe Manifest got timings for " + str(len(times)) + " of "
                + str(n) + " keyframes. Wire Sample Keyframes' `timings` output into "
                "this node's `timings` input: without it the manifest would have to "
                "guess when each pose happens, and a guess there is worse than nothing."
            )

        keyframes = []
        for i in range(n):
            stem = "k" + str(i).zfill(2)
            written = {}
            for view, batch in zip(order, batches):
                p = os.path.join(views_dir, stem + "_" + view + ".png")
                to_pil(batch, i).save(p)
                written[view] = p
            keyframes.append({"index": i, "t": times[i].get("t"),
                              "frame": times[i].get("frame"), "views": written})

        sheet_path = None
        if sheet is not None:
            sheet_path = os.path.join(base, "keyframe_sheet.png")
            to_pil(sheet).save(sheet_path)

        doc = manifest_mod.build(
            subject={"name": subject_name},
            motion={"description": motion_description, "fps": fps, "loop": loop},
            views=order,
            keyframes=keyframes,
            provenance={"generator": "from-mesh-to-motion (ComfyUI)"},
            sheet_path=sheet_path,
        )
        man_path = manifest_mod.write(doc, os.path.join(base, "keyframes.json"))

        text = manifest_mod.handoff_text(doc)
        with open(os.path.join(base, "handoff.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        return (man_path, text)


class IdentityQA:
    """Score the finished sheet for identity drift, inside the graph.

    The same measurement as `python -m pipeline qa`, so a sheet can be checked
    without leaving ComfyUI. Each view's palette comes from its own k00, which has
    been through the same video encoder as every other frame -- anything k00
    shares with them is codec, not drift.

    The metric is pose-blind on purpose; see pipeline/repair.py for the version
    that was not, and why it had to go.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view_a": ("IMAGE",),
                "view_b": ("IMAGE",),
                "view_c": ("IMAGE",),
                "layout": ("FMM_LAYOUT",),
                "ceiling": ("FLOAT", {
                    "default": 0.03, "min": 0.005, "max": 0.2, "step": 0.005,
                    "tooltip": "Drift above this fails the sheet. 0.01 is codec "
                               "noise, 0.03 a tint you would notice, 0.07 a "
                               "character that has changed colour.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("report", "passed")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, view_a, view_b, view_c, layout, ceiling):
        doc = json.loads(layout)
        bg = tuple(doc.get("background", (228, 228, 230)))
        views = [p["view"] for p in doc.get("panels", [])][:3]
        while len(views) < 3:
            views.append("view" + str(len(views)))

        rows, lines = [], []
        for view, batch in zip(views, (view_a, view_b, view_c)):
            palette = repair_mod.anchor_palette(to_pil(batch, 0), bg)
            per_view = []
            for i in range(batch.shape[0]):
                row = repair_mod.assess_panel(to_pil(batch, i), palette, bg)
                row.update(index=i, view=view)
                per_view.append(row)
            rows.extend(per_view)
            s = repair_mod.summarise(per_view)
            lines.append(view.ljust(6) + " mean " + format(s["drift_mean"], ".4f")
                         + "   max " + format(s["drift_max"], ".4f"))

        verdict = repair_mod.gate(repair_mod.summarise(rows), drift_ceiling=ceiling)
        report = ("PASS" if verdict["pass"] else "FAIL") + "\n" + "\n".join(lines)
        report += "\n" + "\n".join("- " + n for n in verdict["notes"])
        return {"ui": {"text": [report]}, "result": (report, bool(verdict["pass"]))}


NODE_CLASS_MAPPINGS = {
    "FMM_MotionPrompt": MotionPrompt,
    "FMM_IdentityQA": IdentityQA,
    "FMM_TriptychCompose": TriptychCompose,
    "FMM_TriptychRegister": TriptychRegister,
    "FMM_SampleKeyframes": SampleKeyframes,
    "FMM_TriptychSplit": TriptychSplit,
    "FMM_KeyframeSheet": KeyframeSheet,
    "FMM_SaveKeyframeManifest": SaveKeyframeManifest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FMM_MotionPrompt": "Motion Prompt",
    "FMM_IdentityQA": "Identity QA",
    "FMM_TriptychCompose": "Triptych Compose",
    "FMM_TriptychRegister": "Triptych Register",
    "FMM_SampleKeyframes": "Sample Keyframes",
    "FMM_TriptychSplit": "Triptych Split",
    "FMM_KeyframeSheet": "Keyframe Sheet",
    "FMM_SaveKeyframeManifest": "Save Keyframe Manifest",
}
