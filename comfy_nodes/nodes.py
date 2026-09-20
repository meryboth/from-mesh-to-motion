"""ComfyUI nodes for the mesh -> keyframe-sheet pipeline.

Six nodes, in the order a graph uses them:

  TriptychCompose   three ortho views  -> one canvas + its layout
  TriptychRegister  canvas + frame 0   -> the layout corrected for what came back
  SampleKeyframes   video frames       -> N evenly spaced stills
  TriptychSplit     stills + layout    -> three per-view batches
  KeyframeSheet     three batches      -> the labelled contact sheet
  SaveKeyframeManifest                 -> keyframes.json + handoff.txt

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
    as 1536x672 in testing -- not a uniform rescale. Splitting on the original
    pixel columns after a naive resize shears the panels by about 2% and cuts the
    character's feet off.

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
        keyframes = []
        for i in range(n):
            stem = "k" + str(i).zfill(2)
            written = {}
            for view, batch in zip(order, batches):
                p = os.path.join(views_dir, stem + "_" + view + ".png")
                to_pil(batch, i).save(p)
                written[view] = p
            t = times[i].get("t") if i < len(times) else i / fps
            keyframes.append({"index": i, "t": t,
                              "frame": times[i].get("frame") if i < len(times) else None,
                              "views": written})

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


NODE_CLASS_MAPPINGS = {
    "FMM_TriptychCompose": TriptychCompose,
    "FMM_TriptychRegister": TriptychRegister,
    "FMM_SampleKeyframes": SampleKeyframes,
    "FMM_TriptychSplit": TriptychSplit,
    "FMM_KeyframeSheet": KeyframeSheet,
    "FMM_SaveKeyframeManifest": SaveKeyframeManifest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FMM_TriptychCompose": "Triptych Compose",
    "FMM_TriptychRegister": "Triptych Register",
    "FMM_SampleKeyframes": "Sample Keyframes",
    "FMM_TriptychSplit": "Triptych Split",
    "FMM_KeyframeSheet": "Keyframe Sheet",
    "FMM_SaveKeyframeManifest": "Save Keyframe Manifest",
}
