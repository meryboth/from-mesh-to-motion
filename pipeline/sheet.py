"""Triptych composition and decomposition.

The triptych is the load-bearing idea of this pipeline. Three orthographic views
are laid out on one canvas so that a video model animates them *inside a single
frame*. Panels in one frame share the model's attention, so they stay in step
with each other; three separate clips do not, and drift apart within a few
frames.

Two rules make the round trip lossless:

1. Panels are never re-cropped. The Blender renderer already gave every view the
   same ortho scale and the same pivot, so the panels are concatenated as-is.
   Trimming each view to its own content would destroy that shared scale.
2. Panel geometry is recorded in a sidecar JSON. The split step reads it back
   rather than re-deriving it, so a resized or letterboxed video still cuts on
   the right pixel columns.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFont

DEFAULT_BG = (228, 228, 230)
# The aspect ratios partner video APIs actually accept. compose() picks the
# one closest to the strip it was given, so the padding stays minimal.
DEFAULT_RATIOS = [21 / 9, 16 / 9, 4 / 3, 1.0]
DEFAULT_GUTTER_BG = (228, 228, 230)


@dataclass
class PanelBox:
    view: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class TriptychLayout:
    width: int
    height: int
    gutter: int
    pad: int
    background: tuple
    panels: list

    def to_json(self) -> dict:
        d = asdict(self)
        d["background"] = list(self.background)
        d["panels"] = [asdict(p) if not isinstance(p, dict) else p for p in self.panels]
        return d

    @staticmethod
    def from_json(d: dict) -> "TriptychLayout":
        return TriptychLayout(
            width=d["width"],
            height=d["height"],
            gutter=d["gutter"],
            pad=d["pad"],
            background=tuple(d["background"]),
            panels=[PanelBox(**p) for p in d["panels"]],
        )


def flatten(img: Image.Image, bg: Sequence[int]) -> Image.Image:
    """Composite RGBA over a flat colour. Video models expect opaque frames."""
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, tuple(bg) + (255,))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def common_crop(
    view_paths: Iterable[str],
    margin: float = 0.06,
    align: int = 8,
) -> tuple:
    """One crop window that fits the subject in *every* view.

    A character is tall and narrow, so three square renders side by side land
    near 2.9:1 -- wide enough that video models letterbox or centre-crop it and
    quietly destroy a panel. Cropping each view to the union of all their alpha
    bounding boxes brings the strip close to 16:9 while keeping the scale shared:
    the same pixel window is applied to every view, so nothing is rescaled.

    Taking the union (not each view's own box) is what preserves that. The side
    view of a character with a backpack is wider than the front; cropping each to
    its own content would silently re-scale them relative to each other.
    """
    boxes = []
    size = None
    for path in view_paths:
        img = Image.open(path).convert("RGBA")
        if size is None:
            size = img.size
        elif img.size != size:
            raise ValueError("Views differ in size; re-render with one --size.")
        bbox = img.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError("View is fully transparent: " + path)
        boxes.append(bbox)

    # The window is re-centred on the canvas midline, so the panel's centre stays
    # the character's own axis -- the split and the contact sheet both rely on
    # that midline being meaningful.
    return crop_from_boxes(boxes, size, margin, align)


def crop_from_boxes(boxes: Sequence[tuple], size: tuple, margin: float, align: int = 8) -> tuple:
    """Turn a set of per-view content boxes into one shared crop window."""
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)

    w, h = size
    mx = int(round((right - left) * margin))
    my = int(round((bottom - top) * margin))
    left, top = max(0, left - mx), max(0, top - my)
    right, bottom = min(w, right + mx), min(h, bottom + my)

    cx = w / 2.0
    half = max(cx - left, right - cx)
    left, right = max(0, int(round(cx - half))), min(w, int(round(cx + half)))

    if align > 1:
        width = ((right - left) // align) * align
        height = ((bottom - top) // align) * align
        right, bottom = left + max(align, width), top + max(align, height)
    return (left, top, right, bottom)


def common_crop_on_bg(
    images: Sequence["Image.Image"],
    bg: Sequence[int],
    margin: float = 0.08,
    align: int = 8,
    tol: int = 18,
) -> tuple:
    """`common_crop` for opaque renders, keyed on the flat background colour.

    ComfyUI hands IMAGE tensors around without an alpha channel, so the alpha
    route is unavailable inside a graph. As long as the views were rendered onto
    one flat background -- which the graph sets on Render Mesh -- the subject's
    extent is just as recoverable from colour.
    """
    sizes = {im.size for im in images}
    if len(sizes) != 1:
        raise ValueError("Views differ in size " + repr(sizes) + "; render them together.")
    boxes = [content_box(im, bg, tol=tol) for im in images]
    return crop_from_boxes(boxes, images[0].size, margin, align)


def compose(
    view_paths: "dict[str, str]",
    order: Sequence[str],
    out_path: str,
    gutter: int = 20,
    pad: int = 20,
    background: Sequence[int] = DEFAULT_BG,
    crop: "tuple | None" = None,
    target_ratio=DEFAULT_RATIOS,
) -> TriptychLayout:
    """Lay the named views out left to right on one opaque canvas.

    `target_ratio` is a menu of accepted aspect ratios; the canvas is padded out
    to whichever is closest. Partner video APIs take a fixed menu, and handing one
    an off-menu canvas gets it centre-cropped or letterboxed -- a centre-cropped
    triptych loses its outer panels outright. Which ratio is closest depends on
    the subject: three panels of a squat character land near 21:9, three of a tall
    thin one near 16:9. See `fit_ratio`.
    """
    missing = [v for v in order if v not in view_paths]
    if missing:
        raise ValueError("Missing views for triptych: " + ", ".join(missing))

    images = []
    for v in order:
        im = Image.open(view_paths[v]).convert("RGBA")
        if crop is not None:
            im = im.crop(crop)
        images.append(flatten(im, background))
    sizes = {im.size for im in images}
    if len(sizes) != 1:
        raise ValueError(
            "Views differ in size (" + repr(sizes) + "). They must share one "
            "ortho framing; re-render them with the same --size."
        )
    pw, ph = images[0].size

    content_w = pw * len(images) + gutter * (len(images) - 1)
    width = pad * 2 + content_w
    height = pad * 2 + ph
    ratios = parse_ratio(target_ratio) if not isinstance(target_ratio, list) else target_ratio
    if ratios:
        width, height = fit_ratio(width, height, ratios)

    # Centre the strip in whatever canvas the ratio asked for, on both axes.
    pad_x = max(0, (width - content_w) // 2)
    pad_y = max(0, (height - ph) // 2)

    canvas = Image.new("RGB", (width, height), tuple(background))

    panels = []
    x = pad_x
    for view, im in zip(order, images):
        canvas.paste(im, (x, pad_y))
        panels.append(PanelBox(view=view, x=x, y=pad_y, w=pw, h=ph))
        x += pw + gutter

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    canvas.save(out_path)

    layout = TriptychLayout(
        width=width, height=height, gutter=gutter, pad=pad_y,
        background=tuple(background), panels=panels,
    )
    with open(_layout_path(out_path), "w", encoding="utf-8") as fh:
        json.dump(layout.to_json(), fh, indent=2)
    return layout


def fit_ratio(width: int, height: int, target) -> tuple:
    """Grow a canvas to an accepted aspect ratio, padding whichever axis is short.

    Padding only ever adds background, so the panels keep their pixel size and the
    shared ortho scale survives.

    `target` may be one ratio or several. Several is the useful case: partner
    video APIs take a fixed menu of aspect ratios, and which one is cheapest to
    reach depends entirely on the subject. Three panels of a squat character land
    near 2.5:1 and want 21:9; three panels of a tall thin one land near 1.7:1 and
    want 16:9. Padding the tall one out to 21:9 would add a third of a canvas of
    empty grey and shrink the character in the returned clip.

    The original version of this only padded vertically, so a subject narrower
    than the target silently missed it -- the thin-robot test case came out at
    1.68 with a 21:9 target and no complaint. Hence padding both ways.
    """
    targets = target if isinstance(target, (list, tuple)) else [target]
    targets = [float(t) for t in targets if t]
    if not targets:
        return width, height

    natural = width / height
    best = min(targets, key=lambda t: abs(t - natural))

    if natural > best:      # too wide: grow the height
        height = int(round(width / best))
    elif natural < best:    # too tall: grow the width
        width = int(round(height * best))
    return width, height


def parse_ratio(spec) -> "list | None":
    """Read '21:9', '1.78', '21:9,16:9,1:1' or 'none' into a list of floats."""
    if spec is None or isinstance(spec, (int, float)):
        return [float(spec)] if spec else None
    text = str(spec).strip().lower()
    if not text or text in ("none", "free", "0"):
        return None

    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out.append(float(a) / float(b))
        else:
            out.append(float(part))
    return out or None


def _layout_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ".layout.json"


def content_box(img: Image.Image, bg: Sequence[int], tol: int = 18) -> tuple:
    """Bounding box of everything that is not the flat background colour."""
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    masks = [
        chan.point(lambda v, base=base: 255 if abs(v - base) > tol else 0)
        for chan, base in zip((r, g, b), bg)
    ]
    mask = ImageChops.lighter(ImageChops.lighter(masks[0], masks[1]), masks[2])
    box = mask.getbbox()
    if box is None:
        raise ValueError("Frame is entirely background; cannot register it.")
    return box


def register(reference_path: str, frame_path: str, bg: Sequence[int]) -> tuple:
    """Solve the scale+offset that maps a returned frame back onto the layout.

    Video models do not hand back what you gave them. This run was fed a
    1904x816 canvas and returned 1536x672 -- not a uniform rescale, so resizing
    the frame to the canvas size shears it by about 2% and the split eats the
    character's feet.

    The first frame of an image-to-video clip is very nearly the input image, so
    both can be reduced to the bounding box of their non-background content and
    the mapping read straight off the two boxes. Flat backgrounds make that box
    exact, which is the reason the triptych is composited onto one.

    Returns (scale_x, scale_y, offset_x, offset_y) mapping frame -> reference.
    """
    ref_box = content_box(Image.open(reference_path), bg)
    frm_box = content_box(Image.open(frame_path), bg)

    ref_w, ref_h = ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]
    frm_w, frm_h = frm_box[2] - frm_box[0], frm_box[3] - frm_box[1]
    if frm_w <= 0 or frm_h <= 0:
        raise ValueError("Degenerate content box in " + frame_path)

    sx, sy = ref_w / frm_w, ref_h / frm_h
    return (sx, sy, ref_box[0] - frm_box[0] * sx, ref_box[1] - frm_box[1] * sy)


def apply_registration(img: Image.Image, transform: tuple, layout: TriptychLayout) -> Image.Image:
    """Resample a frame into the layout's coordinate space."""
    sx, sy, ox, oy = transform
    scaled = img.resize(
        (max(1, int(round(img.width * sx))), max(1, int(round(img.height * sy)))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", (layout.width, layout.height), tuple(layout.background))
    canvas.paste(scaled, (int(round(ox)), int(round(oy))))
    return canvas


def split(
    frame_path: str,
    layout: TriptychLayout,
    out_dir: str,
    stem: str,
    transform: "tuple | None" = None,
) -> "dict[str, str]":
    """Cut one animated frame back into per-view images.

    Pass `transform` from `register()` whenever the clip came back at a different
    size than the canvas; without it the frame is stretched to fit, which is only
    correct when the model preserved the aspect ratio exactly.
    """
    img = Image.open(frame_path).convert("RGB")
    if transform is not None:
        img = apply_registration(img, transform, layout)
    elif img.size != (layout.width, layout.height):
        img = img.resize((layout.width, layout.height), Image.LANCZOS)

    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for panel in layout.panels:
        crop = img.crop((panel.x, panel.y, panel.x + panel.w, panel.y + panel.h))
        path = os.path.join(out_dir, stem + "_" + panel.view + ".png")
        crop.save(path)
        written[panel.view] = path
    return written


def contact_sheet(
    rows: Sequence[dict],
    order: Sequence[str],
    out_path: str,
    thumb: int = 256,
    background: Sequence[int] = (255, 255, 255),
    label_h: int = 26,
    gutter: int = 8,
) -> str:
    """Render the final N-keyframes x M-views sheet, with labels.

    Labels go here and nowhere earlier: text inside an image handed to a video or
    edit model gets garbled and then re-generated as nonsense.
    """
    font = _load_font(15)
    small = _load_font(13)

    header = label_h
    left = 64
    cell = thumb + gutter
    width = left + cell * len(order) + gutter
    height = header + cell * len(rows) + gutter

    canvas = Image.new("RGB", (width, height), tuple(background))
    draw = ImageDraw.Draw(canvas)

    for j, view in enumerate(order):
        x = left + j * cell + gutter
        draw.text((x, 6), view.upper(), fill=(20, 20, 20), font=font)

    for i, row in enumerate(rows):
        y = header + i * cell + gutter
        draw.text((10, y + thumb // 2 - 8), _row_label(row, i), fill=(20, 20, 20), font=small)
        for j, view in enumerate(order):
            path = row.get("views", {}).get(view)
            if not path or not os.path.exists(path):
                continue
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb, thumb), Image.LANCZOS)
            x = left + j * cell + gutter
            canvas.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    canvas.save(out_path)
    return out_path


def _row_label(row: dict, index: int) -> str:
    idx = row.get("index", index)
    t = row.get("t")
    if t is None:
        return "k" + str(idx).zfill(2)
    return "k" + str(idx).zfill(2) + "\n" + format(float(t), ".2f") + "s"


def _load_font(size: int):
    for name in ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_layout(image_path: str) -> TriptychLayout:
    with open(_layout_path(image_path), "r", encoding="utf-8") as fh:
        return TriptychLayout.from_json(json.load(fh))
