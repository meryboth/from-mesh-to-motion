"""Regenerate the figures in docs/report/assets from the run outputs.

Every figure in the write-up is built here rather than exported by hand, so a
re-run cannot leave the prose describing one picture and the page showing
another.

  python scripts/build_docs_assets.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pipeline import sheet as sheet_mod  # noqa: E402

OUT = os.path.join(REPO, "docs", "report", "assets")
BG = (245, 245, 247)
INK = (24, 24, 27)
MUTED = (110, 110, 118)


def font(size, bold=False):
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names + ["DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save(img, name, quality=88):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    if name.endswith(".jpg"):
        img.convert("RGB").save(path, quality=quality, optimize=True)
    else:
        img.save(path, optimize=True)
    print("  " + name + "  " + str(img.size))
    return path


def label_strip(images, captions, width=1600, pad=18, title_h=30):
    """A labelled row of images, scaled to a shared height."""
    n = len(images)
    cell_w = (width - pad * (n + 1)) // n
    scaled = []
    for im in images:
        h = int(im.height * cell_w / im.width)
        scaled.append(im.resize((cell_w, h), Image.LANCZOS))
    cell_h = max(im.height for im in scaled)

    canvas = Image.new("RGB", (width, pad * 2 + title_h + cell_h), BG)
    d = ImageDraw.Draw(canvas)
    for i, (im, cap) in enumerate(zip(scaled, captions)):
        x = pad + i * (cell_w + pad)
        d.text((x, pad), cap, fill=INK, font=font(17, bold=True))
        canvas.paste(im, (x, pad + title_h))
    return canvas


def fig_turnaround():
    """The four Blender anchors: one scale, one pivot, four yaws."""
    d = os.path.join(REPO, "out", "02_anchors")
    views = ["front", "side", "back", "left"]
    ims = []
    for v in views:
        im = Image.open(os.path.join(d, "beauty_" + v + ".png")).convert("RGBA")
        flat = Image.new("RGBA", im.size, BG + (255,))
        flat.alpha_composite(im)
        ims.append(flat.convert("RGB"))
    caps = ["front - yaw 0", "side - yaw 90", "back - yaw 180", "left - yaw 270"]
    save(label_strip(ims, caps, width=1600), "turnaround.jpg")


def fig_triptych():
    """The canvas that is actually submitted to the video model."""
    im = Image.open(os.path.join(REPO, "out", "03_triptych", "triptych_rest.png"))
    w = 1600
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    save(im, "triptych-strip.jpg")


def _sheet_rows(run_dir, indices, views=("front", "side", "back"), thumb=220):
    """A few keyframe rows lifted out of a run, laid out like the real sheet."""
    pad, label_w, head = 10, 70, 28
    cell = thumb + pad
    width = label_w + cell * len(views) + pad
    height = head + cell * len(indices) + pad
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)

    for j, v in enumerate(views):
        d.text((label_w + j * cell + pad, 6), v.upper(), fill=MUTED, font=font(15, bold=True))

    for i, idx in enumerate(indices):
        y = head + i * cell + pad
        d.text((10, y + thumb // 2 - 8), "k" + str(idx).zfill(2), fill=INK, font=font(14))
        for j, v in enumerate(views):
            p = os.path.join(run_dir, "views", "k" + str(idx).zfill(2) + "_" + v + ".png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb), Image.LANCZOS)
            x = label_w + j * cell + pad
            canvas.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
    return canvas


def fig_sheet_hero():
    run = os.path.join(REPO, "out", "05_keyframes")
    save(_sheet_rows(run, [0, 3, 5, 7, 10, 13], thumb=240), "sheet-hero.jpg")


def fig_side_drift():
    """The failure mode: the side panel degrading while front and back hold."""
    run = os.path.join(REPO, "out", "06_keyframes_jump")
    if not os.path.isdir(run):
        print("  (skipped side-drift: no out/06_keyframes_jump)")
        return
    save(_sheet_rows(run, [0, 4, 8, 9, 10, 11], thumb=240), "side-drift.jpg")


def _reshape_cases(src, bg):
    """The three things a video model does to a canvas it was handed."""
    def letterbox(im, W, H):
        k = min(W / im.width, H / im.height)
        r = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        c = Image.new("RGB", (W, H), tuple(bg))
        c.paste(r, ((W - r.width) // 2, (H - r.height) // 2))
        return c

    return {
        "rescaled": lambda im: im.resize((1536, 672), Image.LANCZOS),
        "letterboxed": lambda im: letterbox(im, 1280, 720),
        "cropped": lambda im: im.crop(
            (int(im.width * .03), int(im.height * .03),
             int(im.width * .97), int(im.height * .97))
        ).resize((1536, 672), Image.LANCZOS),
    }


def fig_registration():
    """Registration is insurance, and the figure has to show that honestly.

    For the clip this repo actually ran, a naive stretch is fine -- the model
    rescaled the canvas and cropped nothing. The figure therefore shows the case
    registration exists for (a letterboxed return), and the numbers for all three
    cases are written to registration.json so the write-up quotes measurements
    rather than adjectives.
    """
    import json

    import numpy as np

    trip = os.path.join(REPO, "out", "03_triptych", "triptych_rest.png")
    if not os.path.exists(trip):
        print("  (skipped registration: no triptych)")
        return
    layout = sheet_mod.load_layout(trip)
    src = Image.open(trip).convert("RGB")
    bg = tuple(layout.background)
    tmp = os.path.join(REPO, "out", "_docs_tmp")
    os.makedirs(tmp, exist_ok=True)

    def mae(a, b):
        return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())

    results, figure_panels = {}, None
    for name, reshape in _reshape_cases(src, bg).items():
        path = os.path.join(tmp, "case_" + name + ".png")
        reshape(src).save(path)
        naive = sheet_mod.split(path, layout, tmp, "naive_" + name)
        tf = sheet_mod.register(trip, path, bg)
        fixed = sheet_mod.split(path, layout, tmp, "fixed_" + name, transform=tf)

        n_scores, f_scores = [], []
        for panel in layout.panels:
            ref = src.crop((panel.x, panel.y, panel.x + panel.w, panel.y + panel.h))
            n_scores.append(mae(ref, Image.open(naive[panel.view])))
            f_scores.append(mae(ref, Image.open(fixed[panel.view])))
        results[name] = {
            "naive_mae": round(sum(n_scores) / len(n_scores), 2),
            "registered_mae": round(sum(f_scores) / len(f_scores), 2),
        }
        if name == "letterboxed":
            panel = layout.panels[0]
            figure_panels = [
                src.crop((panel.x, panel.y, panel.x + panel.w, panel.y + panel.h)),
                Image.open(naive[panel.view]),
                Image.open(fixed[panel.view]),
            ]

    with open(os.path.join(OUT, "registration.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("  registration.json  " + json.dumps(results))

    if figure_panels:
        lb = results["letterboxed"]
        caps = [
            "the anchor, as submitted",
            "letterboxed clip, split naively - MAE " + format(lb["naive_mae"], ".2f"),
            "same clip, registered first - MAE " + format(lb["registered_mae"], ".2f"),
        ]
        save(label_strip(figure_panels, caps, width=1500), "registration.jpg")


def _ffmpeg_gif(src, dest, width=720, fps=10, start=0.0, duration=None, colors=64):
    """Clip to GIF via a generated palette.

    ffmpeg's default GIF quantiser posterises flat backgrounds into visible
    bands, which on a page whose whole point is a flat grey canvas looks like a
    pipeline defect rather than an encoder one. palettegen/paletteuse costs one
    extra pass and removes it.
    """
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        print("  (skipped " + os.path.basename(dest) + ": no ffmpeg)")
        return False

    chain = "fps=" + str(fps) + ",scale=" + str(width) + ":-1:flags=lanczos"
    cmd = [exe, "-y", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i", src,
        "-filter_complex",
        chain + ",split[a][b];[a]palettegen=max_colors=" + str(colors)
              + ":stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
        "-loop", "0", dest,
    ]
    subprocess.run(cmd, check=True)
    print("  " + os.path.basename(dest) + "  "
          + format(os.path.getsize(dest) / 1e6, ".1f") + " MB")
    return True


def gif_triptych_motion():
    """The clip itself: three panels moving in step inside one frame."""
    src = os.path.join(REPO, "out", "04_video", "motion_test.mp4")
    if not os.path.exists(src):
        print("  (skipped triptych-motion.gif: no clip)")
        return
    _ffmpeg_gif(src, os.path.join(OUT, "triptych-motion.gif"))


def gif_side_drift():
    """The failure mode, in motion -- much clearer moving than as stills."""
    src = os.path.join(REPO, "out", "04_video", "motion_jump.mp4")
    if not os.path.exists(src):
        print("  (skipped side-drift.gif: no jump clip)")
        return
    _ffmpeg_gif(src, os.path.join(OUT, "side-drift.gif"))


def gif_sheet_step():
    """The deliverable, stepped one keyframe at a time.

    Built from the split panels rather than the clip, so what the page animates
    is exactly what lands in the sheet -- including any registration error.
    """
    run = os.path.join(REPO, "out", "05_keyframes", "views")
    if not os.path.isdir(run):
        print("  (skipped sheet-step.gif: no keyframes)")
        return
    views = ("front", "side", "back")
    pad, head, thumb = 12, 26, 260

    frames = []
    for i in range(64):
        paths = [os.path.join(run, "k" + str(i).zfill(2) + "_" + v + ".png") for v in views]
        if not all(os.path.exists(p) for p in paths):
            break
        width = pad + len(views) * (thumb + pad)
        canvas = Image.new("RGB", (width, head + thumb + pad), BG)
        d = ImageDraw.Draw(canvas)
        for j, (v, p) in enumerate(zip(views, paths)):
            x = pad + j * (thumb + pad)
            d.text((x, 5), v.upper(), fill=MUTED, font=font(14, bold=True))
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb), Image.LANCZOS)
            canvas.paste(im, (x + (thumb - im.width) // 2, head + (thumb - im.height) // 2))
        d.text((width - 52, 5), "k" + str(i).zfill(2), fill=INK, font=font(14, bold=True))
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))

    if not frames:
        print("  (skipped sheet-step.gif: no keyframe views)")
        return
    dest = os.path.join(OUT, "sheet-step.gif")
    frames[0].save(dest, save_all=True, append_images=frames[1:],
                   duration=380, loop=0, optimize=True, disposal=2)
    print("  sheet-step.gif  " + str(len(frames)) + " frames  "
          + format(os.path.getsize(dest) / 1e6, ".1f") + " MB")


def fig_generality():
    """Three characters, three rest-pose triptychs, one figure.

    The pipeline was built around one character, which proves nothing about it.
    These are the two it was tested against afterwards, chosen to break different
    things: a quadruped has no obvious front, and a wire-thin robot with loose
    cloth is the hardest thing to hold together in a profile view.
    """
    specs = [
        ("out/03_triptych/triptych_rest.png", "mascot - 21:9 - PASS"),
        ("out/B_quadruped/03_triptych/triptych_rest.png", "quadruped - 21:9 - PASS"),
        ("out/C_robot/03_triptych/triptych_rest.png", "thin robot - 16:9 - FAIL"),
    ]
    rows = []
    for path, cap in specs:
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            print("  (skipped generality: missing " + path + ")")
            return
        rows.append((Image.open(full).convert("RGB"), cap))

    width, pad = 1500, 16
    scaled = [(im.resize((width - 2 * pad,
                          int(im.height * (width - 2 * pad) / im.width)), Image.LANCZOS), cap)
              for im, cap in rows]
    height = pad + sum(im.height + 30 + pad for im, _ in scaled)
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    y = pad
    for im, cap in scaled:
        d.text((pad, y), cap, fill=INK, font=font(17, bold=True))
        canvas.paste(im, (pad, y + 26))
        y += im.height + 30 + pad
    save(canvas, "generality.jpg")


def gif_generality():
    for key, src in [("B_quadruped", "quadruped"), ("C_robot", "thin-robot")]:
        path = os.path.join(REPO, "out", key, "04_video", "motion.mp4")
        if os.path.exists(path):
            _ffmpeg_gif(path, os.path.join(OUT, src + "-motion.gif"), width=720, fps=10)


def main():
    print("building docs assets ->", os.path.relpath(OUT, REPO))
    for fn in (fig_turnaround, fig_triptych, fig_sheet_hero,
               fig_side_drift, fig_registration,
               gif_triptych_motion, gif_side_drift, gif_sheet_step,
               fig_generality, gif_generality):
        try:
            fn()
        except FileNotFoundError as exc:
            print("  (skipped " + fn.__name__ + ": " + str(exc) + ")")


if __name__ == "__main__":
    main()
