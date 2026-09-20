"""Command line for the local half of the pipeline.

The cloud half (video, identity repair) runs as ComfyUI graphs in workflows/.
Everything here is deterministic and free: geometry in, geometry out.

  python -m pipeline anchors   --mesh mascot.glb --out out/02_anchors
  python -m pipeline triptych  --anchors out/02_anchors --out out/03_triptych
  python -m pipeline keyframes --video motion.mp4 --triptych out/03_triptych/triptych_rest.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from . import manifest as manifest_mod
from . import sheet as sheet_mod
from . import video as video_mod

DEFAULT_VIEWS = ["front", "side", "back"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_blender(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BLENDER")
    if env:
        return env
    found = shutil.which("blender")
    if found:
        return found
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise SystemExit(
        "Blender not found. Pass --blender, or set the BLENDER environment variable."
    )


def load_job(path: str | None) -> dict:
    if not path:
        path = os.path.join(HERE, "config", "job.json")
        if not os.path.exists(path):
            path = os.path.join(HERE, "config", "job.example.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- anchors


def cmd_anchors(args) -> None:
    blender = find_blender(args.blender)
    script = os.path.join(HERE, "blender", "render_ortho.py")
    cmd = [
        blender, "--background", "--python", script, "--",
        "--mesh", os.path.abspath(args.mesh),
        "--out", os.path.abspath(args.out),
        "--size", str(args.size),
        "--views", args.views,
        "--passes", args.passes,
        "--yaw-offset", str(args.yaw_offset),
        "--margin", str(args.margin),
    ]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit("Blender exited with code " + str(proc.returncode))
    print("anchors -> " + os.path.abspath(args.out))


# --------------------------------------------------------------- triptych


def cmd_triptych(args) -> None:
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    paths = {}
    for v in views:
        p = os.path.join(args.anchors, args.pass_name + "_" + v + ".png")
        if not os.path.exists(p):
            raise SystemExit("Missing anchor render: " + p)
        paths[v] = p

    crop = sheet_mod.common_crop(paths.values(), margin=args.crop_margin)
    out_path = os.path.join(args.out, "triptych_rest.png")
    layout = sheet_mod.compose(
        paths, views, out_path,
        gutter=args.gutter, pad=args.pad,
        crop=crop, target_ratio=_ratio(args.ratio),
    )
    print("triptych -> " + out_path)
    print("  canvas " + str(layout.width) + "x" + str(layout.height)
          + "  ratio " + format(layout.width / layout.height, ".4f"))
    print("  layout -> " + os.path.splitext(out_path)[0] + ".layout.json")
    print("\nNext: upload this image and run workflows/03_triptych_to_motion.json")


def _ratio(spec: str | None):
    if not spec or spec.lower() in ("none", "free"):
        return None
    if ":" in spec:
        a, b = spec.split(":")
        return float(a) / float(b)
    return float(spec)


# -------------------------------------------------------------- keyframes


def cmd_keyframes(args) -> None:
    job = load_job(args.job)
    layout = sheet_mod.load_layout(args.triptych)
    views = [p.view for p in layout.panels]

    work = os.path.join(args.out, "_frames")
    frames = video_mod.extract_all(args.video, work)
    meta = video_mod.ffprobe_meta(args.video)
    fps = meta.get("fps") or float(job.get("motion", {}).get("fps", 24)) or 24.0
    print("decoded " + str(len(frames)) + " frames"
          + ("  @ " + format(fps, ".3f") + " fps" if fps else ""))

    # Frame 0 is (nearly) the image we submitted, so it is what tells us how the
    # model resized or cropped the canvas on its way back.
    transform = None
    if not args.no_register:
        transform = sheet_mod.register(args.triptych, frames[0], layout.background)
        sx, sy, ox, oy = transform
        print("registration: scale " + format(sx, ".4f") + " x " + format(sy, ".4f")
              + "  offset " + format(ox, ".1f") + ", " + format(oy, ".1f"))
        if abs(sx - sy) / max(sx, sy) > 0.02:
            print("  warning: non-uniform scale above 2% -- the model reshaped the "
                  "canvas; check the first split before trusting the sheet.")

    loop = bool(job.get("motion", {}).get("loop", False))
    picks = video_mod.pick_keyframes(frames, args.n, include_last=not loop)

    views_dir = os.path.join(args.out, "views")
    keyframes = []
    for i, frame_idx in enumerate(picks):
        stem = "k" + str(i).zfill(2)
        written = sheet_mod.split(frames[frame_idx], layout, views_dir, stem,
                                  transform=transform)
        keyframes.append({
            "index": i,
            "t": frame_idx / fps if fps else float(i),
            "frame": frame_idx,
            "views": written,
        })
        print("  " + stem + "  frame " + str(frame_idx).rjust(4)
              + "  t=" + format(frame_idx / fps if fps else i, ".2f") + "s")

    sheet_path = os.path.join(args.out, "keyframe_sheet.png")
    sheet_mod.contact_sheet(keyframes, views, sheet_path, thumb=args.thumb)

    doc = manifest_mod.build(
        subject=job.get("subject", {}),
        motion=dict(job.get("motion", {}), fps=fps),
        views=views,
        keyframes=keyframes,
        provenance=dict(job.get("provenance", {}), source_video=os.path.abspath(args.video)),
        sheet_path=sheet_path,
    )
    man_path = manifest_mod.write(doc, os.path.join(args.out, "keyframes.json"))

    handoff = os.path.join(args.out, "handoff.txt")
    with open(handoff, "w", encoding="utf-8") as fh:
        fh.write(manifest_mod.handoff_text(doc))

    if not args.keep_frames:
        shutil.rmtree(work, ignore_errors=True)

    print("\nsheet    -> " + sheet_path)
    print("manifest -> " + man_path)
    print("handoff  -> " + handoff)


# ------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("anchors", help="Render orthographic views of the rest-pose mesh")
    a.add_argument("--mesh", required=True)
    a.add_argument("--out", default="out/02_anchors")
    a.add_argument("--size", type=int, default=768)
    a.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    a.add_argument("--passes", default="beauty")
    a.add_argument("--yaw-offset", type=float, default=0.0,
                   help="Rotate every view, for a mesh whose front is not facing -Y")
    a.add_argument("--margin", type=float, default=0.12)
    a.add_argument("--blender", default=None)
    a.set_defaults(func=cmd_anchors)

    t = sub.add_parser("triptych", help="Compose the anchors into one 21:9 canvas")
    t.add_argument("--anchors", default="out/02_anchors")
    t.add_argument("--out", default="out/03_triptych")
    t.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    t.add_argument("--pass-name", dest="pass_name", default="beauty")
    t.add_argument("--gutter", type=int, default=20)
    t.add_argument("--pad", type=int, default=20)
    t.add_argument("--crop-margin", type=float, default=0.08)
    t.add_argument("--ratio", default="21:9", help='Aspect to pad to, or "none"')
    t.set_defaults(func=cmd_triptych)

    k = sub.add_parser("keyframes", help="Sample the animated triptych into a sheet")
    k.add_argument("--video", required=True)
    k.add_argument("--triptych", required=True,
                   help="The composed triptych PNG; its .layout.json sits beside it")
    k.add_argument("--out", default="out/05_keyframes")
    k.add_argument("-n", type=int, default=16)
    k.add_argument("--thumb", type=int, default=256)
    k.add_argument("--job", default=None)
    k.add_argument("--keep-frames", action="store_true")
    k.add_argument("--no-register", action="store_true",
                   help="Skip first-frame registration and stretch to fit instead")
    k.set_defaults(func=cmd_keyframes)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
