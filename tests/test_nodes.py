"""Round-trip test for the node pack, run the way ComfyUI loads it.

The repository directory has hyphens in its name, so it cannot be imported as a
plain package. ComfyUI loads custom nodes from a file path via importlib, and so
does this test -- otherwise the test would pass against an import path that
never happens in practice.

  python tests/test_nodes.py

Uses the checked-in example renders when they exist, and falls back to three
synthetic views so the test runs on a clean clone.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG = (228, 228, 230)


def load_pack():
    """Import the repo the way ComfyUI does: by path, under a legal module name."""
    name = "from_mesh_to_motion_under_test"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "__init__.py"),
        submodule_search_locations=[REPO],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_views(size=(384, 448)):
    """Three fake ortho views: same canvas, same scale, different silhouettes."""
    views = []
    for i, (w_body, has_pack) in enumerate([(120, True), (70, True), (120, False)]):
        im = Image.new("RGB", size, BG)
        d = ImageDraw.Draw(im)
        cx, cy = size[0] // 2, size[1] // 2
        if has_pack and i == 1:
            d.rounded_rectangle([cx - 78, cy - 30, cx - 36, cy + 60], 8, fill=(232, 150, 70))
        d.ellipse([cx - w_body // 2, cy - 90, cx + w_body // 2, cy + 20], fill=(70, 100, 105))
        d.rounded_rectangle([cx - w_body // 2, cy, cx + w_body // 2, cy + 130],
                            20, fill=(70, 100, 105))
        views.append(im)
    return views


def to_tensor(images, torch):
    arrs = [np.asarray(im.convert("RGB")).astype(np.float32) / 255.0 for im in images]
    return torch.from_numpy(np.stack(arrs))


def main() -> int:
    import torch

    pack = load_pack()
    N = pack.NODE_CLASS_MAPPINGS
    expected = {
        "FMM_MotionPrompt", "FMM_TriptychCompose", "FMM_TriptychRegister",
        "FMM_SampleKeyframes", "FMM_TriptychSplit", "FMM_KeyframeSheet",
        "FMM_SaveKeyframeManifest", "FMM_IdentityQA",
    }
    missing = expected - set(N)
    assert not missing, "Missing nodes: " + ", ".join(sorted(missing))
    print("registered " + str(len(N)) + " nodes")

    anchors = os.path.join(REPO, "out", "02_anchors")
    names = ["front", "side", "back"]
    files = [os.path.join(anchors, "beauty_" + v + ".png") for v in names]
    if all(os.path.exists(f) for f in files):
        views = []
        for f in files:
            im = Image.open(f).convert("RGBA")
            flat = Image.new("RGBA", im.size, BG + (255,))
            flat.alpha_composite(im)
            views.append(flat.convert("RGB"))
        print("using example renders from out/02_anchors")
    else:
        views = synthetic_views()
        print("using synthetic views")

    a, b, c = [to_tensor([v], torch) for v in views]

    triptych, layout = N["FMM_TriptychCompose"]().run(
        a, b, c, "front,side,back", 20, 20, "#E4E4E6", "21:9", 0.08)
    lay = json.loads(layout)
    ratio = lay["width"] / lay["height"]
    print("triptych " + str(lay["width"]) + "x" + str(lay["height"])
          + "  ratio " + format(ratio, ".4f"))
    assert abs(ratio - 21 / 9) < 0.01, "target ratio not honoured: " + format(ratio, ".4f")
    assert len(lay["panels"]) == 3

    # The motion prompt must carry the action AND the scaffolding that locks the
    # panels down. A prompt that lost either is the failure worth testing for:
    # losing the action gives the wrong animation, losing the scaffolding gives
    # panels that disagree, and neither raises.
    action = "raises its right arm and waves twice, then lowers it"
    text, summary = N["FMM_MotionPrompt"]().run(
        action, "clay toy character", False, layout=layout)
    assert action in text, "the action was dropped from the prompt"
    assert text.strip().startswith(action),         "the action must lead -- constraints first stopped the motion happening"
    for required in ("no pan", "no zoom", "never change their viewing angle",
                     "perfect unison", "left panel front view",
                     "centre panel side profile", "right panel back view"):
        assert required in text, "prompt lost its scaffolding: " + required
    assert "light grey" in text, "background description does not match the canvas"
    assert summary.startswith("Raises its right arm")
    looped, _ = N["FMM_MotionPrompt"]().run(action, "toy", True, layout=layout)
    assert "loops cleanly" in looped
    print("motion prompt " + str(len(text.split())) + " words, action first")

    # Fake what a video model does to the canvas. A plain rescale is the benign
    # case; a centre-crop is the one registration exists for, so the test uses
    # that -- otherwise it would pass just as well with the node removed.
    tri_pil = Image.fromarray((triptych[0].numpy() * 255).astype(np.uint8), "RGB")
    reshaped = tri_pil.crop((
        int(tri_pil.width * 0.03), int(tri_pil.height * 0.03),
        int(tri_pil.width * 0.97), int(tri_pil.height * 0.97),
    )).resize((1536, 672), Image.LANCZOS)
    frames_t = to_tensor([reshaped] * 12, torch)

    layout2, report = N["FMM_TriptychRegister"]().run(
        triptych, frames_t[0:1], layout)
    print("register: " + report.splitlines()[0])
    tf = json.loads(layout2)["transform"]
    assert tf is not None and len(tf) == 4

    keyframes, timings = N["FMM_SampleKeyframes"]().run(frames_t, 6, 24.0, False)
    times = json.loads(timings)
    assert keyframes.shape[0] == 6, keyframes.shape
    assert times[0]["frame"] == 0 and times[-1]["frame"] == 11, times
    print("sampled " + str(len(times)) + " keyframes, last t=" + str(times[-1]["t"]) + "s")

    va, vb, vc, out_names = N["FMM_TriptychSplit"]().run(keyframes, layout2)
    assert out_names == "front,side,back", out_names
    for t in (va, vb, vc):
        assert t.shape[0] == 6
        assert t.shape[1] == lay["panels"][0]["h"], t.shape
        assert t.shape[2] == lay["panels"][0]["w"], t.shape
    print("split -> 3 x " + str(tuple(va.shape)))

    # The split panels must land back on the source panels, or every pose read
    # off the sheet is off by the registration error. The same split without the
    # registered layout is measured alongside, so the assertion is that the node
    # helps -- not merely that the numbers are small.
    naive_a, naive_b, naive_c, _ = N["FMM_TriptychSplit"]().run(keyframes, layout)
    src = np.asarray(tri_pil).astype(float)
    for panel, got, naive in zip(lay["panels"], (va, vb, vc), (naive_a, naive_b, naive_c)):
        ref = src[panel["y"]:panel["y"] + panel["h"],
                  panel["x"]:panel["x"] + panel["w"]]
        mae = np.abs(ref - got[0].numpy() * 255).mean()
        mae_naive = np.abs(ref - naive[0].numpy() * 255).mean()
        print("  " + panel["view"].ljust(6) + " MAE registered " + format(mae, ".2f")
              + "   naive " + format(mae_naive, ".2f"))
        assert mae < 6.0, panel["view"] + " panel misaligned (MAE " + format(mae, ".2f") + ")"
        assert mae < mae_naive, \
            panel["view"] + " registration did not beat the naive split"

    out_dir = os.path.join(REPO, "out", "_test")
    sheet_img, = N["FMM_KeyframeSheet"]().run(va, vb, vc, out_names, timings, 160)
    assert sheet_img.shape[0] == 1
    print("sheet " + str(sheet_img.shape[2]) + "x" + str(sheet_img.shape[1]))

    man_path, handoff = N["FMM_SaveKeyframeManifest"]().run(
        va, vb, vc, out_names, timings, "TestMascot",
        "raises both arms overhead and jumps", 24.0, False, out_dir, sheet=sheet_img)
    doc = json.load(open(man_path, encoding="utf-8"))
    assert doc["motion"]["keyframe_count"] == 6
    assert len(doc["keyframes"]) == 6
    assert doc["views"] == ["front", "side", "back"]
    assert all(not os.path.isabs(p)
               for kf in doc["keyframes"] for p in kf["views"].values()), \
        "manifest paths must be relative so the folder stays portable"
    assert "6 keyframes" in handoff and "front, side, back" in handoff
    print("manifest -> " + os.path.relpath(man_path, REPO))

    # Real times, not index/fps: k01 of 6 picks over 12 frames is frame 2.
    assert doc["keyframes"][1]["frame"] == times[1]["frame"]
    assert abs(doc["keyframes"][1]["t"] - times[1]["t"]) < 1e-6

    # Unwired timings must refuse, not guess -- the guess was eight times off.
    try:
        N["FMM_SaveKeyframeManifest"]().run(
            va, vb, vc, out_names, "", "x", "y", 24.0, False, out_dir)
    except ValueError as exc:
        assert "timings" in str(exc)
        print("manifest refuses to guess timings")
    else:
        raise AssertionError("manifest accepted missing timings")

    qa = N["FMM_IdentityQA"]().run(va, vb, vc, layout2, 0.03)
    report, passed = qa["result"]
    assert report.startswith(("PASS", "FAIL")) and isinstance(passed, bool)
    print("identity QA: " + report.splitlines()[0])

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
