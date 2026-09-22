"""Build the technical reference: docs/report/reference.html.

The case study (docs/report/index.html) explains the method in prose. This page
is the reference: every stage, every node with its full input/output contract,
every CLI command, every data format, the metrics and their thresholds, and the
measured results for each character the pipeline was run on.

Almost nothing here is typed by hand. The node contracts are read from the node
classes, the CLI from its argparse parser, the graph from the workflow file, and
the numbers from the runs' own qa.json, keyframes.json and layout files -- so the
reference cannot describe a pipeline that no longer exists.

  python scripts/build_reference.py            # uses cached run data if clips are absent
  python scripts/build_reference.py --fragment out.html   # also write a <body>-less copy

Registration figures need the source clips (kept out of git); they are computed
once and cached in docs/report/assets/runs.json so a clean clone still builds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from html import escape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from PIL import Image  # noqa: E402

from pipeline import sheet as sheet_mod  # noqa: E402
from pipeline import video as video_mod  # noqa: E402

ASSETS = os.path.join(REPO, "docs", "report", "assets")
OUT = os.path.join(REPO, "docs", "report", "reference.html")
CACHE = os.path.join(ASSETS, "runs.json")
CEILING = 0.03

RUNS = [
    {
        "key": "mascot", "name": "Mascot",
        "body": "Biped clay toy: one eye, antennae, rigid backpack",
        "triptych": "out/03_triptych/triptych_rest.png",
        "video": "out/04_video/motion_test.mp4",
        "run": "out/05_keyframes",
        "gif": "triptych-motion.gif",
    },
    {
        "key": "quadruped", "name": "Quadruped",
        "body": "Four-legged creature: long tail, large ears",
        "triptych": "out/B_quadruped/03_triptych/triptych_rest.png",
        "video": "out/B_quadruped/04_video/motion.mp4",
        "run": "out/B_quadruped/05_keyframes",
        "gif": "quadruped-motion.gif",
    },
    {
        "key": "thin-robot", "name": "Thin robot",
        "body": "Tall, wire-thin limbs, loose scarf",
        "triptych": "out/C_robot/03_triptych/triptych_rest.png",
        "video": "out/C_robot/04_video/motion.mp4",
        "run": "out/C_robot/05_keyframes",
        "gif": "thin-robot-motion.gif",
    },
]

NODE_ORDER = [
    "FMM_MotionPrompt", "FMM_TriptychCompose", "FMM_TriptychRegister",
    "FMM_SampleKeyframes", "FMM_TriptychSplit", "FMM_KeyframeSheet",
    "FMM_SaveKeyframeManifest", "FMM_IdentityQA",
]

# What each node is accountable for, and the one decision that defines it.
RESPONSIBILITY = {
    "FMM_MotionPrompt": (
        "Owns the video prompt. The user writes only the action; the node adds the "
        "scaffolding that locks the panels, naming each panel by position and "
        "describing the background read from the layout.",
        "The action is placed first, ahead of the layout constraints."),
    "FMM_TriptychCompose": (
        "Owns the canvas. Crops the three views by one shared window, lays them out "
        "left to right and pads to the nearest aspect ratio a video provider accepts.",
        "Refuses views of different sizes instead of rescaling them."),
    "FMM_TriptychRegister": (
        "Owns the correction for whatever the video model did to the canvas. Matches "
        "frame 0 against the submitted canvas and writes the mapping into the layout.",
        "Reports skew and warns above 2%."),
    "FMM_SampleKeyframes": (
        "Owns timing. Picks N evenly spaced frames and records each one's frame index "
        "and time in seconds.",
        "Keeps frame 0, the rest pose; drops the last frame for loops."),
    "FMM_TriptychSplit": (
        "Owns the cut. Turns each sampled frame back into three per-view images using "
        "the corrected layout.",
        "Applies the registration transform before cutting."),
    "FMM_KeyframeSheet": (
        "Owns the human-readable sheet: N rows by three views, with the keyframe "
        "index and time on every row.",
        "Adds labels only here, never to an image a model will see."),
    "FMM_SaveKeyframeManifest": (
        "Owns the handoff. Writes the per-view PNGs, keyframes.json and handoff.txt "
        "to the output folder.",
        "Requires real timings; will not run without them."),
    "FMM_IdentityQA": (
        "Owns the go/no-go. Scores every panel for identity drift against its view's "
        "first frame and gates the sheet.",
        "Pose-blind measure: colour displacement weighted by the rest pose."),
}

STAGE = {
    "FMM_MotionPrompt": "Prompt", "FMM_TriptychCompose": "Compose",
    "FMM_TriptychRegister": "Register", "FMM_SampleKeyframes": "Sample",
    "FMM_TriptychSplit": "Split", "FMM_KeyframeSheet": "Deliver",
    "FMM_SaveKeyframeManifest": "Deliver", "FMM_IdentityQA": "Check",
}

CORE_NODES = {
    "LoadImage": ("Loads one orthographic render.", "local"),
    "MinimaxHailuo03FirstLastFrameNode": (
        "Animates the canvas from its first frame. The only remote, paid step.", "paid"),
    "GetVideoComponents": ("Decodes the clip into frames and reports its frame rate.", "local"),
    "ImageFromBatch": ("Takes frame 0 out of the decoded batch.", "local"),
    "SaveVideo": ("Writes the clip to output/fmm/.", "local"),
    "SaveImage": ("Writes the submitted canvas and the finished sheet.", "local"),
}


# ------------------------------------------------------------------ data


def load_pack():
    name = "fmm_reference_pack"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def first_frame(video: str) -> str | None:
    exe = shutil.which("ffmpeg")
    if not exe or not os.path.exists(video):
        return None
    path = os.path.join(tempfile.mkdtemp(), "f0.png")
    subprocess.run([exe, "-y", "-v", "error", "-i", video, "-frames:v", "1", path], check=True)
    return path


def collect_runs() -> list:
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            cache = {r["key"]: r for r in json.load(fh)}

    runs = []
    for spec in RUNS:
        run_dir = os.path.join(REPO, spec["run"])
        trip = os.path.join(REPO, spec["triptych"])
        layout = sheet_mod.load_layout(trip)
        qa = json.load(open(os.path.join(run_dir, "qa.json"), encoding="utf-8"))["report"]
        man = json.load(open(os.path.join(run_dir, "keyframes.json"), encoding="utf-8"))

        row = dict(cache.get(spec["key"], {}))
        row.update({
            "key": spec["key"], "name": spec["name"], "body": spec["body"],
            "gif": spec["gif"],
            "canvas": [layout.width, layout.height],
            "panel": [layout.panels[0].w, layout.panels[0].h],
            "ratio": round(layout.width / layout.height, 4),
            "motion": man["motion"]["description"],
            "fps": man["motion"]["fps"],
            "duration_s": man["motion"]["duration_s"],
            "keyframes": [{"i": k["index"], "frame": k["frame"], "t": k["t"]}
                          for k in man["keyframes"]],
            "views": {v: {"mean": d["drift_mean"], "max": d["drift_max"]}
                      for v, d in qa["views"].items()},
            "drift_mean": qa["overall"]["drift_mean"],
            "drift_max": qa["overall"]["drift_max"],
            "passed": qa["gate"]["pass"],
        })

        video = os.path.join(REPO, spec["video"])
        meta = video_mod.ffprobe_meta(video) if os.path.exists(video) else {}
        if meta:
            row["clip"] = {"w": meta["width"], "h": meta["height"],
                           "frames": meta["frames"], "fps": meta["fps"],
                           "seconds": round(meta["duration_s"], 3)}
        f0 = first_frame(video)
        if f0:
            sx, sy, ox, oy = sheet_mod.register(trip, f0, layout.background)
            row["registration"] = {"sx": round(sx, 4), "sy": round(sy, 4),
                                   "skew_pct": round(abs(sx - sy) / max(sx, sy) * 100, 2)}
        runs.append(row)

    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2)
    return runs


def make_figures(runs: list) -> None:
    """Per-character triptych and sheet excerpts, sized for the page."""
    import build_docs_assets as B

    for spec in RUNS:
        trip = Image.open(os.path.join(REPO, spec["triptych"])).convert("RGB")
        w = 1200
        trip.resize((w, int(trip.height * w / trip.width)), Image.LANCZOS).save(
            os.path.join(ASSETS, "triptych-" + spec["key"] + ".jpg"), quality=86, optimize=True)
        rows = B._sheet_rows(os.path.join(REPO, spec["run"]), [0, 4, 7, 10, 13], thumb=170)
        rows.save(os.path.join(ASSETS, "sheet-" + spec["key"] + ".jpg"), quality=86, optimize=True)


def node_contracts(pack) -> list:
    out = []
    for key in NODE_ORDER:
        cls = pack.NODE_CLASS_MAPPINGS[key]
        spec = cls.INPUT_TYPES()
        inputs = []
        for group in ("required", "optional"):
            for name, decl in spec.get(group, {}).items():
                kind = decl[0]
                opts = decl[1] if len(decl) > 1 else {}
                kind = "COMBO" if isinstance(kind, list) else kind
                default = opts.get("default")
                widget = kind in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO") \
                    and not opts.get("forceInput")
                inputs.append({
                    "name": name, "type": kind, "optional": group == "optional",
                    "default": default if widget else None, "widget": widget,
                    "tooltip": opts.get("tooltip", ""),
                })
        outputs = list(zip(cls.RETURN_NAMES, cls.RETURN_TYPES))
        out.append({
            "key": key,
            "display": pack.NODE_DISPLAY_NAME_MAPPINGS.get(key, key),
            "inputs": inputs, "outputs": outputs,
            "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        })
    return out


def cli_reference() -> list:
    from pipeline.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {a.dest: a.help for a in sub._choices_actions}
    out = []
    for name, p in sub.choices.items():
        args = []
        for a in p._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            flag = a.option_strings[0] if a.option_strings else a.dest
            default = a.default if a.default not in (None, False, argparse.SUPPRESS) else ""
            args.append({"flag": flag, "required": a.required, "default": default,
                         "help": a.help or ""})
        out.append({"name": name, "help": helps.get(name, ""), "args": args})
    return out


def package_contents() -> tuple:
    ignore = []
    with open(os.path.join(REPO, ".comfyignore"), encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                ignore.append(line)
    files = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True,
                           text=True, check=True).stdout.split()
    kept = []
    for f in files:
        if any(f.startswith(p) if p.endswith("/") else fnmatch.fnmatch(f, p) for p in ignore):
            continue
        kept.append(f)
    size = sum(os.path.getsize(os.path.join(REPO, f)) for f in kept)
    return kept, size


def version() -> str:
    for line in open(os.path.join(REPO, "pyproject.toml"), encoding="utf-8"):
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return "?"


# ---------------------------------------------------------------- render


def fmt(x, nd=4):
    return format(x, "." + str(nd) + "f") if isinstance(x, (int, float)) else escape(str(x))


def drift_chart(runs: list) -> str:
    """Grouped bars: max drift per view per character, against the ceiling."""
    views = ["front", "side", "back"]
    W, H, L, B, T = 640, 250, 46, 40, 16
    top = 0.06
    plot_h = H - B - T
    group_w = (W - L - 10) / len(runs)
    bar = 30

    def y(v):
        return T + plot_h * (1 - v / top)

    parts = ['<svg class="chart" viewBox="0 0 {} {}" role="img" '
             'aria-label="Maximum identity drift per view for each character">'.format(W, H)]
    for tick in (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        yy = y(tick)
        parts.append('<line class="grid" x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}"/>'.format(L, W - 6, yy, yy))
        parts.append('<text class="tick" x="{}" y="{:.1f}">{:.2f}</text>'.format(L - 8, yy + 4, tick))
    for gi, r in enumerate(runs):
        gx = L + gi * group_w + (group_w - bar * 3 - 12) / 2
        for vi, v in enumerate(views):
            val = r["views"][v]["max"]
            x = gx + vi * (bar + 6)
            parts.append('<rect class="bar v{}" x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" rx="2"/>'.format(
                vi, x, y(val), bar, H - B - y(val)))
            parts.append('<text class="val" x="{:.1f}" y="{:.1f}">{:.3f}</text>'.format(
                x + bar / 2, y(val) - 5, val))
        parts.append('<text class="grp" x="{:.1f}" y="{}">{}</text>'.format(
            L + gi * group_w + group_w / 2, H - 14, escape(r["name"])))
    yc = y(CEILING)
    parts.append('<line class="ceil" x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}"/>'.format(L, W - 6, yc, yc))
    # Label the gate at the left, where every character sits below it.
    parts.append('<text class="ceil-l" x="{}" y="{:.1f}">gate 0.03</text>'.format(L + 6, yc - 6))
    parts.append("</svg>")
    return "".join(parts)


def render(runs, nodes, cli, graph_svg, pkg, ver, registration) -> str:
    files, pkg_size = pkg
    today = dt.date.today().isoformat()
    n_graph_nodes = len(json.load(open(os.path.join(REPO, "workflows",
                                                    "02_triptych_to_keyframes.json")))["nodes"])
    mascot = runs[0]
    layout_example = json.load(open(os.path.join(REPO, "out", "03_triptych",
                                                 "triptych_rest.layout.json")))
    man = json.load(open(os.path.join(REPO, "out", "05_keyframes", "keyframes.json")))
    man_kf = man.pop("keyframes")
    if man.get("provenance", {}).get("source_video"):
        man["provenance"]["source_video"] = os.path.basename(man["provenance"]["source_video"])
    man_excerpt = json.dumps(man, indent=2)[:-2] + ',\n  "keyframes": [\n    ' + \
        ",\n    ".join(json.dumps(k) for k in man_kf[:2]) + ",\n    ...\n  ]\n}"
    handoff = open(os.path.join(REPO, "out", "05_keyframes", "handoff.txt"), encoding="utf-8").read()
    qa_doc = json.load(open(os.path.join(REPO, "out", "05_keyframes", "qa.json")))
    qa_excerpt = json.dumps({"report": qa_doc["report"], "rows": qa_doc["rows"][:2] + ["..."]}, indent=2)

    h = []
    a = h.append

    # ---------- head: title, fonts, styles
    a('<title>From Mesh to Motion Reference</title>')
    a('<link rel="preconnect" href="https://fonts.googleapis.com">')
    a('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700'
      '&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">')
    a("<style>" + CSS + "</style>")

    a('<div class="page">')

    # ---------- title block
    a('<header class="sheet-head">')
    a('<p class="kicker">Technical reference</p>')
    a('<h1>From Mesh to Motion</h1>')
    a('<p class="lede">A rest-pose mesh and a one-line description of a motion in; a keyframe sheet out &mdash; '
      'sixteen keyframes, three orthographic views each, in register, with the time of every pose &mdash; '
      'for a rigging tool to animate from. Architecture, node contracts, data formats, metrics and '
      'measured results.</p>')
    cells = [
        ("Pack", "from-mesh-to-motion"), ("Version", ver), ("Pack nodes", str(len(nodes))),
        ("Graph nodes", str(n_graph_nodes)), ("Views", "front · side · back"),
        ("Projection", "orthographic"), ("Keyframes / run", "16"),
        ("Characters tested", str(len(runs))), ("Package", "{:.0f} KB".format(pkg_size / 1024)),
        ("Registry", '<a href="https://registry.comfy.org/nodes/from-mesh-to-motion">@meryboth</a>'),
        ("Source", '<a href="https://github.com/meryboth/from-mesh-to-motion">GitHub</a>'),
        ("Issued", today),
    ]
    a('<dl class="titleblock">' + "".join(
        '<div><dt>{}</dt><dd>{}</dd></div>'.format(k, v) for k, v in cells) + "</dl>")
    a("</header>")

    # ---------- layout: toc + main
    toc = [("overview", "Overview"), ("pipeline", "Pipeline"), ("graph", "ComfyUI graph"),
           ("nodes", "Nodes"), ("cli", "Command line"), ("data", "Data formats"),
           ("metrics", "Metrics"), ("results", "Results"), ("scope", "Scope"),
           ("usage", "Usage"), ("package", "Package")]
    a('<div class="shell">')
    a('<nav class="toc" aria-label="Contents"><p class="toc-h">Contents</p><ol>' + "".join(
        '<li><a href="#{}">{}</a></li>'.format(i, t) for i, t in toc) + "</ol></nav>")
    a('<main>')

    # ---------- overview
    a('<section id="overview"><h2>Overview</h2>')
    a('<p>An auto-rigger can read motion off pictures instead of guessing it from a sentence, if the '
      'pictures agree with each other. This pipeline produces pictures that agree. Its central move is '
      'that no view is ever generated alone: the three orthographic renders are placed on one canvas, '
      'and a video model animates that single image, so the three views move inside one frame and stay '
      'synchronised by construction.</p>')
    a('<figure><img src="assets/triptych-motion.gif" alt="Three orthographic panels of a mascot animating together"'
      ' loading="lazy"><figcaption>The animated canvas for the mascot. The three panels are one image '
      'to the video model.</figcaption></figure>')
    a('<div class="facts">'
      '<div><span class="fact-n">3</span><span class="fact-l">views per keyframe, one shared scale and pivot</span></div>'
      '<div><span class="fact-n">16</span><span class="fact-l">keyframes per run, each with its frame index and time</span></div>'
      '<div><span class="fact-n">1</span><span class="fact-l">remote, paid call per run; every other step is local</span></div>'
      "</div>")
    a("</section>")

    # ---------- pipeline
    a('<section id="pipeline"><h2>Pipeline</h2>')
    a('<p>Seven stages. Geometry is handled locally and deterministically; the only generative step '
      'is the animation.</p>')
    stages = [
        ("Mesh", "Cloud · optional", "Concept image", "Textured A-pose GLB", "workflows/01_concept_to_mesh.json (Meshy 7)"),
        ("Anchors", "Local · Blender", "GLB", "beauty_front / side / back.png", "python -m pipeline anchors"),
        ("Compose", "Local", "Three renders", "Canvas + layout JSON", "Triptych Compose"),
        ("Prompt", "Local", "One-line action + layout", "Full video prompt + summary", "Motion Prompt"),
        ("Animate", "Cloud · paid", "Canvas + prompt", "Video clip", "MiniMax H3"),
        ("Register & sample", "Local", "Clip + canvas", "Corrected layout, 16 frames with times", "Triptych Register, Sample Keyframes"),
        ("Split & deliver", "Local", "Frames + layout", "Sheet, per-view PNGs, keyframes.json, handoff.txt, QA", "Triptych Split, Keyframe Sheet, Save Keyframe Manifest, Identity QA"),
    ]
    a('<div class="tbl"><table><thead><tr><th>Stage</th><th>Runs</th><th>Input</th><th>Output</th>'
      '<th>Component</th></tr></thead><tbody>')
    for s in stages:
        cls = ' class="paid"' if "paid" in s[1] else ""
        a("<tr><td><strong>{}</strong></td><td{}>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            s[0], cls, s[1], s[2], s[3], s[4]))
    a("</tbody></table></div>")
    a('<figure><img src="assets/turnaround.jpg" alt="Four orthographic renders of the mascot" loading="lazy">'
      '<figcaption>Anchors: the object rotates, the camera never moves, so every view shares one '
      'orthographic scale and pivot.</figcaption></figure>')
    a("</section>")

    # ---------- graph
    a('<section id="graph"><h2>ComfyUI graph</h2>')
    a('<p><code>workflows/02_triptych_to_keyframes.json</code>, {} nodes, drawn from the workflow file. '
      'Reads top to bottom.</p>'.format(n_graph_nodes))
    a('<div class="diagram">' + graph_svg + "</div>")
    a('<ul class="legend"><li class="l-ours">pack node</li><li class="l-paid">partner node &mdash; remote, paid</li>'
      '<li class="l-core">ComfyUI core</li><li class="l-save">writes a file</li>'
      '<li class="l-layout">layout link</li></ul>')
    a('<p class="note">Not drawn: timings, frame rate and the motion summary, which run into Keyframe '
      'Sheet and Save Keyframe Manifest.</p>')
    a("<h3>Core nodes used</h3>")
    a('<div class="tbl"><table><thead><tr><th>Node</th><th>Responsibility</th><th>Runs</th></tr></thead><tbody>')
    for k, (resp, where) in CORE_NODES.items():
        a('<tr><td><code>{}</code></td><td>{}</td><td{}>{}</td></tr>'.format(
            k, resp, ' class="paid"' if where == "paid" else "",
            "remote · paid" if where == "paid" else "local"))
    a("</tbody></table></div></section>")

    # ---------- nodes
    a('<section id="nodes"><h2>Nodes</h2>')
    a('<p>The eight nodes in the pack, in the order the graph runs them. All are Pillow and numpy and '
      'run on the CPU. Contracts below are read from the node classes.</p>')
    for n in nodes:
        resp, rule = RESPONSIBILITY[n["key"]]
        a('<article class="node" id="{}">'.format(n["key"]))
        a('<div class="node-h"><h3>{}</h3><code>{}</code><span class="stage">{}</span>{}</div>'.format(
            escape(n["display"]), n["key"], STAGE[n["key"]],
            '<span class="stage out">output node</span>' if n["output_node"] else ""))
        a('<p>{}</p><p class="rule"><span>Rule</span>{}</p>'.format(resp, rule))
        a('<div class="io"><div><h4>Inputs</h4><div class="tbl"><table><tbody>')
        for i in n["inputs"]:
            d = "" if i["default"] in (None, "") else '<span class="def">{}</span>'.format(
                escape(str(i["default"])[:60] + ("…" if len(str(i["default"])) > 60 else "")))
            tip = '<div class="tip">{}</div>'.format(escape(i["tooltip"])) if i["tooltip"] else ""
            a('<tr><td><code>{}</code>{}</td><td class="ty">{}</td><td>{}{}</td></tr>'.format(
                i["name"], ' <span class="opt">optional</span>' if i["optional"] else "",
                i["type"], d, tip))
        a('</tbody></table></div></div><div><h4>Outputs</h4><div class="tbl"><table><tbody>')
        for name, ty in n["outputs"]:
            a('<tr><td><code>{}</code></td><td class="ty">{}</td></tr>'.format(name, ty))
        a("</tbody></table></div></div></div></article>")
    a("</section>")

    # ---------- cli
    a('<section id="cli"><h2>Command line</h2>')
    a('<p>The local half runs without ComfyUI: <code>python -m pipeline &lt;command&gt;</code>. '
      'Arguments are read from the parser.</p>')
    for c in cli:
        a('<h3><code>{}</code></h3><p>{}</p>'.format(c["name"], escape(c["help"])))
        a('<div class="tbl"><table><thead><tr><th>Argument</th><th>Default</th><th>Meaning</th></tr></thead><tbody>')
        for arg in c["args"]:
            a('<tr><td><code>{}</code>{}</td><td class="ty">{}</td><td>{}</td></tr>'.format(
                arg["flag"], ' <span class="opt req">required</span>' if arg["required"] else "",
                escape(str(arg["default"])), escape(arg["help"])))
        a("</tbody></table></div>")
    a("</section>")

    # ---------- data formats
    a('<section id="data"><h2>Data formats</h2>')
    a('<h3>Output folder</h3><div class="tbl"><table><thead><tr><th>File</th><th>For</th><th>Contents</th></tr></thead><tbody>'
      '<tr><td><code>keyframe_sheet.png</code></td><td>people, the rigging tool</td><td>16 rows × 3 views, each row labelled with its keyframe and time</td></tr>'
      '<tr><td><code>views/kNN_&lt;view&gt;.png</code></td><td>tools</td><td>48 single images, one per keyframe and view</td></tr>'
      '<tr><td><code>keyframes.json</code></td><td>tools</td><td>Motion, timing, views and provenance</td></tr>'
      '<tr><td><code>handoff.txt</code></td><td>the rigging tool</td><td>How to read the sheet; paste beside it</td></tr>'
      '<tr><td><code>qa.json</code></td><td>you</td><td>Drift per panel and the gate verdict (CLI); the graph shows the same as a report</td></tr>'
      "</tbody></table></div>")
    a('<figure><img src="assets/sheet-step.gif" alt="The sixteen keyframes stepping one at a time" loading="lazy">'
      '<figcaption>The sheet&rsquo;s content, one keyframe at a time.</figcaption></figure>')
    a('<h3>Layout <span class="sub">FMM_LAYOUT / triptych_rest.layout.json</span></h3>')
    a('<p>Where each panel sits on the canvas, in pixels. Travels through the graph as JSON; Triptych '
      'Register adds <code>transform</code> = [sx, sy, ox, oy].</p>')
    a('<pre><code>{}</code></pre>'.format(escape(json.dumps(layout_example, indent=2))))
    a('<h3><code>keyframes.json</code></h3>')
    a('<pre><code>{}</code></pre>'.format(escape(man_excerpt)))
    a('<h3><code>handoff.txt</code></h3><pre class="prose"><code>{}</code></pre>'.format(escape(handoff)))
    a('<h3><code>qa.json</code></h3><pre><code>{}</code></pre>'.format(escape(qa_excerpt)))
    a("</section>")

    # ---------- metrics
    a('<section id="metrics"><h2>Metrics</h2>')
    a('<h3>Identity drift</h3>')
    a('<p>How far a character&rsquo;s colours have moved from its rest pose, measured so that a change '
      'of pose does not count. Each view&rsquo;s first keyframe is reduced to six dominant colours by '
      'k-means; every panel&rsquo;s subject pixels are assigned to the nearest of them; drift is the '
      'displacement of each colour, weighted by the <em>rest pose&rsquo;s</em> colour mix.</p>')
    a('<pre class="formula"><code>drift = Σⱼ wⱼ · ‖ mean(panel pixels → cⱼ) − cⱼ ‖  /  (255 · Σⱼ wⱼ)\n'
      '        over colours j with ≥ 40 assigned pixels; cⱼ, wⱼ from k00 (k = 6)</code></pre>')
    a('<div class="tbl"><table><thead><tr><th>Drift</th><th>Reads as</th></tr></thead><tbody>'
      '<tr><td class="num">≈ 0.01</td><td>video compression</td></tr>'
      '<tr><td class="num">0.03</td><td>a tint you would notice &mdash; <strong>the gate</strong></td></tr>'
      '<tr><td class="num">≈ 0.07</td><td>a character that has changed colour</td></tr>'
      "</tbody></table></div>")
    a('<p>Weighting by the rest pose is what keeps it pose-blind: a raised arm changes how much of a '
      'colour is visible, not what the colour is. On the mascot, the measure moves 0.006&ndash;0.016 '
      'between arms-down and arms-up frames, while an 8/255 red tint reads three times its baseline.</p>')
    a("<h3>Registration</h3>")
    a('<p>Frame 0 of an image-to-video clip is nearly the submitted canvas, so the mapping back is read '
      'off the two content bounding boxes (flat background, tolerance 18/255):</p>')
    a('<pre class="formula"><code>sx = w_canvas / w_frame     sy = h_canvas / h_frame\n'
      'skew = |sx − sy| / max(sx, sy)      warning above 2 %</code></pre>')
    a('<p>Accuracy of the split panels against the source, mean absolute error per 255:</p>')
    a('<div class="tbl"><table><thead><tr><th>What the model returned</th><th class="num">Plain split</th>'
      '<th class="num">Registered</th></tr></thead><tbody>')
    labels = {"rescaled": "Rescaled (MiniMax H3)", "letterboxed": "Letterboxed into 16:9",
              "cropped": "Centre-cropped 6%"}
    for k, v in registration.items():
        a('<tr><td>{}</td><td class="num">{}</td><td class="num">{}</td></tr>'.format(
            labels.get(k, k), fmt(v["naive_mae"], 2), fmt(v["registered_mae"], 2)))
    a("</tbody></table></div>")
    a("<h3>Thresholds</h3>")
    a('<div class="tbl"><table><thead><tr><th>Check</th><th>Value</th><th>Where</th></tr></thead><tbody>'
      '<tr><td>Identity drift ceiling</td><td class="num">0.03</td><td>Identity QA · <code>pipeline qa --ceiling</code></td></tr>'
      '<tr><td>Registration skew warning</td><td class="num">2 %</td><td>Triptych Register · <code>pipeline keyframes</code></td></tr>'
      '<tr><td>Background tolerance</td><td class="num">18 / 255</td><td>content boxes, subject masks</td></tr>'
      '<tr><td>Palette size</td><td class="num">6</td><td><code>pipeline qa --colors</code></td></tr>'
      "</tbody></table></div></section>")

    # ---------- results
    a('<section id="results"><h2>Results</h2>')
    a('<p>Three characters with different body plans, each run end to end once: concept image, Meshy 7 '
      'mesh, Blender anchors, MiniMax H3 at 768P for five seconds, sixteen keyframes.</p>')
    a('<figure class="chart-fig">' + drift_chart(runs) +
      '<figcaption>Maximum identity drift per view (front, side, back), against the 0.03 gate.</figcaption></figure>')
    a('<ul class="legend chart-legend"><li class="v0">front</li><li class="v1">side</li><li class="v2">back</li></ul>')
    a('<div class="tbl"><table><thead><tr><th>Character</th><th>Canvas</th><th>Ratio</th><th>Clip</th>'
      '<th class="num">Reg. sx × sy</th><th class="num">Skew</th><th class="num">Drift mean</th>'
      '<th class="num">Drift max</th><th>Gate</th></tr></thead><tbody>')
    for r in runs:
        clip = r.get("clip")
        reg = r.get("registration")
        a('<tr><td><strong>{}</strong></td><td>{}×{}</td><td class="num">{}</td><td>{}</td>'
          '<td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td>'
          '<td>{}</td></tr>'.format(
              r["name"], r["canvas"][0], r["canvas"][1], fmt(r["ratio"], 3),
              "{}×{}, {} fr @ {:.0f} fps".format(clip["w"], clip["h"], clip["frames"], clip["fps"]) if clip else "—",
              "{} × {}".format(fmt(reg["sx"]), fmt(reg["sy"])) if reg else "—",
              "{:.2f} %".format(reg["skew_pct"]) if reg else "—",
              fmt(r["drift_mean"]), fmt(r["drift_max"]),
              '<span class="pill ok">pass</span>' if r["passed"] else '<span class="pill out">outside scope</span>'))
    a("</tbody></table></div>")
    for r in runs:
        a('<article class="result" id="r-{}"><h3>{}</h3><p class="body">{} &middot; motion: '
          '<em>{}</em></p>'.format(r["key"], r["name"], escape(r["body"]), escape(r["motion"])))
        a('<figure><img src="assets/triptych-{}.jpg" alt="{} rest-pose triptych" loading="lazy">'
          '<figcaption>Rest-pose canvas, {}×{} ({}).</figcaption></figure>'.format(
              r["key"], r["name"], r["canvas"][0], r["canvas"][1],
              "21:9" if abs(r["ratio"] - 21 / 9) < 0.01 else "16:9" if abs(r["ratio"] - 16 / 9) < 0.01 else fmt(r["ratio"], 3)))
        a('<div class="pair"><figure><img src="assets/{}" alt="{} animated canvas" loading="lazy">'
          '<figcaption>Animated canvas.</figcaption></figure>'.format(r["gif"], r["name"]))
        a('<figure><img src="assets/sheet-{}.jpg" alt="{} keyframes k00, k04, k07, k10, k13" loading="lazy">'
          '<figcaption>Keyframes k00, k04, k07, k10, k13 as cut from the sheet.</figcaption></figure></div>'.format(
              r["key"], r["name"]))
        a('<div class="tbl"><table class="mini"><thead><tr><th>View</th><th class="num">Mean drift</th>'
          '<th class="num">Max drift</th></tr></thead><tbody>')
        for v, d in r["views"].items():
            a('<tr><td>{}</td><td class="num">{}</td><td class="num">{}</td></tr>'.format(v, fmt(d["mean"]), fmt(d["max"])))
        a("</tbody></table></div>")
        ts = ", ".join("k{:02d} {:.2f}s".format(k["i"], k["t"]) for k in r["keyframes"][::3])
        secs = r["clip"]["seconds"] if r.get("clip") else r["duration_s"]
        a('<p class="note">Timing: {} keyframes across a {:.2f} s clip at {:.0f} fps &mdash; {}.</p>'.format(
            len(r["keyframes"]), secs, r["fps"], ts))
        a("</article>")
    a('<p>The thin robot&rsquo;s panels stay synchronised &mdash; its one-sided wave appears mirrored '
      'correctly in the back view on every keyframe &mdash; while its wire-thin profile and loose scarf '
      'fall outside the pipeline&rsquo;s scope and are flagged by the gate.</p>')
    a("</section>")

    # ---------- scope
    a('<section id="scope"><h2>Scope</h2>')
    a('<div class="scope"><div><h3>Designed for</h3><ul>'
      '<li>Chunky, rigid or near-rigid characters, readable from every side, in an A-pose &mdash; humanoid or not</li>'
      '<li>Body motion performed in place, one beat per run: raising arms, waving, crouching, stretching, rearing up, nodding, idles</li>'
      '<li>One clip per run, 4&ndash;15 s as MiniMax H3 accepts (runs here used 5 s); 16 keyframes by default</li>'
      '</ul></div><div><h3>Outside its scope</h3><ul>'
      '<li>Travelling across the frame &mdash; in-place cycles are in scope</li>'
      '<li>Turning on the spot &mdash; the panels are defined as front, side, back</li>'
      '<li>Secondary motion: cloth, hair, free-moving parts</li>'
      '<li>Multi-beat sequences &mdash; built from several runs sharing end poses</li>'
      '<li>Very thin silhouettes</li>'
      '</ul></div><div><h3>Stays with you</h3><ul>'
      '<li>Rigging and the final animation &mdash; Astra, Meshy or a person</li>'
      '<li>The orthographic renders, in Blender with the bundled CLI</li>'
      '<li>A last look at the side column</li>'
      "</ul></div></div></section>")

    # ---------- usage
    a('<section id="usage"><h2>Usage</h2><ol class="steps">'
      '<li><strong>Install</strong> ComfyUI and the pack: ComfyUI Manager, or <code>comfy node install from-mesh-to-motion</code>.</li>'
      '<li><strong>Sign in</strong> to a Comfy account in ComfyUI, or set <em>Settings → API Keys → Comfy API Key</em>. Only the MiniMax node uses it.</li>'
      '<li><strong>Render the views</strong> from the pack folder: <code>python -m pipeline anchors --mesh character.glb --out anchors</code> (add <code>--yaw-offset 90</code> if the character faces the wrong way).</li>'
      '<li><strong>Open</strong> <code>workflows/02_triptych_to_keyframes.json</code> and load <code>beauty_front</code>, <code>beauty_side</code>, <code>beauty_back</code> into the three Load Image nodes.</li>'
      '<li><strong>Write the action</strong> in Motion Prompt &mdash; only what the character does.</li>'
      '<li><strong>Queue</strong>: one paid call, a few minutes; change the seed on the video node for another take.</li>'
      '<li><strong>Check</strong> Triptych Register&rsquo;s skew and Identity QA&rsquo;s verdict.</li>'
      '<li><strong>Hand over</strong> the rigged rest-pose mesh, <code>keyframe_sheet.png</code> and <code>handoff.txt</code> from <code>ComfyUI/output/from_mesh_to_motion/</code>.</li>'
      "</ol></section>")

    # ---------- package
    a('<section id="package"><h2>Package</h2>')
    a('<p>Published to the Comfy Registry as <code>from-mesh-to-motion</code> {} under <code>@meryboth</code>; '
      'a version bump in <code>pyproject.toml</code> publishes through GitHub Actions. The registry package '
      'is {} files, {:.0f} KB; example renders, meshes, clips and the write-ups stay in the repository.</p>'.format(
          ver, len(files), pkg_size / 1024))
    a('<div class="tbl"><table><thead><tr><th>Shipped file</th><th class="num">Size</th></tr></thead><tbody>')
    for f in files:
        a('<tr><td><code>{}</code></td><td class="num">{:.1f} KB</td></tr>'.format(
            f, os.path.getsize(os.path.join(REPO, f)) / 1024))
    a("</tbody></table></div>")
    a('<h3>Models used in the documented runs</h3><div class="tbl"><table><tbody>'
      '<tr><td>Concept images</td><td>Flux 2 Pro</td></tr>'
      '<tr><td>Meshes</td><td>Meshy 7, 25k-face quad remesh, 2k texture</td></tr>'
      '<tr><td>Animation</td><td>MiniMax H3 image-to-video, 768P, 5 s, seed 11</td></tr>'
      '<tr><td>Anchors</td><td>Blender 5.1 EEVEE, 768 px square, orthographic</td></tr>'
      "</tbody></table></div>")
    a('<p class="note">Code MIT; the characters, meshes and renders are CC0, generated for this project. '
      'Generated {} by <code>scripts/build_reference.py</code>.</p>'.format(today))
    a("</section>")

    a("</main></div></div>")
    return "\n".join(h)


CSS = r"""
:root{
  --paper:#f3f4f2; --surface:#fbfbfa; --ink:#1b1e22; --ink-2:#4a5058; --ink-3:#747b84;
  --rule:#d6d9d6; --rule-soft:#e6e8e5; --code:#eceeeb;
  --ours:#1f7a80; --ours-soft:#e3f0f0; --paid:#c4561f; --paid-soft:#f8e8de;
  --ok:#2f7a45; --ok-soft:#e2f0e6; --out:#8a6a1a; --out-soft:#f4ecd6;
  --v0:#1f7a80; --v1:#7d8f3a; --v2:#c4561f;
  --display:"Barlow Condensed","Arial Narrow",sans-serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#15181b; --surface:#1b1f23; --ink:#e7e9ea; --ink-2:#b3b9bf; --ink-3:#8a9199;
  --rule:#2f353b; --rule-soft:#262b30; --code:#20252a;
  --ours:#57b6bb; --ours-soft:#17292b; --paid:#ef8a52; --paid-soft:#2c1e16;
  --ok:#6cc488; --ok-soft:#162619; --out:#d9b35a; --out-soft:#2a2414;
  --v0:#57b6bb; --v1:#a9bb5f; --v2:#ef8a52;
}}
:root[data-theme="dark"]{
  --paper:#15181b; --surface:#1b1f23; --ink:#e7e9ea; --ink-2:#b3b9bf; --ink-3:#8a9199;
  --rule:#2f353b; --rule-soft:#262b30; --code:#20252a;
  --ours:#57b6bb; --ours-soft:#17292b; --paid:#ef8a52; --paid-soft:#2c1e16;
  --ok:#6cc488; --ok-soft:#162619; --out:#d9b35a; --out-soft:#2a2414;
  --v0:#57b6bb; --v1:#a9bb5f; --v2:#ef8a52;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15.5px;line-height:1.62;
  -webkit-font-smoothing:antialiased;margin:0;padding-inline:20px;padding-block:0 64px}
.page{max-width:1120px;margin:0 auto}
a{color:var(--ours);text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--ours);outline-offset:2px}
h1,h2,h3,.kicker,.toc-h,dt,.stage,.fact-n,th{font-family:var(--display)}
h1{font-size:clamp(2.4rem,6vw,3.6rem);line-height:1;margin:6px 0 14px;font-weight:700;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:1.9rem;font-weight:700;margin:0 0 12px;letter-spacing:.005em;text-wrap:balance}
h3{font-size:1.3rem;font-weight:600;margin:30px 0 8px;letter-spacing:.01em}
h3 code,h2 code{font-family:var(--mono);font-size:.85em}
h4{font:600 .78rem/1 var(--sans);text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3);margin:0 0 6px}
p{margin:0 0 14px;max-width:68ch}
code{font-family:var(--mono);font-size:.86em;background:var(--code);padding:.1em .35em;border-radius:3px}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;overflow-x:auto;
  font-family:var(--mono);font-size:.8rem;line-height:1.55;margin:0 0 18px}
pre code{background:none;padding:0}
pre.prose{white-space:pre-wrap}
pre.formula{background:var(--ours-soft);border-color:transparent;border-left:3px solid var(--ours)}
.sheet-head{padding-block:48px 28px;border-bottom:2px solid var(--ink)}
.kicker{font-size:.9rem;font-weight:600;text-transform:uppercase;letter-spacing:.16em;color:var(--ours);margin:0}
.lede{font-size:1.08rem;color:var(--ink-2);max-width:70ch;margin-bottom:26px}
.titleblock{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:0;
  border-top:1px solid var(--ink);border-left:1px solid var(--ink)}
.titleblock div{border-right:1px solid var(--ink);border-bottom:1px solid var(--ink);padding:7px 10px 8px;min-width:0}
.titleblock dt{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--ink-3)}
.titleblock dd{margin:1px 0 0;font-family:var(--mono);font-size:.84rem;overflow-wrap:anywhere}
.shell{display:grid;grid-template-columns:1fr;gap:28px;margin-top:32px}
@media (min-width:1000px){.shell{grid-template-columns:180px minmax(0,1fr)}
  .toc{position:sticky;top:calc(env(safe-area-inset-top,0px) + 20px);align-self:start}}
.toc-h{font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.14em;color:var(--ink-3);margin:0 0 8px}
.toc ol{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:4px 14px}
@media (min-width:1000px){.toc ol{display:block}.toc li{margin:0 0 5px}}
.toc a{color:var(--ink-2);text-decoration:none;font-size:.9rem}
.toc a:hover{color:var(--ours)}
main{min-width:0}
section{padding-block:30px 26px;border-top:1px solid var(--rule)}
section:first-child{border-top:0;padding-top:0}
figure{margin:18px 0 22px}
figure img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
figcaption{font-size:.84rem;color:var(--ink-3);margin-top:7px;max-width:70ch}
.tbl{overflow-x:auto;margin:0 0 18px}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th{font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);text-align:left;
  padding:7px 10px;border-bottom:1.5px solid var(--ink-3);white-space:nowrap}
td{padding:8px 10px;border-bottom:1px solid var(--rule-soft);vertical-align:top}
.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.ty{font-family:var(--mono);font-size:.8rem;color:var(--ink-2);white-space:nowrap}
td.paid{color:var(--paid);font-weight:500}
.opt{font-size:.72rem;color:var(--ink-3);font-style:italic}
.opt.req{color:var(--paid);font-style:normal}
.def{font-family:var(--mono);font-size:.8rem;background:var(--code);padding:.05em .35em;border-radius:3px}
.tip{color:var(--ink-3);font-size:.8rem;margin-top:3px;max-width:52ch}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--rule);
  border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin:6px 0 4px}
.facts div{background:var(--surface);padding:14px 16px;display:flex;gap:12px;align-items:baseline}
.fact-n{font-size:2.2rem;font-weight:700;color:var(--ours);line-height:1}
.fact-l{font-size:.86rem;color:var(--ink-2)}
.diagram{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:10px}
.graph{width:100%;min-width:560px;height:auto;display:block;font-family:var(--sans)}
.graph .g-edge{fill:none;stroke:var(--ink-3);stroke-width:1.4;opacity:.8}
.graph .g-edge.layout{stroke:var(--ours);opacity:1}
.graph .g-arrow{fill:var(--ink-3)}
.graph .g-mult{font-size:11px;fill:var(--ink-3)}
.graph .g-node rect{fill:var(--paper);stroke:var(--rule);stroke-width:1.2}
.graph .g-node.ours rect{fill:var(--ours-soft);stroke:var(--ours)}
.graph .g-node.cloud rect{fill:var(--paid-soft);stroke:var(--paid)}
.graph .g-node.save rect{stroke-dasharray:4 3}
.graph .g-name{font-size:13px;font-weight:600;fill:var(--ink);text-anchor:middle}
.graph .g-sub{font-size:11px;fill:var(--ink-2);text-anchor:middle}
.legend{list-style:none;padding:0;margin:10px 0 6px;display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.82rem;color:var(--ink-3)}
.legend li::before{content:"";display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;
  vertical-align:-1px;border:1px solid var(--rule);background:var(--paper)}
.legend .l-ours::before{background:var(--ours-soft);border-color:var(--ours)}
.legend .l-paid::before{background:var(--paid-soft);border-color:var(--paid)}
.legend .l-save::before{border-style:dashed}
.legend .l-layout::before{height:0;border:0;border-top:2px solid var(--ours);border-radius:0;vertical-align:3px}
.legend .v0::before{background:var(--v0);border:0}.legend .v1::before{background:var(--v1);border:0}
.legend .v2::before{background:var(--v2);border:0}
.note{font-size:.84rem;color:var(--ink-3)}
.node{border-top:1px solid var(--rule);padding-block:18px 6px}
.node-h{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px;margin-bottom:8px}
.node-h h3{margin:0}
.node-h code{font-size:.78rem;color:var(--ink-3)}
.stage{font-size:.74rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--ours);
  border:1px solid var(--ours);border-radius:3px;padding:1px 6px}
.stage.out{color:var(--ink-3);border-color:var(--rule)}
.rule{font-size:.9rem;color:var(--ink-2)}
.rule span{font:600 .72rem var(--sans);text-transform:uppercase;letter-spacing:.1em;color:var(--ink-3);margin-right:8px}
.io{display:grid;grid-template-columns:1fr;gap:4px 24px}
@media (min-width:760px){.io{grid-template-columns:minmax(0,2fr) minmax(0,1fr)}}
.chart-fig{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 14px 8px}
.chart{width:100%;height:auto;display:block;font-family:var(--sans)}
.chart .grid{stroke:var(--rule-soft);stroke-width:1}
.chart .tick{font-size:10.5px;fill:var(--ink-3);text-anchor:end;font-family:var(--mono)}
.chart .val{font-size:10px;fill:var(--ink-2);text-anchor:middle;font-family:var(--mono)}
.chart .grp{font-size:12.5px;fill:var(--ink);text-anchor:middle;font-weight:600}
.chart .bar.v0{fill:var(--v0)}.chart .bar.v1{fill:var(--v1)}.chart .bar.v2{fill:var(--v2)}
.chart .ceil{stroke:var(--ink);stroke-width:1.2;stroke-dasharray:5 4}
.chart .ceil-l{font-size:10.5px;fill:var(--ink);text-anchor:start;font-family:var(--mono)}
.chart-legend{margin-top:-8px}
.pill{font-size:.76rem;font-weight:600;padding:2px 8px;border-radius:99px;white-space:nowrap}
.pill.ok{background:var(--ok-soft);color:var(--ok)}
.pill.out{background:var(--out-soft);color:var(--out)}
.result{border-top:1px solid var(--rule);padding-top:6px;margin-top:22px}
.result .body{color:var(--ink-2);font-size:.92rem}
.pair{display:grid;grid-template-columns:1fr;gap:0 18px}
@media (min-width:760px){.pair{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);align-items:start}}
table.mini{max-width:420px}
.scope{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 26px}
.scope h3{margin-top:10px}
.scope ul,.steps{padding-left:1.2em;margin:0}
.scope li,.steps li{margin:0 0 8px}
.steps{max-width:70ch}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", default=None,
                    help="Also write a copy without the html/head/body wrapper")
    args = ap.parse_args(argv)

    import build_graph_diagram as G

    pack = load_pack()
    runs = collect_runs()
    make_figures(runs)
    graph = json.load(open(os.path.join(REPO, "workflows", "02_triptych_to_keyframes.json")))
    registration = json.load(open(os.path.join(ASSETS, "registration.json")))

    body = render(runs, node_contracts(pack), cli_reference(), G.build_svg(graph),
                  package_contents(), version(), registration)

    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '</head>\n<body>\n' + body + "\n</body>\n</html>\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print("reference -> " + os.path.relpath(OUT, REPO)
          + "  ({:.0f} KB, {} runs)".format(len(doc) / 1024, len(runs)))
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("fragment  -> " + args.fragment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
