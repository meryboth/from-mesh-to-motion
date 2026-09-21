"""Draw the ComfyUI graph into the write-up, from the workflow file itself.

A hand-drawn diagram of a graph is a second copy of the graph, and second copies
drift. This one reads workflows/02_triptych_to_keyframes.json, so a node or link
added there shows up here the next time it runs -- and a node it does not know
where to put makes it fail loudly instead of drawing something stale.

  python scripts/build_graph_diagram.py

The SVG is written inline between two markers in docs/report/index.html, not as
an <img>, so it picks up the page's light and dark colour tokens.
"""

from __future__ import annotations

import json
import os
import sys
from html import escape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(REPO, "workflows", "02_triptych_to_keyframes.json")
REPORT = os.path.join(REPO, "docs", "report", "index.html")
START, END = "<!-- graph:start -->", "<!-- graph:end -->"

W, H = 760, 985
BOX_W, BOX_H, SMALL_W = 206, 54, 150

# Where each node sits (centre x, centre y). Stages read top to bottom.
POS = {
    1: (135, 50), 2: (380, 50), 3: (625, 50),
    10: (250, 165), 5: (560, 165),
    11: (650, 280), 20: (400, 280),
    22: (400, 395), 21: (650, 395),
    23: (330, 500),
    30: (160, 610), 31: (580, 610),
    32: (400, 720),
    40: (140, 835), 42: (400, 835), 43: (650, 835),
    41: (140, 945),
}

# One line under each title: what the node is for, not what it is.
SUBTITLE = {
    1: "front render", 2: "side render", 3: "back render",
    5: "you write the action here",
    10: "3 views -> one canvas",
    11: "keeps the submitted canvas",
    20: "animates all panels at once",
    21: "keeps the clip",
    22: "clip -> frames",
    23: "frame 0 only",
    30: "undoes the model's resize",
    31: "16 frames + their times",
    32: "canvas -> 3 view batches",
    40: "labelled contact sheet",
    41: "writes the sheet PNG",
    42: "keyframes.json, handoff.txt",
    43: "drift gate: pass / fail",
}

# Links that carry numbers rather than pixels: timings, fps, the motion summary.
# They all run into the last row, and drawn they made a tangle that hid the data
# path, so they are left out and named under the figure instead. The one kept is
# the prompt, which is the whole point of the Motion Prompt node.
META_TYPES = {"FLOAT", "STRING"}
KEEP_META = {(5, 20)}


def kind(node_type: str) -> str:
    if node_type.startswith("FMM_"):
        return "ours"
    if node_type.startswith("Minimax"):
        return "cloud"
    if node_type.startswith("Save"):
        return "save"
    if node_type == "LoadImage":
        return "input"
    return "core"


def box(nid):
    x, y = POS[nid]
    w = SMALL_W if nid in (11, 21, 41) else BOX_W
    return x - w / 2, y - BOX_H / 2, w, BOX_H


def edge_path(src, dst, lane_state):
    sx, sy, sw, sh = box(src)
    dx, dy, dw, dh = box(dst)
    x1, y1 = sx + sw / 2, sy + sh
    x2, y2 = dx + dw / 2, dy

    if abs((sy + sh / 2) - (dy + dh / 2)) < 5:            # same row: side to side
        if x1 < x2:
            a, b = (sx + sw, sy + sh / 2), (dx, dy + dh / 2)
        else:
            a, b = (sx, sy + sh / 2), (dx + dw, dy + dh / 2)
        return "M{:.0f},{:.0f} L{:.0f},{:.0f}".format(a[0], a[1], b[0], b[1])

    if y2 - y1 < 150:                                     # short: a soft curve
        my = (y1 + y2) / 2
        return "M{:.0f},{:.0f} C{:.0f},{:.0f} {:.0f},{:.0f} {:.0f},{:.0f}".format(
            x1, y1, x1, my, x2, my, x2, y2)

    # Long: run down an outside lane so it does not cut through other nodes.
    side = "left" if (x1 + x2) / 2 < W / 2 else "right"
    k = lane_state[side]
    lane_state[side] += 1
    lx = 12 + 9 * k if side == "left" else W - 12 - 9 * k
    r = 8
    s = 1 if lx > x1 else -1
    e = 1 if x2 > lx else -1
    return ("M{x1:.0f},{y1:.0f} L{x1:.0f},{ya:.0f} "
            "Q{x1:.0f},{yb:.0f} {x1r:.0f},{yb:.0f} L{lxa:.0f},{yb:.0f} "
            "Q{lx:.0f},{yb:.0f} {lx:.0f},{ybr:.0f} L{lx:.0f},{yca:.0f} "
            "Q{lx:.0f},{yc:.0f} {lxe:.0f},{yc:.0f} L{x2e:.0f},{yc:.0f} "
            "Q{x2:.0f},{yc:.0f} {x2:.0f},{ycr:.0f} L{x2:.0f},{y2:.0f}").format(
        x1=x1, y1=y1, ya=y1 + 6, yb=y1 + 14, x1r=x1 + s * r, lxa=lx - s * r,
        lx=lx, ybr=y1 + 14 + r, yca=y2 - 14 - r, yc=y2 - 14, lxe=lx + e * r,
        x2e=x2 - e * r, x2=x2, ycr=y2 - 14 + r, y2=y2)


def build_svg(graph: dict) -> str:
    nodes = {n["id"]: n for n in graph["nodes"]}
    unknown = sorted(set(nodes) - set(POS))
    if unknown:
        raise SystemExit(
            "The workflow has nodes this diagram does not place: " + repr(unknown)
            + ". Add them to POS and SUBTITLE rather than letting the figure lie."
        )

    # Merge parallel links (Split -> Sheet carries three) into one drawn edge.
    edges = {}
    for _lid, src, _ss, dst, _ds, ltype in graph["links"]:
        key = (src, dst)
        edges.setdefault(key, []).append(ltype)

    lanes = {"left": 0, "right": 0}
    parts = [
        '<svg class="graph" viewBox="0 0 {w} {h}" role="img" '
        'aria-label="The ComfyUI graph, top to bottom" '
        'xmlns="http://www.w3.org/2000/svg">'.format(w=W, h=H),
        '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" class="g-arrow"/></marker></defs>',
    ]

    ordered = sorted(edges.items(),
                     key=lambda kv: POS[kv[0][1]][1] - POS[kv[0][0]][1])
    for (src, dst), types in ordered:
        if (src, dst) not in KEEP_META:
            types = [t for t in types if t not in META_TYPES]
        if not types:
            continue
        cls = "g-edge layout" if "FMM_LAYOUT" in types else "g-edge"
        d = edge_path(src, dst, lanes)
        parts.append('<path class="{c}" d="{d}" marker-end="url(#arr)"/>'.format(c=cls, d=d))
        # Multiplicity only on short edges, where the label can sit on the line.
        if len(types) > 1 and " Q" not in d:
            x1, y1 = POS[src]
            x2, y2 = POS[dst]
            parts.append('<text class="g-mult" x="{:.0f}" y="{:.0f}">x{}</text>'.format(
                (x1 + x2) / 2 + 8, (y1 + y2) / 2 + 4, len(types)))

    for nid, node in nodes.items():
        x, y, w, h = box(nid)
        k = kind(node["type"])
        title = node.get("title", node["type"])
        # Strip the "3 - " step prefixes the graph uses; the stage is in the layout.
        if len(title) > 3 and title[0].isdigit() and title[1:4] == " - ":
            title = title[4:]
        title = {1: "Load image", 2: "Load image", 3: "Load image"}.get(nid, title)
        name = node["type"].replace("FMM_", "")
        if k == "cloud":
            name = "MiniMax H3  (paid)"
        parts.append(
            '<g class="g-node {k}"><rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" '
            'height="{h:.0f}" rx="8"/>'
            '<text class="g-name" x="{cx:.0f}" y="{t1:.0f}">{name}</text>'
            '<text class="g-sub" x="{cx:.0f}" y="{t2:.0f}">{sub}</text></g>'.format(
                k=k, x=x, y=y, w=w, h=h, cx=x + w / 2, t1=y + 22, t2=y + 40,
                name=escape(name), sub=escape(SUBTITLE.get(nid, title))))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    with open(WORKFLOW, "r", encoding="utf-8") as fh:
        graph = json.load(fh)
    svg = build_svg(graph)

    with open(REPORT, "r", encoding="utf-8") as fh:
        html = fh.read()
    if START not in html or END not in html:
        raise SystemExit("Markers not found in " + REPORT)
    head, rest = html.split(START, 1)
    _, tail = rest.split(END, 1)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(head + START + "\n" + svg + "\n" + END + tail)

    print("diagram: " + str(len(graph["nodes"])) + " nodes, "
          + str(len(graph["links"])) + " links -> " + os.path.relpath(REPORT, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
