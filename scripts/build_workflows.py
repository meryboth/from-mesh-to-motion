"""Generate the ComfyUI graphs in workflows/ from one declarative spec.

Graphs are written twice: the editor save format that ComfyUI opens, and the API
format a headless run posts to /prompt. Hand-maintaining those two in parallel is
how they drift, so both come from the same node list here.

  python scripts/build_workflows.py

Widget order matters and is not inferable: ComfyUI stores widget values as a flat
positional list, so each node below declares its widgets in the exact order the
node class defines them. For a dynamic-combo node that means the selector first,
then its conditional sub-fields.
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "workflows")

MOTION_PROMPT = (
    "A three-panel split-screen character turnaround sheet on a flat light grey "
    "background. The same clay toy character is shown three times side by side, each "
    "panel a locked-off camera on a different angle: left panel front view, centre "
    "panel side profile, right panel back view. The camera is completely static in "
    "all three panels: no pan, no zoom, no dolly, no rotation, no cuts. The three "
    "panels never move, never resize, and never change their viewing angle. All three "
    "copies of the character perform one identical action in perfect unison, frame for "
    "frame, each seen from its own fixed angle: the character raises both arms "
    "straight overhead, bends its knees, springs up into a single jump, lands, and "
    "lowers its arms back down. The character stays centred in its own panel and does "
    "not walk out of frame. Nothing is added to the scene."
)

MOTION_SHORT = "Raises both arms overhead, bends its knees, jumps once, lands and lowers its arms"


class Node:
    """One graph node: its class, its widget values, and its wired inputs."""

    def __init__(self, nid, class_type, pos, widgets=None, links=None,
                 outputs=None, title=None, size=(340, 120), optional=(),
                 widget_links=None):
        self.id = nid
        self.class_type = class_type
        self.pos = list(pos)
        self.size = list(size)
        self.title = title
        # widgets: ordered [(name, value), ...] in the node class's own order.
        # A name prefixed with "~" is a frontend-only control (LoadImage's upload
        # button): it occupies a slot in widgets_values but is not a real input,
        # so the API format must leave it out or the server rejects the graph.
        self.widgets = list(widgets or [])
        # links: {input_name: (source_node_id, source_slot, type)}
        self.links = dict(links or {})
        # outputs: [(name, type), ...]
        self.outputs = list(outputs or [])
        # Input names the node class declares as optional. ComfyUI marks these
        # with shape 7 so the editor draws them as an optional socket.
        self.optional = set(optional)
        # Links that drive a WIDGET instead of a socket. ComfyUI marks these with
        # a "widget" key on the input and lists them AFTER the real sockets, so
        # they are kept separate to preserve that ordering.
        self.widget_links = dict(widget_links or {})


def build_save_format(nodes, title):
    """Emit the editor graph format (what ComfyUI's Load opens)."""
    by_id = {n.id: n for n in nodes}
    links = []
    link_id = 0

    # Pre-compute, per node, the list of outgoing link ids per output slot.
    out_links = {n.id: {i: [] for i in range(len(n.outputs))} for n in nodes}

    for node in nodes:
        ordered = list(node.links) + list(node.widget_links)
        for input_name in ordered:
            store = node.links if input_name in node.links else node.widget_links
            src_id, src_slot, ltype = store[input_name][:3]
            link_id += 1
            target_slot = ordered.index(input_name)
            links.append([link_id, src_id, src_slot, node.id, target_slot, ltype])
            out_links[src_id][src_slot].append(link_id)
            store[input_name] = (src_id, src_slot, ltype, link_id)

    graph_nodes = []
    for order, node in enumerate(nodes):
        inputs = []
        for name, (src_id, src_slot, ltype, lid) in node.links.items():
            entry_in = {"name": name, "type": ltype, "link": lid}
            if name in node.optional:
                entry_in["shape"] = 7
            inputs.append(entry_in)
        for name, (src_id, src_slot, ltype, lid) in node.widget_links.items():
            inputs.append({"name": name, "type": ltype,
                           "widget": {"name": name}, "link": lid})
        outputs = []
        for slot, (name, otype) in enumerate(node.outputs):
            outputs.append({
                "name": name, "type": otype,
                "links": out_links[node.id][slot] or None,
                "slot_index": slot,
            })
        entry = {
            "id": node.id,
            "type": node.class_type,
            "pos": node.pos,
            "size": node.size,
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": node.class_type},
            "widgets_values": [v for _, v in node.widgets],
            "widgets_values_named": {
                n.lstrip("~"): v for n, v in node.widgets
            },
        }
        if node.title:
            entry["title"] = node.title
        graph_nodes.append(entry)

    return {
        "id": title,
        "revision": 0,
        "last_node_id": max(n.id for n in nodes),
        "last_link_id": link_id,
        "nodes": graph_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def build_api_format(nodes):
    """Emit the /prompt format: node-id keyed, widgets and links in one dict."""
    api = {}
    for node in nodes:
        inputs = {name: value for name, value in node.widgets
                  if not name.startswith("~")}
        for name, link in list(node.links.items()) + list(node.widget_links.items()):
            inputs[name] = [str(link[0]), link[1]]
        api[str(node.id)] = {"class_type": node.class_type, "inputs": inputs}
    return api


# ------------------------------------------------------ the core workflow


def triptych_to_keyframes():
    """Three ortho views in, a labelled keyframe sheet and a manifest out."""
    n = []

    for i, view in enumerate(["front", "side", "back"]):
        n.append(Node(
            1 + i, "LoadImage", (40, 40 + i * 190),
            widgets=[("image", "beauty_" + view + ".png"), ("~upload", "image")],
            outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
            title="Ortho anchor - " + view,
            size=(300, 314),
        ))

    n.append(Node(
        5, "FMM_MotionPrompt", (400, 620),
        widgets=[
            ("action", "raises both arms straight overhead, bends its knees, "
                       "springs up into a single jump, lands, and lowers its arms "
                       "back down"),
            ("subject", "clay toy character"),
            ("loop", False),
            ("view_names", "front,side,back"),
            ("extra", ""),
        ],
        links={"layout": (10, 1, "FMM_LAYOUT")},
        optional=("layout",),
        outputs=[("prompt", "STRING"), ("summary", "STRING")],
        title="Write only the action here",
        size=(360, 260),
    ))

    n.append(Node(
        10, "FMM_TriptychCompose", (400, 40),
        widgets=[
            ("names", "front,side,back"),
            ("gutter", 20), ("pad", 20),
            ("background", "#E4E4E6"),
            ("target_ratio", "21:9"),
            ("crop_margin", 0.08),
        ],
        links={
            "view_a": (1, 0, "IMAGE"),
            "view_b": (2, 0, "IMAGE"),
            "view_c": (3, 0, "IMAGE"),
        },
        outputs=[("triptych", "IMAGE"), ("layout", "FMM_LAYOUT")],
        title="1 - Compose the triptych",
        size=(340, 200),
    ))

    n.append(Node(
        11, "SaveImage", (400, 300),
        widgets=[("filename_prefix", "fmm/triptych")],
        links={"images": (10, 0, "IMAGE")},
        title="Keep the canvas that was submitted",
        size=(300, 270),
    ))

    n.append(Node(
        20, "MinimaxHailuo03FirstLastFrameNode", (790, 40),
        widgets=[
            ("model", "MiniMax H3"),
            # Driven by the Motion Prompt node; the value is kept so the graph
            # still reads sensibly if that link is ever removed.
            ("model.prompt", MOTION_PROMPT),
            ("model.resolution", "768P"),
            ("model.duration", 5),
            ("seed", 11),
            ("~control_after_generate", "fixed"),
            ("watermark", False),
        ],
        links={"first_frame": (10, 0, "IMAGE")},
        widget_links={"model.prompt": (5, 0, "STRING")},
        outputs=[("VIDEO", "VIDEO")],
        title="2 - Animate all three panels at once",
        size=(420, 330),
    ))

    n.append(Node(
        21, "SaveVideo", (790, 410),
        widgets=[("filename_prefix", "fmm/motion"), ("format", "auto"),
                 ("format.codec", "auto")],
        links={"video": (20, 0, "VIDEO")},
        title="Keep the clip",
        size=(320, 240),
    ))

    n.append(Node(
        22, "GetVideoComponents", (790, 690),
        links={"video": (20, 0, "VIDEO")},
        outputs=[("IMAGE", "IMAGE"), ("AUDIO", "AUDIO"), ("FLOAT", "FLOAT"),
                 ("COMBO", "COMBO"), ("COMBO", "COMBO")],
        title="Clip to frames",
        size=(280, 120),
    ))

    n.append(Node(
        23, "ImageFromBatch", (790, 860),
        widgets=[("batch_index", 0), ("length", 1)],
        links={"image": (22, 0, "IMAGE")},
        outputs=[("IMAGE", "IMAGE")],
        title="First frame",
        size=(260, 90),
    ))

    n.append(Node(
        30, "FMM_TriptychRegister", (1250, 40),
        links={
            "triptych": (10, 0, "IMAGE"),
            "first_frame": (23, 0, "IMAGE"),
            "layout": (10, 1, "FMM_LAYOUT"),
        },
        outputs=[("layout", "FMM_LAYOUT"), ("report", "STRING")],
        title="3 - Undo the model's reshaping",
        size=(330, 110),
    ))

    n.append(Node(
        31, "FMM_SampleKeyframes", (1250, 210),
        widgets=[("count", 16), ("fps", 24.0), ("loop", False)],
        links={"frames": (22, 0, "IMAGE"), "fps_in": (22, 2, "FLOAT")},
        optional=("fps_in",),
        outputs=[("keyframes", "IMAGE"), ("timings", "STRING")],
        title="4 - Pick 16 keyframes",
        size=(330, 150),
    ))

    n.append(Node(
        32, "FMM_TriptychSplit", (1250, 420),
        links={"frames": (31, 0, "IMAGE"), "layout": (30, 0, "FMM_LAYOUT")},
        outputs=[("view_a", "IMAGE"), ("view_b", "IMAGE"),
                 ("view_c", "IMAGE"), ("names", "STRING")],
        title="5 - Cut the panels apart",
        size=(330, 120),
    ))

    n.append(Node(
        40, "FMM_KeyframeSheet", (1660, 40),
        widgets=[("names", "front,side,back"), ("timings", ""), ("thumb", 256)],
        links={
            "view_a": (32, 0, "IMAGE"),
            "view_b": (32, 1, "IMAGE"),
            "view_c": (32, 2, "IMAGE"),
        },
        outputs=[("sheet", "IMAGE")],
        title="6 - The contact sheet",
        size=(330, 200),
    ))

    n.append(Node(
        41, "SaveImage", (1660, 290),
        widgets=[("filename_prefix", "fmm/keyframe_sheet")],
        links={"images": (40, 0, "IMAGE")},
        title="Save the sheet",
        size=(320, 300),
    ))

    n.append(Node(
        42, "FMM_SaveKeyframeManifest", (2040, 40),
        widgets=[
            ("names", "front,side,back"),
            ("timings", ""),
            ("subject_name", "mascot"),
            ("motion_description", MOTION_SHORT),
            ("fps", 24.0),
            ("loop", False),
            ("output_dir", "from_mesh_to_motion"),
        ],
        links={
            "view_a": (32, 0, "IMAGE"),
            "view_b": (32, 1, "IMAGE"),
            "view_c": (32, 2, "IMAGE"),
            "sheet": (40, 0, "IMAGE"),
            "fps_in": (22, 2, "FLOAT"),
        },
        widget_links={"motion_description": (5, 1, "STRING")},
        optional=("sheet", "fps_in"),
        outputs=[("manifest_path", "STRING"), ("handoff_text", "STRING")],
        title="7 - keyframes.json + handoff.txt",
        size=(360, 320),
    ))

    return n


def concept_to_mesh():
    """An image to a textured, A-posed GLB -- the optional first leg."""
    return [
        Node(1, "LoadImage", (40, 40),
             widgets=[("image", "concept_front.png"), ("~upload", "image")],
             outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
             title="Character concept, front on, flat background",
             size=(300, 314)),
        Node(2, "MeshyImageToModelNode", (400, 40),
             widgets=[
                 ("model", "meshy-7"),
                 ("should_remesh", "true"),
                 ("should_remesh.topology", "quad"),
                 ("should_remesh.target_polycount", 25000),
                 ("symmetry_mode", "on"),
                 ("should_texture", "true"),
                 ("should_texture.enable_pbr", False),
                 ("should_texture.texture_prompt", ""),
                 ("should_texture.texture_resolution", "2k"),
                 ("pose_mode", "A-pose"),
                 ("seed", 7),
                 ("ultra_mode", False),
             ],
             links={"image": (1, 0, "IMAGE")},
             outputs=[("STRING", "STRING"), ("MESHY_TASK_ID", "MESHY_TASK_ID"),
                      ("FILE_3D_GLB", "FILE_3D_GLB"), ("FILE_3D_FBX", "FILE_3D_FBX")],
             title="Image to mesh, held in A-pose for rigging",
             size=(380, 380)),
        Node(3, "SaveGLB", (830, 40),
             widgets=[("filename_prefix", "fmm/character")],
             links={"mesh": (2, 2, "FILE_3D_GLB")},
             title="The mesh Blender renders and Astra rigs",
             size=(320, 90)),
    ]


WORKFLOWS = {
    "01_concept_to_mesh": concept_to_mesh,
    "02_triptych_to_keyframes": triptych_to_keyframes,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "api"), exist_ok=True)
    for name, factory in WORKFLOWS.items():
        nodes = factory()
        api = build_api_format(nodes)          # before links get link-ids attached
        save = build_save_format(nodes, name)

        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(save, fh, indent=2)
        with open(os.path.join(OUT, "api", name + ".api.json"), "w", encoding="utf-8") as fh:
            json.dump(api, fh, indent=2)
        print(name + ": " + str(len(nodes)) + " nodes, "
              + str(len(save["links"])) + " links")


if __name__ == "__main__":
    main()
