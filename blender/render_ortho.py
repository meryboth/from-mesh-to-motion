"""Headless orthographic turnaround renderer.

Renders a rest-pose mesh from N yaw angles with a *single* camera and a *single*
ortho scale, so every view shares one pixel scale and one pivot. That shared
framing is the whole point: downstream, a pose read off the front view can be
trusted to line up with the same pose read off the back view.

The object rotates; the camera never moves. That is what guarantees the framing
is identical across views (moving the camera re-derives framing per view and
drifts by a pixel or two, which is enough to break a contact sheet).

Lights are fixed in world space, which -- because the camera is also fixed --
means every view is lit from the same direction *relative to the viewer*. A back
view lit from behind would read as a different character to an image model.

Usage:
  blender --background --python render_ortho.py -- \
      --mesh character.glb --out out/01_anchors --size 768 \
      --views front,side,back --passes beauty,clay
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

# Yaw applied to the object, in degrees, for each named view. The camera sits on
# -Y looking toward +Y, so yaw 0 shows whatever face of the object points at -Y.
VIEW_YAW = {
    "front": 0.0,
    "right": 90.0,
    "side": 90.0,   # alias: "side" means the right side
    "back": 180.0,
    "left": 270.0,
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="render_ortho")
    p.add_argument("--mesh", required=True, help="Path to .glb/.gltf/.fbx/.obj")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--size", type=int, default=768, help="Square render size in px")
    p.add_argument("--views", default="front,side,back",
                   help="Comma-separated view names: front,back,left,right,side")
    p.add_argument("--passes", default="beauty,clay",
                   help="Comma-separated: beauty (textured), clay (matte override)")
    p.add_argument("--margin", type=float, default=0.12,
                   help="Fraction of empty space around the subject")
    p.add_argument("--yaw-offset", type=float, default=0.0,
                   help="Degrees added to every view, to correct a mesh whose "
                        "front does not face -Y on import")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--engine", default="auto", help="auto | EEVEE | CYCLES")
    p.add_argument("--clay-color", default="0.62,0.62,0.64")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        # The operator moved between Blender versions; try both spellings.
        try:
            bpy.ops.import_scene.gltf(filepath=path)
        except AttributeError:
            bpy.ops.wm.gltf_import(filepath=path)
    elif ext == ".fbx":
        try:
            bpy.ops.import_scene.fbx(filepath=path)
        except AttributeError:
            bpy.ops.wm.fbx_import(filepath=path)
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise SystemExit("Unsupported mesh format: " + ext)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("No mesh objects found in the imported file")
    return meshes


def world_bbox(objs):
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for o in objs:
        for corner in o.bound_box:
            p = o.matrix_world @ Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], p[i])
                hi[i] = max(hi[i], p[i])
    return lo, hi


def normalize(objs):
    """Centre the subject on the Z axis with its feet at z=0.

    Everything is parented to one empty at that spot, so a single rotation on
    the empty spins the whole character about its own vertical axis. Without a
    shared pivot each view would orbit a slightly different centre.
    """
    lo, hi = world_bbox(objs)
    offset = Vector(((lo.x + hi.x) / 2.0, (lo.y + hi.y) / 2.0, lo.z))

    roots = [o for o in objs if o.parent is None]
    for o in roots:
        o.matrix_world.translation -= offset
    bpy.context.view_layer.update()

    pivot = bpy.data.objects.new("pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = (0.0, 0.0, 0.0)

    for o in roots:
        o.parent = pivot
        o.matrix_parent_inverse = pivot.matrix_world.inverted()

    bpy.context.view_layer.update()
    lo2, hi2 = world_bbox(objs)
    return pivot, lo2, hi2


def setup_camera(lo, hi, margin):
    size = hi - lo
    height = size.z
    # One ortho scale for every view: the subject must fit at any yaw, so take
    # the largest horizontal extent, not the front-on one.
    horizontal = max(size.x, size.y)
    extent = max(horizontal, height) * (1.0 + 2.0 * margin)

    cam_data = bpy.data.cameras.new("ortho_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent
    cam_data.clip_start = 0.01
    cam_data.clip_end = max(1000.0, extent * 20.0)

    cam = bpy.data.objects.new("ortho_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    # Sit on -Y, look along +Y, dead level: a true elevation view.
    cam.location = (0.0, -extent * 4.0, height / 2.0)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam, extent


def setup_lights(extent):
    """Three-point rig fixed in world space, i.e. fixed relative to the camera."""
    specs = [
        ("key",  (-1.0, -1.6, 1.4), 4.0),
        ("fill", (1.6, -1.2, 0.3), 1.6),
        ("rim",  (0.4, 1.5, 1.8), 2.6),
    ]
    for name, direction, energy in specs:
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = energy
        data.angle = math.radians(25.0)
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        d = Vector(direction).normalized()
        obj.location = d * extent * 5.0
        obj.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.35
    bpy.context.scene.world = world


def pick_engine(requested):
    enum = list(bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys())
    if requested.upper() == "CYCLES" and "CYCLES" in enum:
        return "CYCLES"
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in enum:
            return candidate
    return enum[0]


def setup_render(size, samples, engine, out_dir):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "Standard"

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        try:
            scene.eevee.taa_render_samples = samples
        except AttributeError:
            pass
    os.makedirs(out_dir, exist_ok=True)


def make_clay_material(rgb):
    mat = bpy.data.materials.new("clay_override")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.75
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    return mat


def main():
    args = parse_args()
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    for v in views:
        if v not in VIEW_YAW:
            raise SystemExit("Unknown view: " + v + ". Known: " + ", ".join(sorted(VIEW_YAW)))

    reset_scene()
    meshes = import_mesh(os.path.abspath(args.mesh))
    pivot, lo, hi = normalize(meshes)
    cam, extent = setup_camera(lo, hi, args.margin)
    setup_lights(extent)

    engine = pick_engine(args.engine)
    out_dir = os.path.abspath(args.out)
    setup_render(args.size, args.samples, engine, out_dir)

    clay = make_clay_material(tuple(float(c) for c in args.clay_color.split(",")))
    view_layer = bpy.context.scene.view_layers[0]

    rendered = []
    for pass_name in passes:
        view_layer.material_override = clay if pass_name == "clay" else None
        for view in views:
            yaw = VIEW_YAW[view] + args.yaw_offset
            pivot.rotation_euler = (0.0, 0.0, math.radians(yaw))
            bpy.context.view_layer.update()
            path = os.path.join(out_dir, pass_name + "_" + view + ".png")
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            rendered.append({"pass": pass_name, "view": view, "yaw": yaw, "file": path})
            print("[render_ortho] " + pass_name + "/" + view + " -> " + path)

    meta = {
        "mesh": os.path.abspath(args.mesh),
        "engine": engine,
        "size": args.size,
        "ortho_scale": extent,
        "yaw_offset": args.yaw_offset,
        "bbox_min": list(lo),
        "bbox_max": list(hi),
        "subject_height": hi.z - lo.z,
        "views": views,
        "passes": passes,
        "renders": rendered,
    }
    meta_path = os.path.join(out_dir, "anchors.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("[render_ortho] wrote " + meta_path)


if __name__ == "__main__":
    main()
