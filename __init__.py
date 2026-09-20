"""from-mesh-to-motion -- ComfyUI custom node pack.

Cloning this repository into ComfyUI/custom_nodes/ registers the six nodes that
turn a rest-pose mesh into a keyframe sheet a rigging tool can animate from.
"""

from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
