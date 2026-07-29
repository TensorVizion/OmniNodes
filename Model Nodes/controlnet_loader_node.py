"""
TensorVizion ComfyUI Nodes
controlnet_loader_node.py — Loads a ControlNet model file, for
canny/depth/pose/lineart-guided generation workflows. A thin,
TensorVizion-branded wrapper around ComfyUI's own core ControlNetLoader —
delegated directly, not reimplemented, so any ControlNet format ComfyUI
itself supports (including newer union/multi-type ControlNets) works here
identically.
"""

from nodes import ControlNetLoader as _CoreControlNetLoader


class ControlNetLoaderNode:
    """
    Loads `control_net_name` from the `controlnet/` model folder using
    ComfyUI's own core ControlNet-loading logic. Output plugs directly
    into ComfyUI's Apply ControlNet node (or any custom equivalent) —
    this node only handles loading the model file itself, not applying it
    to a conditioning.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreControlNetLoader.INPUT_TYPES()

    RETURN_TYPES  = ("CONTROL_NET", "STRING")
    RETURN_NAMES  = ("control_net",  "summary")
    FUNCTION      = "run"

    def run(self, control_net_name):
        core = _CoreControlNetLoader()
        result = core.load_controlnet(control_net_name)
        control_net = result[0]
        summary = f"ControlNet loaded: {control_net_name}"
        return (control_net, summary)


NODE_CLASS_MAPPINGS = {
    "ControlNetLoaderNode": ControlNetLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetLoaderNode": "ControlNet Loader 🕹️",
}
