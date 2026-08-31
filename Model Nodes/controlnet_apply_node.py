"""
TensorVizion ComfyUI Nodes
controlnet_apply_node.py — Applies a loaded ControlNet to a conditioning,
using an image built by controlnet_preprocessor_node.py (or any IMAGE
conditioning source). Closes the loop between Model Nodes/
controlnet_loader_node.py and Model Nodes/controlnet_preprocessor_node.py,
neither of which push their output onto a conditioning by themselves — this
node is that missing "apply" step, so a full ControlNet workflow can be
built from OmniNodes alone. A thin, TensorVizion-branded wrapper around
ComfyUI's own core ControlNetApplyAdvanced, so behaviour (including
start/end percent windowing and optional VAE for latent-input ControlNets)
matches the stock node exactly.
"""

from nodes import ControlNetApplyAdvanced as _CoreControlNetApplyAdvanced


class ControlNetApplyNode:
    """
    Applies `control_net` to both `positive` and `negative` conditioning
    using `image` as the control/hint image, scaled by `strength` and
    windowed to only affect sampling between `start_percent` and
    `end_percent` of the denoise schedule. `vae` is optional and only
    needed for ControlNets that expect a latent-space hint image rather
    than a pixel-space one.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreControlNetApplyAdvanced.INPUT_TYPES()

    RETURN_TYPES  = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES  = ("positive",     "negative",     "summary")
    FUNCTION      = "run"

    def run(self, positive, negative, control_net, image, strength,
             start_percent, end_percent, vae=None):
        core = _CoreControlNetApplyAdvanced()
        result = core.apply_controlnet(
            positive, negative, control_net, image, strength,
            start_percent, end_percent, vae=vae,
        )
        pos, neg = result[0], result[1]
        summary = (
            f"ControlNet applied: strength={strength:.2f}, "
            f"window=[{start_percent:.2f}, {end_percent:.2f}]"
        )
        return (pos, neg, summary)


NODE_CLASS_MAPPINGS = {
    "ControlNetApplyNode": ControlNetApplyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetApplyNode": "ControlNet Apply (Advanced) 🕹️",
}
