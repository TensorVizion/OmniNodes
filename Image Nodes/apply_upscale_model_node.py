"""
TensorVizion ComfyUI Nodes
apply_upscale_model_node.py — Runs an UPSCALE_MODEL (loaded by Model
Nodes/upscale_model_loader_node.py) over an image, tiling internally to
keep VRAM bounded on large images. That loader node only reads the model
file from disk; this node is the missing execution step, so an ESRGAN-
family upscale pass can be built from OmniNodes alone. A thin,
TensorVizion-branded wrapper around ComfyUI's own core ImageUpscaleWithModel,
with an added post-resize so output can optionally be clamped to a target
long-edge size instead of always returning the model's native scale factor.
"""

from nodes import ImageUpscaleWithModel as _CoreImageUpscaleWithModel
import torch
import torch.nn.functional as F


class ApplyUpscaleModelNode:
    """
    Runs `upscale_model` on `image`, then optionally resizes the result so
    its longer edge equals `max_long_edge` (0 disables resizing and returns
    the model's native output size, e.g. 4x for a 4x ESRGAN model).
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "max_long_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "summary")
    FUNCTION = "run"

    def run(self, upscale_model, image, max_long_edge):
        core = _CoreImageUpscaleWithModel()
        upscaled = core.upscale(upscale_model, image)[0]

        h, w = upscaled.shape[1], upscaled.shape[2]
        if max_long_edge and max(h, w) > max_long_edge:
            scale = max_long_edge / max(h, w)
            new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
            chw = upscaled.permute(0, 3, 1, 2)
            resized = F.interpolate(chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            upscaled = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)
            summary = f"Upscaled then resized to long edge {max_long_edge}px ({w}x{h} -> {new_w}x{new_h})"
        else:
            summary = f"Upscaled {w}x{h} -> {upscaled.shape[2]}x{upscaled.shape[1]}"

        return (upscaled, summary)


NODE_CLASS_MAPPINGS = {
    "ApplyUpscaleModelNode": ApplyUpscaleModelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyUpscaleModelNode": "Apply Upscale Model 🔭 (TensorVizion)",
}
