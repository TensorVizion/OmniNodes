"""
TensorVizion ComfyUI Nodes
image_prompt_apply_node.py — Takes a reference IMAGE plus the CLIP_VISION
/ STYLE_MODEL pair from Model Nodes/image_prompt_loader_node.py and applies
it to a conditioning, so the resulting image is guided by the reference
image's style/content in addition to (or, at strength 1.0 with an empty
text prompt, instead of) the text prompt. This is the apply half of the
pack's first image-conditioning path — closes the loop opened by
image_prompt_loader_node.py. A thin, TensorVizion-branded wrapper around
ComfyUI's own core CLIPVisionEncode and StyleModelApply.
"""

from nodes import CLIPVisionEncode as _CoreCLIPVisionEncode
from nodes import StyleModelApply as _CoreStyleModelApply


class ImagePromptApplyNode:
    """
    Encodes `reference_image` with `clip_vision`, then applies it through
    `style_model` onto `conditioning` at `strength`. Higher strength
    pulls the result further toward the reference image's style/content
    and further from the plain text prompt.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip_vision": ("CLIP_VISION",),
                "style_model": ("STYLE_MODEL",),
                "reference_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "summary")
    FUNCTION = "run"

    def run(self, conditioning, clip_vision, style_model, reference_image, strength):
        vision_encoding = _CoreCLIPVisionEncode().encode(clip_vision, reference_image)[0]

        applier = _CoreStyleModelApply()
        apply_fn = getattr(applier, applier.FUNCTION)
        try:
            result = apply_fn(conditioning, style_model, vision_encoding, strength)
        except TypeError:
            # Older core signature without a strength parameter.
            result = apply_fn(conditioning, style_model, vision_encoding)

        summary = f"Image-prompt conditioning applied at strength={strength:.2f}"
        return (result[0], summary)


NODE_CLASS_MAPPINGS = {
    "ImagePromptApplyNode": ImagePromptApplyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePromptApplyNode": "Image Prompt Apply 🖼️ (TensorVizion)",
}
