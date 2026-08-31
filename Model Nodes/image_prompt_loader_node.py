"""
TensorVizion ComfyUI Nodes
image_prompt_loader_node.py — Loads the two model files an image-prompting
("style transfer from a reference image") workflow needs in one node: a
CLIP vision encoder and a style model. The pack had no image-conditioning
path at all before this — everything upstream was text-prompt only. Pairs
with Model Nodes/image_prompt_apply_node.py, which takes this node's
outputs plus a reference image and actually applies the style. A thin,
TensorVizion-branded wrapper around ComfyUI's own core CLIPVisionLoader
and StyleModelLoader, loaded together since they're always used as a pair.
"""

from nodes import CLIPVisionLoader as _CoreCLIPVisionLoader
from nodes import StyleModelLoader as _CoreStyleModelLoader


class ImagePromptLoaderNode:
    """
    Loads `clip_vision_name` from `clip_vision/` and `style_model_name`
    from `style_models/` in one call. Both outputs feed directly into
    Model Nodes/image_prompt_apply_node.py alongside a reference IMAGE.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        clip_vision_inputs = _CoreCLIPVisionLoader.INPUT_TYPES()["required"]
        style_model_inputs = _CoreStyleModelLoader.INPUT_TYPES()["required"]
        return {
            "required": {
                "clip_vision_name": clip_vision_inputs["clip_name"],
                "style_model_name": style_model_inputs["style_model_name"],
            }
        }

    RETURN_TYPES  = ("CLIP_VISION", "STYLE_MODEL", "STRING")
    RETURN_NAMES  = ("clip_vision",  "style_model", "summary")
    FUNCTION      = "run"

    def run(self, clip_vision_name, style_model_name):
        clip_vision = _CoreCLIPVisionLoader().load_clip(clip_vision_name)[0]
        style_model = _CoreStyleModelLoader().load_style_model(style_model_name)[0]
        summary = f"CLIP vision: {clip_vision_name}  |  Style model: {style_model_name}"
        return (clip_vision, style_model, summary)


NODE_CLASS_MAPPINGS = {
    "ImagePromptLoaderNode": ImagePromptLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePromptLoaderNode": "Image Prompt Loader 🖼️ (TensorVizion)",
}
