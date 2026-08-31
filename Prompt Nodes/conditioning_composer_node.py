"""
TensorVizion ComfyUI Nodes
conditioning_composer_node.py — One node covering the three ways two
conditionings get merged for regional / multi-concept prompting: combine
(both apply everywhere, averaged influence), concat (both apply
everywhere, concatenated tokens — stronger joint influence), and set_area
(second conditioning confined to a rectangular region of the latent).
Without this, building a regional-prompting workflow from OmniNodes alone
required dropping back to three separate stock nodes; this node is a
mode-switching wrapper delegating to ComfyUI's own core
ConditioningCombine / ConditioningConcat / ConditioningSetArea so each
mode's behaviour matches the stock nodes exactly.
"""

from nodes import (
    ConditioningCombine as _CoreConditioningCombine,
    ConditioningConcat as _CoreConditioningConcat,
    ConditioningSetArea as _CoreConditioningSetArea,
)


class ConditioningComposerNode:
    """
    mode:
      combine   — conditioning_1 and conditioning_2 both apply across the
                   whole image, averaged in influence. Use for blending
                   two style/subject prompts globally.
      concat    — conditioning_1 and conditioning_2 token sequences are
                   concatenated rather than averaged, giving both a
                   stronger combined pull than combine mode.
      set_area  — conditioning_2 is confined to a rectangular region
                   (width/height/x/y, in latent-space multiples of 8)
                   with `area_strength`, while conditioning_1 continues to
                   apply globally. Use for placing a second subject/detail
                   in a specific part of the frame.
    width/height/x/y and area_strength are ignored outside set_area mode.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "mode": (["combine", "concat", "set_area"],),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "area_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "summary")
    FUNCTION = "run"

    def run(self, conditioning_1, conditioning_2, mode, width, height, x, y, area_strength):
        if mode == "combine":
            result = _CoreConditioningCombine().combine(conditioning_1, conditioning_2)[0]
            summary = "Conditioning combined (averaged influence)"
        elif mode == "concat":
            result = _CoreConditioningConcat().concat(conditioning_1, conditioning_2)[0]
            summary = "Conditioning concatenated (stronger joint influence)"
        else:
            area_cond = _CoreConditioningSetArea().append(
                conditioning_2, width, height, x, y, area_strength
            )[0]
            result = _CoreConditioningCombine().combine(conditioning_1, area_cond)[0]
            summary = (
                f"conditioning_2 confined to {width}x{height} at ({x},{y}), "
                f"strength={area_strength:.2f}, layered over global conditioning_1"
            )

        return (result, summary)


NODE_CLASS_MAPPINGS = {
    "ConditioningComposerNode": ConditioningComposerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningComposerNode": "Conditioning Composer 🧩 (TensorVizion)",
}
