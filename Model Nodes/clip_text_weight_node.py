"""
TensorVizion ComfyUI Nodes
clip_text_weight_node.py — Encodes a prompt and scales the conditioning tensor
by a user-defined weight, enabling precise prompt strength control without
bracket syntax.
"""

import torch


class CLIPTextWeightNode:
    """
    Encodes a text prompt and multiplies the conditioning by a weight scalar.
    Provides a numeric slider alternative to (prompt:1.3) bracket syntax,
    and can also blend two conditionings at a chosen mix ratio.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":    ("CLIP",),
                "prompt":  ("STRING", {"multiline": True, "default": ""}),
                "weight":  ("FLOAT",  {"default": 1.0,  "min": -3.0, "max": 3.0,  "step": 0.05}),
            },
            "optional": {
                "conditioning_in": ("CONDITIONING",),
                "blend_ratio":     ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("CONDITIONING",)
    RETURN_NAMES  = ("conditioning",)
    FUNCTION      = "encode_with_weight"

    def encode_with_weight(self, clip, prompt, weight, conditioning_in=None, blend_ratio=0.5):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        # Scale by weight
        cond_weighted = cond * weight
        pooled_weighted = pooled * weight

        if conditioning_in is not None:
            # Blend with incoming conditioning
            cond_in   = conditioning_in[0][0]
            pooled_in = conditioning_in[0][1].get("pooled_output", pooled_weighted)

            # Match shapes if needed
            min_len = min(cond_in.shape[1], cond_weighted.shape[1])
            cond_in_t = cond_in[:, :min_len, :]
            cond_w_t  = cond_weighted[:, :min_len, :]

            cond_out   = cond_in_t * (1.0 - blend_ratio) + cond_w_t * blend_ratio
            pooled_out = pooled_in * (1.0 - blend_ratio) + pooled_weighted * blend_ratio
        else:
            cond_out   = cond_weighted
            pooled_out = pooled_weighted

        conditioning = [[cond_out, {"pooled_output": pooled_out}]]
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "CLIPTextWeightNode": CLIPTextWeightNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextWeightNode": "CLIP Text Weight ⚖️",
}
