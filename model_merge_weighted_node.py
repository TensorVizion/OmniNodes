"""
TensorVizion ComfyUI Nodes
model_merge_weighted_node.py — Merges two MODEL checkpoints using one of three
strategies: weighted sum, sigmoid blend, or layer-selective merge.
"""

import torch


class ModelMergeWeightedNode:
    """
    Merges two diffusion models using a chosen blend strategy.

    Strategies
    ----------
    weighted_sum   : out = A * (1 - ratio) + B * ratio  (classic)
    sigmoid_blend  : applies a smooth sigmoid curve to the ratio across layers
    layer_select   : encoder layers from A, decoder layers from B (split at ratio)
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a":  ("MODEL",),
                "model_b":  ("MODEL",),
                "ratio":    ("FLOAT",  {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "strategy": (["weighted_sum", "sigmoid_blend", "layer_select"],),
            }
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("merged_model",)
    FUNCTION      = "merge_models"

    def merge_models(self, model_a, model_b, ratio, strategy):
        import comfy.model_management

        m = model_a.clone()
        kp = model_b.get_key_patches("diffusion_model.")

        keys = list(kp.keys())
        n    = len(keys)

        for i, key in enumerate(keys):
            if strategy == "weighted_sum":
                r = ratio

            elif strategy == "sigmoid_blend":
                # Smooth S-curve: layers near ratio boundary blend softly
                t = i / max(n - 1, 1)
                r = 1.0 / (1.0 + torch.exp(torch.tensor(-10.0 * (t - ratio))).item())

            elif strategy == "layer_select":
                # Layers before ratio*n come from A (r=0), rest from B (r=1)
                r = 0.0 if (i / max(n - 1, 1)) < ratio else 1.0

            else:
                r = ratio

            m.add_patches({key: kp[key]}, r, 1.0 - r)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "ModelMergeWeightedNode": ModelMergeWeightedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeWeightedNode": "Model Merge Weighted 🔀",
}
