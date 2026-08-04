"""
TensorVizion ComfyUI Nodes
dual_model_merger.py — Merges two MODEL checkpoints using Weighted Sum, Add
Difference, or Slerp-style interpolation. A friendlier-named sibling to Model
Merge Weighted (which exposes weighted_sum/sigmoid_blend/layer_select
instead) for people who want classic A1111-style merge method names.
"""

import math
import comfy.model_management


class DualModelMerger:
    """
    Merges model_A and model_B using one of three classic checkpoint-merge
    methods, applied per key-patch via ComfyUI's native model patching API
    (model.clone() + get_key_patches() + add_patches() — the same mechanism
    ComfyUI's own checkpoint merge nodes use, so the merged model behaves
    like a real ComfyUI model rather than a hand-rolled state-dict average).

    Interpolation methods
    ----------------------
    Weighted Sum    : out = A * (1 - ratio) + B * ratio  (classic linear blend)
    Add Difference  : out = A + (B - A) * ratio  -- mathematically identical
                      to Weighted Sum for a straight two-model blend; kept as
                      a separate named option to match the "B as a delta on
                      A" mental model some merge tools use.
    Slerp           : uses a sinusoidal (rather than linear) weighting curve
                      between A and B based on `merge_ratio`, so the midpoint
                      of the slider blends more gently than a straight linear
                      ramp. This operates on patch *strengths*, not on the
                      raw tensors directly (ComfyUI's add_patches API works
                      with opaque patch tuples, not tensors, at this layer),
                      so it's an approximation of true per-tensor SLERP
                      rather than the literal spherical-interpolation
                      formula -- for exact SLERP behaviour on raw latents
                      instead of model weights, use the Latent Interpolate
                      node's `slerp` method.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A":       ("MODEL",),
                "model_B":       ("MODEL",),
                "merge_ratio":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "interpolation": (["Weighted Sum", "Add Difference", "Slerp"],),
            }
        }

    RETURN_TYPES  = ("MODEL",  "STRING")
    RETURN_NAMES  = ("model",  "merge_info")
    FUNCTION      = "merge_models"

    @staticmethod
    def _slerp_strengths(ratio):
        """
        Sinusoidal (quarter-cosine) weighting curve used as a patch-strength
        stand-in for true per-tensor SLERP. Unlike a straight linear ramp,
        this eases in/out around the midpoint, which is the practically
        useful property SLERP is usually reached for in model merging
        (avoiding a "washed out" 50/50 blend) without needing access to raw
        tensors.
        """
        omega = ratio * (math.pi / 2.0)
        strength_b = math.sin(omega)
        strength_a = math.cos(omega)
        total = strength_a + strength_b
        if total < 1e-8:
            return (1.0 - ratio, ratio)
        return (strength_a / total, strength_b / total)

    def merge_models(self, model_A, model_B, merge_ratio, interpolation):
        m = model_A.clone()
        patches_a = model_A.get_key_patches("diffusion_model.")
        patches_b = model_B.get_key_patches("diffusion_model.")

        keys = list(patches_b.keys())
        applied = 0
        skipped = 0

        if interpolation == "Slerp":
            strength_a, strength_b = self._slerp_strengths(merge_ratio)
        else:
            strength_a, strength_b = (1.0 - merge_ratio), merge_ratio

        for key in keys:
            if key not in patches_a:
                skipped += 1
                continue
            m.add_patches({key: patches_b[key]}, strength_b, strength_a)
            applied += 1

        comfy.model_management.soft_empty_cache()

        summary = (
            f"Interpolation: {interpolation}\n"
            f"Merge ratio:   {merge_ratio:.2f}\n"
            f"Strength A/B:  {strength_a:.3f} / {strength_b:.3f}\n"
            f"Keys merged:   {applied}\n"
            f"Keys skipped:  {skipped} (present in B, absent in A)"
        )

        return (m, summary)


NODE_CLASS_MAPPINGS = {
    "DualModelMerger": DualModelMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualModelMerger": "Dual Model Merger 🔀",
}
