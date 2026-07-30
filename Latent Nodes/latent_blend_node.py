"""
TensorVizion ComfyUI Nodes
latent_blend_node.py — Blends two LATENT tensors using one of several
blend modes, with optional per-channel weight control and a MASK for
spatial blending.
"""

import torch
import numpy as np


class LatentBlendNode:
    """
    Blends two latent tensors (A and B) using a chosen blend mode and ratio.

    Blend modes
    -----------
    lerp         : standard linear interpolation  out = A*(1-r) + B*r
    add          : out = A + B*strength
    subtract     : out = A - B*strength
    multiply     : out = A * B  (normalised)
    screen       : out = 1 - (1-A)*(1-B)  (latent values normalised)
    overlay      : photoshop-style overlay in latent space
    difference   : out = |A - B|
    hardlight    : hardlight composite
    spatial_mask : use a MASK to blend spatially (A outside, B inside)
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a":   ("LATENT",),
                "latent_b":   ("LATENT",),
                "blend_mode": (["lerp", "add", "subtract", "multiply",
                                 "screen", "overlay", "difference",
                                 "hardlight", "spatial_mask"],),
                "ratio":      ("FLOAT", {"default": 0.5,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "strength":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 3.0,  "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("latent", "blend_info")
    FUNCTION      = "blend_latents"

    # ------------------------------------------------------------------
    def _norm01(self, t):
        """Normalise tensor to [0, 1] per-batch per-channel for photoshop modes."""
        mn = t.amin(dim=(-2, -1), keepdim=True)
        mx = t.amax(dim=(-2, -1), keepdim=True)
        return (t - mn) / (mx - mn + 1e-8), mn, mx

    def _denorm(self, t, mn, mx):
        return t * (mx - mn + 1e-8) + mn

    # ------------------------------------------------------------------
    def blend_latents(
        self,
        latent_a, latent_b,
        blend_mode, ratio, strength, normalize_output,
        mask=None,
    ):
        a = latent_a["samples"].clone()
        b = latent_b["samples"].clone()

        # Match spatial size — resize b to a if needed
        if a.shape != b.shape:
            b = torch.nn.functional.interpolate(
                b, size=(a.shape[2], a.shape[3]),
                mode="bilinear", align_corners=False
            )

        B, C, H, W = a.shape

        if blend_mode == "lerp":
            out = torch.lerp(a, b, ratio) * strength

        elif blend_mode == "add":
            out = a + b * ratio * strength

        elif blend_mode == "subtract":
            out = a - b * ratio * strength

        elif blend_mode == "multiply":
            an, mn_a, mx_a = self._norm01(a)
            bn, _, _       = self._norm01(b)
            out = self._denorm(an * bn, mn_a, mx_a) * strength

        elif blend_mode == "screen":
            an, mn_a, mx_a = self._norm01(a)
            bn, _, _       = self._norm01(b)
            blended        = 1.0 - (1.0 - an) * (1.0 - bn)
            out = self._denorm(torch.lerp(an, blended, ratio), mn_a, mx_a) * strength

        elif blend_mode == "overlay":
            an, mn_a, mx_a = self._norm01(a)
            bn, _, _       = self._norm01(b)
            low  = 2.0 * an * bn
            high = 1.0 - 2.0 * (1.0 - an) * (1.0 - bn)
            blended = torch.where(an < 0.5, low, high)
            out = self._denorm(torch.lerp(an, blended, ratio), mn_a, mx_a) * strength

        elif blend_mode == "difference":
            out = torch.abs(a - b) * strength

        elif blend_mode == "hardlight":
            an, mn_a, mx_a = self._norm01(a)
            bn, _, _       = self._norm01(b)
            low  = 2.0 * an * bn
            high = 1.0 - 2.0 * (1.0 - an) * (1.0 - bn)
            blended = torch.where(bn < 0.5, low, high)
            out = self._denorm(torch.lerp(an, blended, ratio), mn_a, mx_a) * strength

        elif blend_mode == "spatial_mask":
            if mask is not None:
                m = mask.float()
                if m.ndim == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.ndim == 3:
                    m = m.unsqueeze(0)
                m = torch.nn.functional.interpolate(
                    m, size=(H, W), mode="bilinear", align_corners=False
                )
                m = m.expand(B, C, H, W).to(a.device)
                out = a * (1.0 - m) + b * m * strength
            else:
                # Fall back to lerp if no mask provided
                out = torch.lerp(a, b, ratio) * strength
        else:
            out = torch.lerp(a, b, ratio)

        if normalize_output:
            peak = out.abs().amax()
            if peak > 1e-8:
                out = out / peak

        blend_info = (
            f"Blend mode:  {blend_mode}\n"
            f"Ratio:       {ratio:.2f}\n"
            f"Strength:    {strength:.2f}\n"
            f"Shape A:     {list(a.shape)}\n"
            f"Shape B:     {list(b.shape)}\n"
            f"Normalize:   {normalize_output}\n"
            f"Mask used:   {mask is not None and blend_mode == 'spatial_mask'}"
        )

        return ({"samples": out}, blend_info)


NODE_CLASS_MAPPINGS = {
    "LatentBlendNode": LatentBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentBlendNode": "Latent Blend 🌀",
}
