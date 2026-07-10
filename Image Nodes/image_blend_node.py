"""
TensorVizion ComfyUI Nodes
image_blend_node.py — Blends two IMAGE tensors using one of several blend
modes, with optional per-channel strength control and a MASK for spatial
blending. Pixel-space counterpart to the Latent Blend node.
"""

import torch
import torch.nn.functional as F


class ImageBlendNode:
    """
    Blends two images (A and B) using a chosen blend mode and ratio.

    Blend modes
    -----------
    lerp         : standard linear interpolation  out = A*(1-r) + B*r
    add          : out = A + B*ratio*strength
    subtract     : out = A - B*ratio*strength
    multiply     : classic multiply, blended in by ratio
    screen       : out = 1 - (1-A)*(1-B), blended in by ratio
    overlay      : photoshop-style overlay
    difference   : out = |A - B|, blended in by ratio
    hardlight    : hardlight composite
    spatial_mask : use a MASK to blend spatially (A outside, B inside)

    If image_a and image_b differ in resolution, B is resized to match A.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend_mode": (["lerp", "add", "subtract", "multiply", "screen",
                                 "overlay", "difference", "hardlight", "spatial_mask"],),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "clamp_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "blend_info")
    FUNCTION = "blend_images"

    # ------------------------------------------------------------------
    def _resize_to(self, img, h, w):
        """img: (B,H,W,C) -> resized (B,h,w,C)."""
        chw = img.permute(0, 3, 1, 2)
        chw = F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)
        return chw.permute(0, 2, 3, 1)

    # ------------------------------------------------------------------
    def blend_images(self, image_a, image_b, blend_mode, ratio, strength, clamp_output, mask=None):
        a = image_a.clone()
        b = image_b.clone()

        if a.shape[1:3] != b.shape[1:3]:
            b = self._resize_to(b, a.shape[1], a.shape[2])
        if b.shape[0] != a.shape[0]:
            # Broadcast a single-image batch to match the other side
            if b.shape[0] == 1:
                b = b.expand(a.shape[0], -1, -1, -1)
            elif a.shape[0] == 1:
                a = a.expand(b.shape[0], -1, -1, -1)

        B, H, W, C = a.shape

        if blend_mode == "lerp":
            out = torch.lerp(a, b, ratio) * strength

        elif blend_mode == "add":
            out = a + b * ratio * strength

        elif blend_mode == "subtract":
            out = a - b * ratio * strength

        elif blend_mode == "multiply":
            blended = a * b
            out = torch.lerp(a, blended, ratio) * strength

        elif blend_mode == "screen":
            blended = 1.0 - (1.0 - a) * (1.0 - b)
            out = torch.lerp(a, blended, ratio) * strength

        elif blend_mode == "overlay":
            low = 2.0 * a * b
            high = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
            blended = torch.where(a < 0.5, low, high)
            out = torch.lerp(a, blended, ratio) * strength

        elif blend_mode == "difference":
            blended = torch.abs(a - b)
            out = torch.lerp(a, blended, ratio) * strength

        elif blend_mode == "hardlight":
            low = 2.0 * a * b
            high = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
            blended = torch.where(b < 0.5, low, high)
            out = torch.lerp(a, blended, ratio) * strength

        elif blend_mode == "spatial_mask":
            if mask is not None:
                m = mask.float()
                if m.ndim == 2:
                    m = m.unsqueeze(0)
                m = m.unsqueeze(-1)  # (B or 1, H, W, 1)
                if m.shape[1:3] != (H, W):
                    m = self._resize_to(m.expand(-1, -1, -1, 1), H, W)
                if m.shape[0] != B:
                    m = m.expand(B, -1, -1, -1)
                m = m.expand(-1, -1, -1, C).to(a.device)
                out = a * (1.0 - m) + b * m * strength
            else:
                out = torch.lerp(a, b, ratio) * strength
        else:
            out = torch.lerp(a, b, ratio)

        if clamp_output:
            out = out.clamp(0.0, 1.0)

        blend_info = (
            f"Blend mode:  {blend_mode}\n"
            f"Ratio:       {ratio:.2f}\n"
            f"Strength:    {strength:.2f}\n"
            f"Shape A:     {list(image_a.shape)}\n"
            f"Shape B:     {list(image_b.shape)}\n"
            f"Clamped:     {clamp_output}\n"
            f"Mask used:   {mask is not None and blend_mode == 'spatial_mask'}"
        )

        return (out, blend_info)


NODE_CLASS_MAPPINGS = {
    "ImageBlendNode": ImageBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendNode": "Image Blend 🖌️",
}
