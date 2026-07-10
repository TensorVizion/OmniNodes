"""
TensorVizion ComfyUI Nodes
image_vignette_glow_node.py — Radial vignette darkening combined with a
bloom-style glow around bright areas, for IMAGE tensors. Pure PyTorch.
"""

import torch
import torch.nn.functional as F


class ImageVignetteGlowNode:
    """
    Two independent effects in one node, applied in this order:

      1. Glow/bloom — pixels above glow_threshold are extracted, blurred,
         and added back on top of the original at glow_strength.
      2. Vignette — radial darkening from the center outward. vignette_radius
         controls how far out the falloff starts (1.0 ~ reaches the corners
         of the frame); vignette_amount controls how dark the corners get.

    Set either amount to 0 to disable that half of the effect.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vignette_amount": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vignette_radius": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05}),
                "glow_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "glow_radius": ("INT", {"default": 15, "min": 1, "max": 63, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    # ------------------------------------------------------------------
    def _gaussian_blur(self, chw, radius):
        radius = max(1, radius | 1)
        sigma = max(radius / 3.0, 0.5)
        coords = torch.arange(radius, dtype=torch.float32, device=chw.device) - radius // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        C = chw.shape[1]
        kx = kernel_1d.view(1, 1, 1, radius).expand(C, 1, 1, radius)
        ky = kernel_1d.view(1, 1, radius, 1).expand(C, 1, radius, 1)

        pad = radius // 2
        out = F.pad(chw, (pad, pad, 0, 0), mode="reflect")
        out = F.conv2d(out, kx, groups=C)
        out = F.pad(out, (0, 0, pad, pad), mode="reflect")
        out = F.conv2d(out, ky, groups=C)
        return out

    # ------------------------------------------------------------------
    def apply(self, image, vignette_amount, vignette_radius, glow_threshold, glow_strength, glow_radius):
        img = image.clone().float()  # (B, H, W, C)
        B, H, W, C = img.shape

        out = img

        # --- glow ---
        if glow_strength > 0.0:
            if C >= 3:
                weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device)
                luminance = (img[..., :3] * weights).sum(dim=-1, keepdim=True)
            else:
                luminance = img.mean(dim=-1, keepdim=True)

            bright_mask = ((luminance - glow_threshold) / max(1e-4, 1.0 - glow_threshold)).clamp(0.0, 1.0)
            bright_pass = img * bright_mask
            chw = bright_pass.permute(0, 3, 1, 2)
            blurred = self._gaussian_blur(chw, glow_radius).permute(0, 2, 3, 1)
            out = out + blurred * glow_strength

        # --- vignette ---
        if vignette_amount > 0.0:
            ys = torch.linspace(-1.0, 1.0, H, device=img.device).view(H, 1).expand(H, W)
            xs = torch.linspace(-1.0, 1.0, W, device=img.device).view(1, W).expand(H, W)
            dist = torch.sqrt((xs / vignette_radius) ** 2 + (ys / vignette_radius) ** 2)
            vignette_mask = 1.0 - vignette_amount * dist.clamp(0.0, 1.0) ** 2
            vignette_mask = vignette_mask.view(1, H, W, 1)
            out = out * vignette_mask

        out = out.clamp(0.0, 1.0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ImageVignetteGlowNode": ImageVignetteGlowNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageVignetteGlowNode": "Image Vignette & Glow ✨",
}
