"""
TensorVizion ComfyUI Nodes
image_sharpen_blur_node.py — Unsharp-mask sharpening and gaussian/box blur
for IMAGE tensors, in a single mode-selectable node. Pure PyTorch, no extra
dependencies (no PIL filters / no OpenCV).
"""

import torch
import torch.nn.functional as F


class ImageSharpenBlurNode:
    """
    mode:
      sharpen       — unsharp mask: out = image + (image - blurred) * strength
      gaussian_blur — gaussian blur, blended with the original by `strength`
                       (strength 1.0 = fully blurred, 0.0 = no change)
      box_blur      — box/average blur, same blend behaviour as gaussian_blur

    radius controls the blur kernel size for all three modes (bigger radius =
    softer blur / broader sharpening halo). Must be odd; even values are
    rounded up automatically.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["sharpen", "gaussian_blur", "box_blur"],),
                "radius": ("INT", {"default": 5, "min": 1, "max": 51, "step": 2}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

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

    def _box_blur(self, chw, radius):
        radius = max(1, radius | 1)
        C = chw.shape[1]
        kernel = torch.ones(C, 1, radius, radius, device=chw.device) / (radius * radius)
        pad = radius // 2
        out = F.pad(chw, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(out, kernel, groups=C)
        return out

    # ------------------------------------------------------------------
    def process(self, image, mode, radius, strength):
        chw = image.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        if mode == "sharpen":
            # strength can exceed 1.0 here -- that's a legitimately stronger unsharp mask.
            blurred = self._gaussian_blur(chw, radius)
            out = chw + (chw - blurred) * strength
        else:
            # For the two blur modes, strength is a 0-1 blend weight toward fully blurred;
            # values above 1 are clamped since "more than fully blurred" isn't meaningful.
            blend_weight = min(float(strength), 1.0)
            blurred = self._box_blur(chw, radius) if mode == "box_blur" else self._gaussian_blur(chw, radius)
            out = torch.lerp(chw, blurred, blend_weight)

        out = out.clamp(0.0, 1.0)
        return (out.permute(0, 2, 3, 1),)


NODE_CLASS_MAPPINGS = {
    "ImageSharpenBlurNode": ImageSharpenBlurNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSharpenBlurNode": "Image Sharpen & Blur 🔎",
}
