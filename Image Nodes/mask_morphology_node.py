"""
TensorVizion ComfyUI Nodes
mask_morphology_node.py — Grow, shrink, blur, feather, and invert a MASK.
The pack has rich latent-space masking (Latent Mask, Latent Blend) but no
plain IMAGE-space MASK utilities — this fills that gap. Common uses: grow
a face-detect mask before compositing so the edge doesn't hard-cut through
hair, or feather a hand-painted mask for a softer inpaint blend boundary.
"""

import torch
import torch.nn.functional as F


class MaskMorphologyNode:
    """
    operation:
      grow      — dilates the mask outward by `amount` pixels (max-pool based)
      shrink    — erodes the mask inward by `amount` pixels (min-pool based)
      feather   — gaussian-blurs the mask edge by `amount` pixels, softening
                    a hard boundary into a gradient without changing its
                    overall size the way grow/shrink do
      invert    — flips the mask (1 - mask); `amount` is ignored in this mode

    `amount` is in pixels for grow/shrink/feather. Grow and shrink use
    successive small-kernel passes rather than one huge kernel — this
    keeps the operation reasonably fast for large `amount` values and
    avoids the exact-shape distortion a single giant kernel can produce
    at image borders.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (["grow", "shrink", "feather", "invert"],),
                "amount": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "summary")
    FUNCTION = "process"

    # ------------------------------------------------------------------
    def _grow(self, mask_bhw, amount):
        if amount <= 0:
            return mask_bhw
        x = mask_bhw.unsqueeze(1)  # (B,1,H,W)
        # Repeated 3x3 max-pool passes approximate a larger dilation kernel
        # while staying cheap; `amount` passes ~= amount-pixel dilation radius.
        for _ in range(amount):
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x.squeeze(1)

    def _shrink(self, mask_bhw, amount):
        if amount <= 0:
            return mask_bhw
        x = mask_bhw.unsqueeze(1)
        for _ in range(amount):
            x = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        return x.squeeze(1)

    def _feather(self, mask_bhw, amount):
        if amount <= 0:
            return mask_bhw
        x = mask_bhw.unsqueeze(1)
        radius = max(1, amount | 1)  # ensure odd kernel size
        sigma = max(radius / 3.0, 0.5)
        coords = torch.arange(radius, dtype=torch.float32, device=x.device) - radius // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kx = kernel_1d.view(1, 1, 1, radius)
        ky = kernel_1d.view(1, 1, radius, 1)
        pad = radius // 2

        x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
        x = F.conv2d(x, kx)
        x = F.pad(x, (0, 0, pad, pad), mode="reflect")
        x = F.conv2d(x, ky)
        return x.squeeze(1)

    def process(self, mask, operation, amount):
        mask = mask.float().clamp(0.0, 1.0)

        if operation == "grow":
            out = self._grow(mask, amount)
            summary = f"Grew mask by ~{amount}px"
        elif operation == "shrink":
            out = self._shrink(mask, amount)
            summary = f"Shrank mask by ~{amount}px"
        elif operation == "feather":
            out = self._feather(mask, amount)
            summary = f"Feathered mask edge by ~{amount}px"
        else:
            out = 1.0 - mask
            summary = "Inverted mask"

        return (out.clamp(0.0, 1.0), summary)


NODE_CLASS_MAPPINGS = {
    "MaskMorphologyNode": MaskMorphologyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMorphologyNode": "Mask Morphology 🩹",
}
