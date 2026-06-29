"""
TensorVizion ComfyUI Nodes
latent_mask_node.py — Applies a spatial mask to a LATENT tensor, zeroing or
blending selected regions. Supports geometric shapes and an optional MASK input.
"""

import torch
import numpy as np


class LatentMaskNode:
    """
    Masks a region of a latent tensor using either a geometric primitive
    (rectangle, ellipse, gradient) or a passed-in MASK tensor.

    blend_mode controls what happens inside the masked region:
      zero     — set to 0 (full erase)
      noise    — fill with gaussian noise
      clamp    — clamp values to [-clamp_val, clamp_val]
      invert   — multiply by -1
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":       ("LATENT",),
                "mask_shape":   (["rectangle", "ellipse", "gradient_h", "gradient_v", "external"],),
                "blend_mode":   (["zero", "noise", "clamp", "invert"],),
                "x":            ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y":            ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width":        ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "height":       ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather":      ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5,  "step": 0.01}),
                "clamp_value":  ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0, "step": 0.1}),
                "invert_mask":  ("BOOLEAN", {"default": False}),
                "seed":         ("INT",   {"default": 0,    "min": 0,   "max": 2**32 - 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES  = ("LATENT", "MASK")
    RETURN_NAMES  = ("latent", "mask_out")
    FUNCTION      = "apply_mask"

    # ------------------------------------------------------------------
    def _build_shape_mask(self, H, W, shape, x, y, w, h, feather):
        """Returns a float32 (H, W) mask in [0, 1]; 1 = inside masked region."""
        mask = np.zeros((H, W), dtype=np.float32)

        x0 = int(x * W);  y0 = int(y * H)
        x1 = int((x + w) * W); y1 = int((y + h) * H)
        cx = (x0 + x1) / 2.0;  cy = (y0 + y1) / 2.0
        rx = max(1, (x1 - x0) / 2.0); ry = max(1, (y1 - y0) / 2.0)

        ys, xs = np.mgrid[0:H, 0:W]

        if shape == "rectangle":
            inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            mask[inside] = 1.0
            if feather > 0:
                # Distance-based soft edge
                dx = np.maximum(0, np.maximum(x0 - xs, xs - (x1 - 1))) / (W * feather + 1e-8)
                dy = np.maximum(0, np.maximum(y0 - ys, ys - (y1 - 1))) / (H * feather + 1e-8)
                dist = np.sqrt(dx**2 + dy**2)
                mask = np.clip(1.0 - dist, 0.0, 1.0)

        elif shape == "ellipse":
            dist = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
            if feather > 0:
                outer = 1.0 + feather
                mask  = np.clip((outer - dist) / feather, 0.0, 1.0).astype(np.float32)
            else:
                mask[dist <= 1.0] = 1.0

        elif shape == "gradient_h":
            grad  = np.linspace(0, 1, W, dtype=np.float32)
            mask  = np.tile(grad, (H, 1))
            # Restrict to y band
            mask[:y0, :] = 0.0
            mask[y1:, :] = 0.0

        elif shape == "gradient_v":
            grad = np.linspace(0, 1, H, dtype=np.float32)[:, np.newaxis]
            mask = np.tile(grad, (1, W))
            mask[:, :x0] = 0.0
            mask[:, x1:] = 0.0

        return mask.astype(np.float32)

    # ------------------------------------------------------------------
    def apply_mask(
        self,
        latent, mask_shape, blend_mode,
        x, y, width, height, feather,
        clamp_value, invert_mask, seed,
        mask=None,
    ):
        samples = latent["samples"].clone()   # (B, C, H, W)
        B, C, H, W = samples.shape
        rng = np.random.default_rng(seed)

        if mask_shape == "external" and mask is not None:
            # Resize external mask to latent spatial dims
            m = mask.float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
            mask_np = m.cpu().numpy()
        else:
            mask_np = self._build_shape_mask(H, W, mask_shape, x, y, width, height, feather)

        if invert_mask:
            mask_np = 1.0 - mask_np

        mask_t = torch.from_numpy(mask_np).float().to(samples.device)  # (H, W)

        for b in range(B):
            for c in range(C):
                s = samples[b, c]

                if blend_mode == "zero":
                    fill = torch.zeros_like(s)
                elif blend_mode == "noise":
                    fill = torch.from_numpy(
                        rng.standard_normal((H, W)).astype(np.float32)
                    ).to(s.device)
                elif blend_mode == "clamp":
                    fill = s.clamp(-clamp_value, clamp_value)
                elif blend_mode == "invert":
                    fill = -s
                else:
                    fill = torch.zeros_like(s)

                samples[b, c] = s * (1.0 - mask_t) + fill * mask_t

        mask_out = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, H, W)
        return ({"samples": samples}, mask_out)


NODE_CLASS_MAPPINGS = {
    "LatentMaskNode": LatentMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentMaskNode": "Latent Mask 🎭",
}
