"""
TensorVizion ComfyUI Nodes
image_mask_composite_node.py — Applies a regional effect (darken, brighten,
desaturate, solid colour fill, invert, or blur) to a geometric or external
mask region of an IMAGE. Pixel-space counterpart to the Latent Mask node.
"""

import numpy as np
import torch
import torch.nn.functional as F


class ImageMaskCompositeNode:
    """
    Applies an effect to a masked region of a single image (not a two-image
    composite -- for blending two images together use Image Blend's
    spatial_mask mode instead).

    mask_shape: rectangle, ellipse, gradient_h, gradient_v, or external
    (connect a MASK socket and set mask_shape to "external" to use it).

    effect (applied only inside the mask):
      darken       — multiply by darken_amount
      brighten     — add brighten_amount
      desaturate   — convert to grayscale luminance
      solid_color  — fill with color_r/g/b
      invert       — photographic negative (1 - x)
      blur         — gaussian blur, radius set by blur_radius
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_shape": (["rectangle", "ellipse", "gradient_h", "gradient_v", "external"],),
                "effect": (["darken", "brighten", "desaturate", "solid_color", "invert", "blur"],),
                "x": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "darken_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brighten_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 9, "min": 1, "max": 51, "step": 2}),
                "color_r": ("INT", {"default": 255, "min": 0, "max": 255}),
                "color_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask_out")
    FUNCTION = "apply_effect"

    # ------------------------------------------------------------------
    def _build_shape_mask(self, H, W, shape, x, y, w, h, feather):
        """Returns a float32 (H, W) mask in [0, 1]; 1 = inside masked region."""
        mask = np.zeros((H, W), dtype=np.float32)

        x0 = int(x * W)
        y0 = int(y * H)
        x1 = int((x + w) * W)
        y1 = int((y + h) * H)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(1, (x1 - x0) / 2.0)
        ry = max(1, (y1 - y0) / 2.0)

        ys, xs = np.mgrid[0:H, 0:W]

        if shape == "rectangle":
            if feather > 0:
                dx = np.maximum(0, np.maximum(x0 - xs, xs - (x1 - 1))) / (W * feather + 1e-8)
                dy = np.maximum(0, np.maximum(y0 - ys, ys - (y1 - 1))) / (H * feather + 1e-8)
                dist = np.sqrt(dx ** 2 + dy ** 2)
                mask = np.clip(1.0 - dist, 0.0, 1.0)
            else:
                inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                mask[inside] = 1.0

        elif shape == "ellipse":
            dist = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
            if feather > 0:
                outer = 1.0 + feather
                mask = np.clip((outer - dist) / feather, 0.0, 1.0).astype(np.float32)
            else:
                mask[dist <= 1.0] = 1.0

        elif shape == "gradient_h":
            grad = np.linspace(0, 1, W, dtype=np.float32)
            mask = np.tile(grad, (H, 1))
            mask[:y0, :] = 0.0
            mask[y1:, :] = 0.0

        elif shape == "gradient_v":
            grad = np.linspace(0, 1, H, dtype=np.float32)[:, np.newaxis]
            mask = np.tile(grad, (1, W))
            mask[:, :x0] = 0.0
            mask[:, x1:] = 0.0

        return mask.astype(np.float32)

    def _gaussian_blur(self, img_bhwc, radius):
        """img_bhwc: (B,H,W,C) in [0,1]. radius controls kernel size/sigma."""
        radius = max(1, radius | 1)  # force odd
        sigma = max(radius / 3.0, 0.5)
        coords = torch.arange(radius, dtype=torch.float32) - radius // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        chw = img_bhwc.permute(0, 3, 1, 2)  # (B,C,H,W)
        C = chw.shape[1]
        kx = kernel_1d.view(1, 1, 1, radius).expand(C, 1, 1, radius)
        ky = kernel_1d.view(1, 1, radius, 1).expand(C, 1, radius, 1)

        pad = radius // 2
        blurred = F.pad(chw, (pad, pad, 0, 0), mode="reflect")
        blurred = F.conv2d(blurred, kx, groups=C)
        blurred = F.pad(blurred, (0, 0, pad, pad), mode="reflect")
        blurred = F.conv2d(blurred, ky, groups=C)

        return blurred.permute(0, 2, 3, 1)

    # ------------------------------------------------------------------
    def apply_effect(
        self, image, mask_shape, effect,
        x, y, width, height, feather, invert_mask,
        darken_amount, brighten_amount, blur_radius,
        color_r, color_g, color_b,
        mask=None,
    ):
        img = image.clone()  # (B, H, W, C)
        B, H, W, C = img.shape

        if mask_shape == "external" and mask is not None:
            m = mask.float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            mask_np = m[0].cpu().numpy()
        else:
            mask_np = self._build_shape_mask(H, W, mask_shape, x, y, width, height, feather)

        if invert_mask:
            mask_np = 1.0 - mask_np

        mask_t = torch.from_numpy(mask_np).float().to(img.device)  # (H, W)
        mask_bhwc = mask_t.view(1, H, W, 1).expand(B, H, W, C)

        if effect == "darken":
            fill = img * (1.0 - darken_amount)
        elif effect == "brighten":
            fill = (img + brighten_amount).clamp(0.0, 1.0)
        elif effect == "desaturate":
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device)
            if C >= 3:
                lum = (img[..., :3] * weights).sum(dim=-1, keepdim=True).expand(-1, -1, -1, C)
            else:
                lum = img.mean(dim=-1, keepdim=True).expand(-1, -1, -1, C)
            fill = lum
        elif effect == "solid_color":
            color = torch.tensor([color_r / 255.0, color_g / 255.0, color_b / 255.0], device=img.device)
            if C == 3:
                fill = color.view(1, 1, 1, 3).expand(B, H, W, 3)
            else:
                fill = img.clone()
                fill[..., :3] = color.view(1, 1, 1, 3).expand(B, H, W, 3)
        elif effect == "invert":
            fill = 1.0 - img
        elif effect == "blur":
            fill = self._gaussian_blur(img, blur_radius)
        else:
            fill = img

        out = img * (1.0 - mask_bhwc) + fill * mask_bhwc
        out = out.clamp(0.0, 1.0)

        mask_out = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, H, W)
        return (out, mask_out)


NODE_CLASS_MAPPINGS = {
    "ImageMaskCompositeNode": ImageMaskCompositeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMaskCompositeNode": "Image Mask Composite 🖼️",
}
