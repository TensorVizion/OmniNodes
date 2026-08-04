"""
TensorVizion ComfyUI Nodes
image_noise_inject_node.py — Injects controllable noise/grain into an IMAGE
using several noise types and blend modes. Pixel-space counterpart to the
Latent Noise Inject node.
"""

import numpy as np
import torch


class ImageNoiseInjectNode:
    """
    Adds structured or random noise directly to image pixels.

    Noise types : gaussian, uniform, perlin (approximate), salt_pepper
    Blend modes : add, multiply, lerp (linear interpolate)

    monochrome applies the identical noise pattern to R/G/B (classic film
    grain look); off applies independent noise per channel (colour noise).

    The seed parameter makes noise deterministic and reproducible.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_type": (["gaussian", "uniform", "perlin", "salt_pepper"],),
                "blend_mode": (["add", "multiply", "lerp"],),
                "strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "monochrome": ("BOOLEAN", {"default": True}),
                "clamp_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inject_noise"

    # ------------------------------------------------------------------
    def _perlin_noise(self, shape, rng):
        """Approximate 2-D tileable Perlin-like noise via octave summation."""
        H, W = shape
        result = np.zeros((H, W), dtype=np.float32)
        amp, freq, total_amp = 1.0, 1, 0.0
        from PIL import Image as _P
        for _ in range(4):
            gh = max(1, H // freq)
            gw = max(1, W // freq)
            grid = rng.standard_normal((gh + 1, gw + 1)).astype(np.float32)
            up = np.array(_P.fromarray(grid).resize((W, H), _P.BILINEAR))
            result += up * amp
            total_amp += amp
            amp *= 0.5
            freq *= 2
        return (result / total_amp).astype(np.float32)

    def _make_noise(self, noise_type, H, W, rng, strength):
        if noise_type == "gaussian":
            return rng.standard_normal((H, W)).astype(np.float32)
        if noise_type == "uniform":
            return rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32)
        if noise_type == "perlin":
            return self._perlin_noise((H, W), rng)
        if noise_type == "salt_pepper":
            base = np.zeros((H, W), dtype=np.float32)
            roll = rng.random((H, W))
            base[roll < strength * 0.5] = 1.0
            base[roll > 1.0 - strength * 0.5] = -1.0
            return base
        return np.zeros((H, W), dtype=np.float32)

    # ------------------------------------------------------------------
    def inject_noise(self, image, noise_type, blend_mode, strength, seed, monochrome, clamp_output):
        img = image.clone()  # (B, H, W, C)
        B, H, W, C = img.shape
        rng = np.random.default_rng(seed)

        for b in range(B):
            if monochrome:
                noise_np = self._make_noise(noise_type, H, W, rng, strength)
                noise = torch.from_numpy(noise_np).to(img.device).unsqueeze(-1).expand(H, W, C)
            else:
                planes = [self._make_noise(noise_type, H, W, rng, strength) for _ in range(C)]
                noise = torch.from_numpy(np.stack(planes, axis=-1)).to(img.device)

            s = img[b]
            if noise_type == "salt_pepper":
                # Direct replacement rather than blend -- matches how salt & pepper
                # noise behaves on real sensors/film (hard hits, not additive).
                img[b] = torch.where(noise > 0.5, torch.ones_like(s),
                                      torch.where(noise < -0.5, torch.zeros_like(s), s))
                continue

            if blend_mode == "add":
                img[b] = s + noise * strength
            elif blend_mode == "multiply":
                img[b] = s * (1.0 + noise * strength)
            elif blend_mode == "lerp":
                img[b] = torch.lerp(s, (noise * 0.5 + 0.5), strength)

        if clamp_output:
            img = img.clamp(0.0, 1.0)

        return (img,)


NODE_CLASS_MAPPINGS = {
    "ImageNoiseInjectNode": ImageNoiseInjectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNoiseInjectNode": "Image Noise Inject 🎞️",
}
