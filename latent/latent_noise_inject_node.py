"""
TensorVizion ComfyUI Nodes
latent_noise_inject_node.py — Injects controllable noise into a LATENT tensor
using several noise types and blend modes. Useful for latent space exploration,
variation seeding, and creative diffusion steering.
"""

import torch
import numpy as np


class LatentNoiseInjectNode:
    """
    Adds structured or random noise directly to a latent tensor.

    Noise types : gaussian, uniform, perlin (approximate), salt_pepper
    Blend modes : add, multiply, lerp (linear interpolate)

    The seed parameter makes noise deterministic and reproducible.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":      ("LATENT",),
                "noise_type":  (["gaussian", "uniform", "perlin", "salt_pepper"],),
                "blend_mode":  (["add", "multiply", "lerp"],),
                "strength":    ("FLOAT",   {"default": 0.1,  "min": 0.0, "max": 2.0,  "step": 0.01}),
                "seed":        ("INT",     {"default": 0,    "min": 0,   "max": 2**32 - 1}),
                "channel_mask":("STRING",  {"default": "all",
                                            "multiline": False,
                                            "placeholder": "all  or  0,1  or  2,3"}),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "inject_noise"

    # ------------------------------------------------------------------
    def _perlin_noise(self, shape, rng):
        """Approximate 2-D tileable Perlin-like noise via octave summation."""
        H, W   = shape[-2], shape[-1]
        result = np.zeros((H, W), dtype=np.float32)
        amp, freq, total_amp = 1.0, 1, 0.0
        for _ in range(4):
            gh = max(1, H // freq)
            gw = max(1, W // freq)
            grid = rng.standard_normal((gh + 1, gw + 1)).astype(np.float32)
            # Bilinear upsample
            from PIL import Image as _P
            up = np.array(_P.fromarray(grid).resize((W, H), _P.BILINEAR))
            result    += up * amp
            total_amp += amp
            amp       *= 0.5
            freq      *= 2
        return (result / total_amp).astype(np.float32)

    # ------------------------------------------------------------------
    def inject_noise(self, latent, noise_type, blend_mode, strength, seed, channel_mask):
        samples = latent["samples"].clone()          # (B, C, H, W)
        B, C, H, W = samples.shape
        rng = np.random.default_rng(seed)

        # Parse channel mask
        if channel_mask.strip().lower() == "all":
            channels = list(range(C))
        else:
            try:
                channels = [int(x.strip()) for x in channel_mask.split(",") if x.strip()]
                channels = [c for c in channels if 0 <= c < C]
            except ValueError:
                channels = list(range(C))

        for b in range(B):
            for c in channels:
                s = samples[b, c]

                if noise_type == "gaussian":
                    noise = torch.from_numpy(
                        rng.standard_normal((H, W)).astype(np.float32)
                    ).to(s.device)

                elif noise_type == "uniform":
                    noise = torch.from_numpy(
                        rng.uniform(-1.0, 1.0, (H, W)).astype(np.float32)
                    ).to(s.device)

                elif noise_type == "perlin":
                    noise = torch.from_numpy(
                        self._perlin_noise((H, W), rng)
                    ).to(s.device)

                elif noise_type == "salt_pepper":
                    base  = np.zeros((H, W), dtype=np.float32)
                    mask  = rng.random((H, W))
                    base[mask < strength * 0.5]  =  1.0
                    base[mask > 1.0 - strength * 0.5] = -1.0
                    noise = torch.from_numpy(base).to(s.device)
                    # salt_pepper uses direct assignment — skip blend modes
                    samples[b, c] = torch.where(
                        torch.from_numpy(mask).to(s.device) < strength * 0.5,
                        torch.ones_like(s),
                        torch.where(
                            torch.from_numpy(mask).to(s.device) > 1.0 - strength * 0.5,
                            -torch.ones_like(s), s
                        )
                    )
                    continue

                # Apply blend mode
                if blend_mode == "add":
                    samples[b, c] = s + noise * strength
                elif blend_mode == "multiply":
                    samples[b, c] = s * (1.0 + noise * strength)
                elif blend_mode == "lerp":
                    samples[b, c] = torch.lerp(s, noise, strength)

        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "LatentNoiseInjectNode": LatentNoiseInjectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentNoiseInjectNode": "Latent Noise Inject 🌊",
}
