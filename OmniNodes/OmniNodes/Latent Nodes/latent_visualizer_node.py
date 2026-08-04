"""
TensorVizion ComfyUI Nodes
latent_visualizer_node.py — Renders a fast false-colour preview of a LATENT
tensor's raw channels (no VAE decode needed) and reports per-channel
statistics. Useful for quickly sanity-checking a latent mid-pipeline.
"""

import numpy as np
import torch


class LatentVisualizerNode:
    """
    Turns raw latent channels into a viewable IMAGE without running them
    through the VAE. This is NOT what the final decoded image will look
    like — it's a structural/diagnostic preview (composition, contrast,
    whether a channel is dead/saturated) that costs almost nothing to
    generate compared to a full VAE decode.

    layout:
      grid            — tiles every channel into a grid, each cell
                         normalised independently (or globally, see below)
      single_channel  — shows just `channel_index`, larger and centred

    Only the first batch item is visualised; per-channel stats are computed
    over the full batch.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "layout": (["grid", "single_channel"],),
                "colormap": (["grayscale", "viridis", "hot", "signed_redblue"],),
                "channel_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "normalize_per_channel": ("BOOLEAN", {"default": True}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "stats")
    FUNCTION = "visualize"

    # ------------------------------------------------------------------
    def _apply_colormap(self, x, colormap):
        """x: (H, W) float32 array in [0, 1] (or [-1, 1] for signed_redblue). Returns (H, W, 3)."""
        if colormap == "grayscale":
            return np.stack([x, x, x], axis=-1).astype(np.float32)

        if colormap == "hot":
            r = np.clip(x * 3.0, 0.0, 1.0)
            g = np.clip(x * 3.0 - 1.0, 0.0, 1.0)
            b = np.clip(x * 3.0 - 2.0, 0.0, 1.0)
            return np.stack([r, g, b], axis=-1).astype(np.float32)

        if colormap == "viridis":
            stops = np.array([
                [0.267, 0.005, 0.329],
                [0.229, 0.322, 0.545],
                [0.128, 0.567, 0.551],
                [0.369, 0.789, 0.383],
                [0.993, 0.906, 0.144],
            ], dtype=np.float32)
            idx = np.clip(x, 0.0, 1.0) * (len(stops) - 1)
            lo = np.floor(idx).astype(int)
            hi = np.clip(lo + 1, 0, len(stops) - 1)
            frac = (idx - lo)[..., None]
            return (stops[lo] * (1.0 - frac) + stops[hi] * frac).astype(np.float32)

        if colormap == "signed_redblue":
            # x expected in [-1, 1]; red = positive, blue = negative
            r = np.clip(x, 0.0, 1.0)
            b = np.clip(-x, 0.0, 1.0)
            g = np.zeros_like(x)
            return np.stack([r, g, b], axis=-1).astype(np.float32)

        return np.stack([x, x, x], axis=-1).astype(np.float32)

    def _normalize(self, arr, colormap):
        if colormap == "signed_redblue":
            peak = float(np.abs(arr).max()) + 1e-8
            return arr / peak  # keep sign, range -> [-1, 1]
        mn, mx = float(arr.min()), float(arr.max())
        return (arr - mn) / (mx - mn + 1e-8)

    # ------------------------------------------------------------------
    def visualize(self, latent, layout, colormap, channel_index, normalize_per_channel, tile_size):
        samples = latent["samples"]  # (B, C, H, W)
        B, C, H, W = samples.shape
        first = samples[0].detach().cpu().float().numpy()  # (C, H, W)

        # Stats over the full batch
        flat = samples.detach().cpu().float().numpy()
        lines = [f"Batch: {B}   Channels: {C}   Spatial: {H}x{W}"]
        for c in range(C):
            ch = flat[:, c]
            lines.append(
                f"  ch{c}: mean={ch.mean():+.4f} std={ch.std():.4f} "
                f"min={ch.min():+.4f} max={ch.max():+.4f}"
            )
        stats = "\n".join(lines)

        from PIL import Image as _P

        if layout == "single_channel":
            idx = max(0, min(channel_index, C - 1))
            plane = first[idx]
            norm = self._normalize(plane, colormap) if normalize_per_channel else plane
            rgb = self._apply_colormap(norm, colormap)
            img = np.array(
                _P.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((tile_size, tile_size), _P.NEAREST)
            ).astype(np.float32) / 255.0
            out = img[None, ...]  # (1, H, W, 3)
        else:
            cols = int(np.ceil(np.sqrt(C)))
            rows = int(np.ceil(C / cols))
            canvas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.float32)

            global_norm = None
            if not normalize_per_channel:
                global_norm = self._normalize(first, colormap)

            for c in range(C):
                plane = first[c]
                norm = self._normalize(plane, colormap) if normalize_per_channel else global_norm[c]
                rgb = self._apply_colormap(norm, colormap)
                tile = np.array(
                    _P.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((tile_size, tile_size), 0)
                ).astype(np.float32) / 255.0
                r, cidx = divmod(c, cols)
                canvas[r * tile_size:(r + 1) * tile_size, cidx * tile_size:(cidx + 1) * tile_size] = tile

            out = canvas[None, ...]  # (1, H, W, 3)

        out_tensor = torch.from_numpy(out.astype(np.float32))
        return (out_tensor, stats)


NODE_CLASS_MAPPINGS = {
    "LatentVisualizerNode": LatentVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentVisualizerNode": "Latent Visualizer 🔬",
}
