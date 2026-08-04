"""
TensorVizion ComfyUI Nodes
latent_histogram_node.py — Renders a per-channel value-distribution
histogram as a viewable IMAGE, plus an outlier-percentage stat. Latent
Visualizer already reports per-channel mean/std/min/max as text; this
node is a genuinely different diagnostic — a distribution SHAPE view
(spotting a bimodal, heavy-tailed, or spiky distribution that summary
statistics alone can hide) and a direct "how much of this latent is
statistically extreme" answer, useful after a merge/mix/channel-mixer
node to sanity check whether the result looks broken before spending a
full VAE decode to find out visually.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


class LatentHistogramNode:
    """
    Computes a histogram of latent values (`num_bins` buckets, pooled
    across the whole batch) either per-channel (small multiples, one
    histogram per channel) or combined (all channels pooled into one
    histogram) depending on `mode`.

    `outlier_std_threshold` defines what counts as an outlier: any value
    more than this many standard deviations from the channel's (or
    overall, in combined mode) mean. `outlier_percent` reports what
    fraction of all values crossed that line — a healthy latent from a
    normal sampling process is usually in the low single digits; a much
    higher number often indicates a bad merge, an over-strength noise
    injection, or a channel-mixer setting that's pushed values far
    outside their normal range.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["per_channel", "combined"],),
                "num_bins": ("INT", {"default": 48, "min": 8, "max": 256, "step": 1}),
                "outlier_std_threshold": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "chart_width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32}),
                "chart_height": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("histogram_image", "outlier_percent", "summary")
    FUNCTION = "process"

    def _draw_single_histogram(self, values, num_bins, width, height, title=""):
        counts, edges = np.histogram(values, bins=num_bins)
        max_count = max(counts.max(), 1)

        img = Image.new("RGB", (width, height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)

        margin_bottom = 20
        margin_top = 16 if title else 4
        plot_h = height - margin_bottom - margin_top
        bar_w = width / num_bins

        for i, count in enumerate(counts):
            bar_h = (count / max_count) * plot_h
            x0 = i * bar_w
            x1 = x0 + max(1, bar_w - 1)
            y1 = height - margin_bottom
            y0 = y1 - bar_h
            draw.rectangle([x0, y0, x1, y1], fill=(120, 200, 255))

        # Zero line marker, since latents are typically zero-centered and
        # it's useful to see at a glance whether the distribution is
        # symmetric around it.
        if edges[0] < 0 < edges[-1]:
            zero_frac = (0 - edges[0]) / (edges[-1] - edges[0])
            zero_x = zero_frac * width
            draw.line([(zero_x, margin_top), (zero_x, height - margin_bottom)], fill=(255, 100, 100), width=1)

        if title:
            draw.text((4, 2), title, fill=(230, 230, 230))

        return np.array(img).astype(np.float32) / 255.0

    def process(self, latent, mode, num_bins, outlier_std_threshold, chart_width, chart_height):
        samples = latent["samples"]  # (B, C, H, W)
        flat = samples.detach().cpu().float().numpy()
        B, C, H, W = flat.shape

        overall_mean = flat.mean()
        overall_std = flat.std()
        outlier_mask = np.abs(flat - overall_mean) > (outlier_std_threshold * overall_std)
        outlier_percent = float(outlier_mask.mean() * 100.0)

        if mode == "combined":
            chart = self._draw_single_histogram(
                flat.reshape(-1), num_bins, chart_width, chart_height,
                title=f"All channels (mean={overall_mean:+.3f} std={overall_std:.3f})",
            )
        else:
            # Tile one small histogram per channel into a grid, similar
            # spirit to Latent Visualizer's grid layout but for
            # distributions instead of spatial planes.
            cols = min(4, C)
            rows = (C + cols - 1) // cols
            cell_w = chart_width // cols
            cell_h = chart_height // rows if rows > 0 else chart_height

            grid = np.ones((cell_h * rows, cell_w * cols, 3), dtype=np.float32) * (20 / 255.0)
            for c in range(C):
                ch_values = flat[:, c].reshape(-1)
                cell = self._draw_single_histogram(ch_values, max(8, num_bins // 2), cell_w, cell_h, title=f"ch{c}")
                r, col = divmod(c, cols)
                grid[r * cell_h:(r + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = cell
            chart = grid

        chart_tensor = torch.from_numpy(chart).unsqueeze(0)  # (1,H,W,3)

        summary = (
            f"Batch: {B}  Channels: {C}  Spatial: {H}x{W}\n"
            f"Overall mean={overall_mean:+.4f} std={overall_std:.4f}\n"
            f"Outliers (>{outlier_std_threshold} std from mean): {outlier_percent:.2f}%"
        )
        return (chart_tensor, outlier_percent, summary)


NODE_CLASS_MAPPINGS = {
    "LatentHistogramNode": LatentHistogramNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentHistogramNode": "Latent Histogram 📊",
}
