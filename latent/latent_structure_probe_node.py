"""
TensorVizion ComfyUI Nodes
latent_structure_probe_node.py — Computes per-region statistical energy
maps of a latent (variance/magnitude by spatial grid cell) to diagnose
*where* in the composition a generation is under- or over-saturated,
without a VAE decode. Latent Visualizer answers "what does this roughly
look like"; this node answers "which regions are numerically hot/flat"
— a pre-decode analytical companion, not another preview.
"""

import numpy as np
import torch
from PIL import Image


class LatentStructureProbeNode:
    """
    Divides the latent's spatial dimensions into a `grid_rows` x `grid_cols`
    grid and computes a chosen statistic (`variance`, `mean_magnitude`, or
    `max_magnitude`) per cell, averaged across channels and batch. Outputs
    a small heatmap IMAGE (grid-cell-resolution, one pixel block per cell)
    plus a text report flagging the hottest and flattest cells — useful for
    spotting a likely-oversaturated corner or a suspiciously flat/dead
    region before spending a full VAE decode + visual inspection cycle.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":       ("LATENT",),
                "statistic":    (["variance", "mean_magnitude", "max_magnitude"], {"default": "variance"}),
                "grid_rows":    ("INT", {"default": 4, "min": 1, "max": 32}),
                "grid_cols":    ("INT", {"default": 4, "min": 1, "max": 32}),
                "cell_pixels":  ("INT", {"default": 64, "min": 8, "max": 256}),
            }
        }

    RETURN_TYPES  = ("IMAGE",   "STRING")
    RETURN_NAMES  = ("heatmap", "report")
    FUNCTION      = "run"

    @staticmethod
    def _colorize(norm_grid):
        """Simple blue -> yellow -> red heatmap colormap, no matplotlib dep."""
        r = np.clip(1.5 * norm_grid - 0.5, 0, 1)
        g = np.clip(1.5 * (1 - np.abs(norm_grid - 0.5) * 2), 0, 1)
        b = np.clip(1.5 * (0.5 - norm_grid) + 0.5, 0, 1)
        return np.stack([r, g, b], axis=-1)

    def run(self, latent, statistic, grid_rows, grid_cols, cell_pixels):
        samples = latent["samples"]  # (B, C, H, W)
        B, C, H, W = samples.shape

        # Average over batch/channel first so the grid reflects overall structure
        combined = samples.mean(dim=0)  # (C, H, W)

        row_bounds = np.linspace(0, H, grid_rows + 1).astype(int)
        col_bounds = np.linspace(0, W, grid_cols + 1).astype(int)

        grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        for r in range(grid_rows):
            for c in range(grid_cols):
                cell = combined[:, row_bounds[r]:row_bounds[r + 1], col_bounds[c]:col_bounds[c + 1]]
                cell_np = cell.detach().cpu().numpy()

                if statistic == "variance":
                    grid[r, c] = float(np.var(cell_np))
                elif statistic == "mean_magnitude":
                    grid[r, c] = float(np.mean(np.abs(cell_np)))
                else:  # max_magnitude
                    grid[r, c] = float(np.max(np.abs(cell_np))) if cell_np.size else 0.0

        vmin, vmax = float(grid.min()), float(grid.max())
        norm = (grid - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(grid)

        rgb = self._colorize(norm)  # (rows, cols, 3), values 0-1
        rgb_upsampled = np.kron(rgb, np.ones((cell_pixels, cell_pixels, 1)))
        rgb_upsampled = np.clip(rgb_upsampled, 0.0, 1.0).astype(np.float32)

        heatmap_tensor = torch.from_numpy(rgb_upsampled)[None,]  # (1, H', W', 3)

        flat_idx_max = int(np.argmax(grid))
        flat_idx_min = int(np.argmin(grid))
        hot_r, hot_c = divmod(flat_idx_max, grid_cols)
        cold_r, cold_c = divmod(flat_idx_min, grid_cols)

        report = (
            f"Statistic:      {statistic}\n"
            f"Grid:           {grid_rows} rows x {grid_cols} cols\n"
            f"Latent shape:   B={B}, C={C}, H={H}, W={W}\n"
            f"Range:          {vmin:.4f} to {vmax:.4f}\n"
            f"Hottest cell:   row {hot_r}, col {hot_c}  (value {grid[hot_r, hot_c]:.4f})\n"
            f"Flattest cell:  row {cold_r}, col {cold_c}  (value {grid[cold_r, cold_c]:.4f})\n"
            f"Note: a very flat/near-zero cell often decodes to a dead or\n"
            f"low-detail region; a much hotter cell than its neighbours can\n"
            f"indicate an artifact or over-saturated area worth a closer look."
        )

        return (heatmap_tensor, report)


NODE_CLASS_MAPPINGS = {
    "LatentStructureProbeNode": LatentStructureProbeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentStructureProbeNode": "Latent Structure Probe 📡",
}
