"""
TensorVizion ComfyUI Nodes
latent_anomaly_mask_node.py — Flags statistically outlier regions in a
latent (values many local-std-devs from their neighbourhood mean) before
a VAE decode. A debugging tool for spotting where a generation is likely
to produce an artifact — a hot pixel, a seam, a blown-out patch — without
spending the decode + visual-inspection cycle first.
"""

import numpy as np
import torch


class LatentAnomalyMaskNode:
    """
    For each spatial position, compares the local value (averaged across
    channels) against a local neighbourhood mean/std computed with a
    sliding window of `window_size`, and flags any position more than
    `std_threshold` standard deviations away as anomalous. Outputs a MASK
    highlighting flagged regions (usable directly with Image Mask
    Composite or any inpainting workflow after decode) plus a cleaned
    LATENT with flagged positions optionally softened toward the local
    mean (`auto_correct`), and a text report of how much of the latent
    was flagged.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":        ("LATENT",),
                "window_size":   ("INT",   {"default": 5,   "min": 3,   "max": 31,  "step": 2}),
                "std_threshold": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "auto_correct":  ("BOOLEAN", {"default": False}),
                "correct_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("MASK",  "LATENT",           "STRING")
    RETURN_NAMES  = ("mask",  "corrected_latent", "report")
    FUNCTION      = "run"

    @staticmethod
    def _box_filter(arr, window):
        """Fast local-mean via cumulative sum, same-size 'same' padding."""
        pad = window // 2
        padded = np.pad(arr, pad, mode="edge")
        cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant")

        H, W = arr.shape
        out = np.zeros_like(arr)
        for i in range(H):
            for j in range(W):
                r0, r1 = i, i + window
                c0, c1 = j, j + window
                total = (
                    cumsum[r1, c1] - cumsum[r0, c1]
                    - cumsum[r1, c0] + cumsum[r0, c0]
                )
                out[i, j] = total / (window * window)
        return out

    def run(self, latent, window_size, std_threshold, auto_correct, correct_strength):
        samples = latent["samples"].clone()  # (B, C, H, W)
        B, C, H, W = samples.shape

        # Combine channels into one "energy" map per batch item for anomaly detection
        combined_np = samples.mean(dim=1).detach().cpu().numpy()  # (B, H, W)

        masks = np.zeros((B, H, W), dtype=np.float32)

        for b in range(B):
            plane = combined_np[b]
            local_mean = self._box_filter(plane, window_size)
            local_sq_mean = self._box_filter(plane ** 2, window_size)
            local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-8)
            local_std = np.sqrt(local_var)

            z_score = np.abs(plane - local_mean) / local_std
            flagged = (z_score > std_threshold).astype(np.float32)
            masks[b] = flagged

            if auto_correct:
                blend = flagged * correct_strength
                for c in range(C):
                    ch = samples[b, c]
                    local_mean_t = torch.from_numpy(local_mean).to(ch.device).to(ch.dtype)
                    blend_t = torch.from_numpy(blend).to(ch.device).to(ch.dtype)
                    samples[b, c] = ch * (1 - blend_t) + local_mean_t * blend_t

        mask_tensor = torch.from_numpy(masks)
        total_flagged = float(masks.mean()) * 100.0

        worst_batch = int(np.argmax(masks.mean(axis=(1, 2))))
        worst_pct = float(masks[worst_batch].mean()) * 100.0

        report = (
            f"Window size:      {window_size}\n"
            f"Std threshold:    {std_threshold}\n"
            f"Flagged overall:  {total_flagged:.2f}% of latent area\n"
            f"Worst batch item: #{worst_batch} ({worst_pct:.2f}% flagged)\n"
            f"Auto-correct:     {'on, strength ' + str(correct_strength) if auto_correct else 'off (mask only)'}\n"
            f"Note: this is a pre-decode statistical heuristic, not a\n"
            f"guarantee — always confirm flagged regions visually after decode."
        )

        return (mask_tensor, {"samples": samples}, report)


NODE_CLASS_MAPPINGS = {
    "LatentAnomalyMaskNode": LatentAnomalyMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentAnomalyMaskNode": "Latent Anomaly Mask 🚩",
}
