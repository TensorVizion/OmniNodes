import torch
import torch.nn.functional as F


class TemporalConsistencyBridge:
    """
    Bridges identity between consecutive frames using lightweight statistical
    conditioning extracted from previous frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_frame": ("IMAGE",),
                "previous_frames": ("IMAGE",),
                "identity_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "drift_tolerance": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "feature_mode": (["mean_std", "histogram", "combined"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_frame", "drift_report")
    FUNCTION = "bridge"
    CATEGORY = "creative/director_toolkit"

    def bridge(self, current_frame, previous_frames, identity_strength,
               drift_tolerance, feature_mode):
        # Aggregate statistics from previous frames
        pf = previous_frames
        prev_mean = pf.mean(dim=(0, 1, 2), keepdim=True)
        prev_std = pf.std(dim=(0, 1, 2), keepdim=True) + 1e-8
        curr_mean = current_frame.mean(dim=(0, 1, 2), keepdim=True)
        curr_std = current_frame.std(dim=(0, 1, 2), keepdim=True) + 1e-8

        # Drift metric: L2 distance between statistical fingerprints
        drift = torch.sqrt(((prev_mean - curr_mean) ** 2).sum() +
                           ((prev_std - curr_std) ** 2).sum()).item()
        drift_normalized = min(drift / 2.0, 1.0)

        warning = ""
        if drift_normalized > drift_tolerance:
            warning = f" [WARNING: drift {drift_normalized:.3f} exceeds tolerance {drift_tolerance:.2f}]"

        # Color stabilization: nudge current frame toward previous statistics
        if feature_mode in ("mean_std", "combined"):
            stabilized = (current_frame - curr_mean) / curr_std * prev_std + prev_mean
            stabilized = current_frame * (1 - identity_strength) + stabilized * identity_strength
        else:
            stabilized = current_frame.clone()

        # Histogram matching approximation (per-channel CDF transfer)
        if feature_mode in ("histogram", "combined"):
            stabilized = self._histogram_match(stabilized, pf, identity_strength * 0.5)

        stabilized = torch.clamp(stabilized, 0.0, 1.0)

        report = (
            f"Drift score: {drift_normalized:.4f}\n"
            f"Tolerance: {drift_tolerance:.2f}\n"
            f"Identity strength applied: {identity_strength:.2f}\n"
            f"Feature mode: {feature_mode}{warning}"
        )
        return (stabilized, report)

    def _histogram_match(self, source, reference, strength):
        matched = source.clone()
        for c in range(source.shape[-1]):
            src_vals = source[..., c].flatten()
            ref_vals = reference[..., c].flatten()
            # Quantile-based mapping (approximate histogram match)
            src_sorted, src_idx = torch.sort(src_vals)
            ref_sorted, _ = torch.sort(ref_vals)
            n = min(len(src_sorted), len(ref_sorted))
            mapping = torch.interp(
                torch.linspace(0, 1, n, device=source.device),
                torch.linspace(0, 1, n, device=source.device),
                ref_sorted[:n]
            )
            # Reorder source by rank
            ranks = src_idx.argsort()
            transferred = torch.zeros_like(src_vals)
            transferred[src_idx] = mapping[ranks][:len(src_vals)]
            transferred = transferred.reshape(source[..., c].shape)
            matched[..., c] = source[..., c] * (1 - strength) + transferred * strength
        return matched