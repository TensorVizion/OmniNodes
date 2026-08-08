"""
TensorVizion ComfyUI Nodes
latent_qc_gate_node.py — Inspects a sampled LATENT for common failure
signatures BEFORE it reaches VAEDecode/Save, and routes to a fallback
latent if something looks broken. Try/Catch (Workflow Nodes) already
covers this pattern for STRING/scalar values, but nothing in the pack
checks a LATENT tensor itself — the actual output of every sampler in
this pack, and the thing most worth checking before burning a decode +
save on a long unattended batch run.

Checks performed, each independently toggleable:
  - NaN / Inf values anywhere in the tensor (a real, not-rare failure
    mode from a bad LoRA/DoRA merge strength, an incompatible VAE, or
    certain sampler/scheduler combinations diverging)
  - Near-zero variance (a suspiciously flat/blank latent — often means
    the sampler effectively did nothing, e.g. denoise=0 by mistake, or
    conditioning collapsed to empty)
  - Extreme outlier saturation using median/MAD (robust to large-
    fraction contamination, unlike a naive mean/std z-score — see the
    check's own comment for why that distinction matters), as an
    automatic PASS/FAIL gate you can wire into a real branch, rather
    than a chart you have to read yourself every single run
"""

import torch


class LatentQCGateNode:
    """
    Runs the enabled checks against `latent`. If ALL enabled checks pass,
    `latent` is returned unchanged and `passed=True`. If ANY enabled
    check fails, `fallback_latent` is returned instead (if connected —
    otherwise the original `latent` is returned anyway, since silently
    producing a completely empty LATENT output would break most
    downstream nodes) and `passed=False`, with `reason` explaining
    exactly which check(s) failed.

    Wire `passed` into a Conditional Gate to skip the decode/save chain
    entirely on a failed batch item, or into Discord Notify (Web API
    Nodes) to get pinged only when something actually goes wrong during
    a long unattended run — rather than checking on it manually.

    `variance_threshold` and `outlier_std_threshold` mirror the same
    statistical language as Latent Histogram (Latent Nodes) so a
    threshold that looked reasonable on that node's chart can be reused
    here as a hard pass/fail cutoff.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "check_nan_inf": ("BOOLEAN", {"default": True}),
                "check_near_blank": ("BOOLEAN", {"default": True}),
                "variance_threshold": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "check_outlier_saturation": ("BOOLEAN", {"default": True}),
                "outlier_std_threshold": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 15.0, "step": 0.1}),
                "outlier_percent_limit": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "fallback_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("latent", "passed", "reason")
    FUNCTION = "check"

    def check(self, latent, check_nan_inf, check_near_blank, variance_threshold,
              check_outlier_saturation, outlier_std_threshold, outlier_percent_limit,
              fallback_latent=None):
        samples = latent["samples"]
        failures = []

        if check_nan_inf:
            has_nan = torch.isnan(samples).any().item()
            has_inf = torch.isinf(samples).any().item()
            if has_nan or has_inf:
                bad = []
                if has_nan:
                    bad.append("NaN")
                if has_inf:
                    bad.append("Inf")
                failures.append(f"Contains {'/'.join(bad)} values")

        variance = None
        if check_near_blank:
            variance = samples.float().var().item()
            if variance < variance_threshold:
                failures.append(f"Near-blank latent (variance={variance:.6f} < threshold={variance_threshold})")

        outlier_pct = None
        if check_outlier_saturation:
            flat = samples.float()
            median = flat.median()
            # Median Absolute Deviation, scaled by 0.6745 so it's
            # comparable to a standard deviation for a normal
            # distribution — the standard robust-statistics fix for
            # exactly the failure mode plain mean/std has here: if a
            # large FRACTION of values are extreme (not just a rare
            # few), they inflate the mean/std enough to hide themselves
            # from a naive z-score check (the "masking effect" — mean
            # and std have a 0% breakdown point, median/MAD have 50%).
            # A QC gate specifically needs to catch that large-fraction
            # case (e.g. a whole corrupted region from a bad merge), not
            # just scattered single-pixel noise.
            mad = (flat - median).abs().median()
            robust_scale = mad / 0.6745 if mad > 0 else flat.std()
            if robust_scale > 0:
                outlier_mask = (flat - median).abs() > (outlier_std_threshold * robust_scale)
                outlier_pct = outlier_mask.float().mean().item() * 100.0
                if outlier_pct > outlier_percent_limit:
                    failures.append(
                        f"Outlier saturation {outlier_pct:.2f}% exceeds limit {outlier_percent_limit}% "
                        f"(>{outlier_std_threshold} robust-z from median)"
                    )

        passed = len(failures) == 0

        if passed:
            reason = "All enabled checks passed."
            return (latent, True, reason)

        reason = "FAILED: " + "; ".join(failures)
        output_latent = fallback_latent if fallback_latent is not None else latent
        if fallback_latent is None:
            reason += " (no fallback_latent connected — passing the original latent through anyway)"
        return (output_latent, False, reason)


NODE_CLASS_MAPPINGS = {
    "LatentQCGateNode": LatentQCGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentQCGateNode": "Latent QC Gate 🚧",
}
