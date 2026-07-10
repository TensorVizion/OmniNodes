"""
TensorVizion ComfyUI Nodes
latent_interpolate_node.py — Generates a batch of interpolated latents
between two latent tensors, using either linear or spherical (SLERP)
interpolation with selectable easing. Useful for latent walks, morph
animations, and exploring the space between two seeds/prompts.
"""

import torch


class LatentInterpolateNode:
    """
    Produces `steps` latents walking from latent_a to latent_b and returns
    them stacked as a single LATENT batch (feed straight into a batched
    KSampler / VAE Decode / Save Image sequence for an animation).

    method:
      lerp  — straight linear interpolation, fastest, can dip in "energy"
              between very different latents
      slerp — spherical interpolation over the flattened latent vector,
              keeps roughly constant magnitude along the path (generally
              the better choice for diffusion latents)

    easing reshapes the timestep spacing, not the interpolation math itself:
      linear       — even spacing
      ease_in      — slow start, fast finish
      ease_out     — fast start, slow finish
      ease_in_out  — slow start and finish, fast middle

    Note: only the first batch item of latent_a / latent_b is used. If your
    latents come from single-image encodes (the common case) this is exactly
    what you want; if either input has batch size > 1 the rest is ignored.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "steps": ("INT", {"default": 8, "min": 2, "max": 64}),
                "method": (["slerp", "lerp"],),
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                "include_endpoints": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "walk_info")
    FUNCTION = "interpolate"

    # ------------------------------------------------------------------
    def _ease(self, t, mode):
        if mode == "ease_in":
            return t * t
        if mode == "ease_out":
            return 1.0 - (1.0 - t) * (1.0 - t)
        if mode == "ease_in_out":
            return 3.0 * t * t - 2.0 * t * t * t  # smoothstep
        return t  # linear

    def _lerp(self, a, b, t):
        return torch.lerp(a, b, t)

    def _slerp(self, a, b, t, eps=1e-6):
        """Spherical interpolation treating the whole tensor as one vector."""
        shape = a.shape
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)

        a_norm = a_flat / (a_flat.norm() + eps)
        b_norm = b_flat / (b_flat.norm() + eps)

        dot = torch.clamp((a_norm * b_norm).sum(), -1.0 + eps, 1.0 - eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        if sin_omega.abs() < eps:
            # Nearly parallel vectors -- fall back to linear interpolation
            out = torch.lerp(a_flat, b_flat, t)
        else:
            coef_a = torch.sin((1.0 - t) * omega) / sin_omega
            coef_b = torch.sin(t * omega) / sin_omega
            out = coef_a * a_flat + coef_b * b_flat

        return out.reshape(shape)

    # ------------------------------------------------------------------
    def interpolate(self, latent_a, latent_b, steps, method, easing, include_endpoints):
        a = latent_a["samples"][0:1].clone()  # (1, C, H, W)
        b = latent_b["samples"][0:1].clone()

        if a.shape != b.shape:
            b = torch.nn.functional.interpolate(
                b, size=(a.shape[2], a.shape[3]), mode="bilinear", align_corners=False
            )

        if include_endpoints:
            raw_ts = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]
        else:
            raw_ts = [(i + 1) / (steps + 1) for i in range(steps)]

        eased_ts = [self._ease(t, easing) for t in raw_ts]

        frames = []
        for t_val in eased_ts:
            t = torch.tensor(t_val, dtype=a.dtype, device=a.device)
            if method == "slerp":
                frame = self._slerp(a[0], b[0], t)
            else:
                frame = self._lerp(a[0], b[0], t)
            frames.append(frame.unsqueeze(0))

        out = torch.cat(frames, dim=0)  # (steps, C, H, W)

        walk_info = (
            f"Method:      {method}\n"
            f"Easing:      {easing}\n"
            f"Steps:       {steps}\n"
            f"Endpoints:   {include_endpoints}\n"
            f"Output batch:{list(out.shape)}"
        )

        return ({"samples": out}, walk_info)


NODE_CLASS_MAPPINGS = {
    "LatentInterpolateNode": LatentInterpolateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentInterpolateNode": "Latent Interpolate 🌉",
}
