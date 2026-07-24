"""
TensorVizion ComfyUI Nodes
video_speed_ramp_node.py — Retimes an IMAGE batch to simulate a speed
ramp (slow-mo into normal speed, speed-up into freeze, etc.) by resampling
frame indices along an eased curve rather than a flat linear rate. The
frame-timing analogue of Prompt Weight Scheduler's eased value curves —
same curve shapes, applied to "which frame plays when" instead of "what
weight applies when".
"""

import numpy as np
import torch


class VideoSpeedRampNode:
    """
    Produces `output_frames` frames by resampling `images` along a curve
    from `start_speed` to `end_speed` (both relative multipliers — 1.0 =
    normal speed, 0.5 = half speed/slow-mo, 2.0 = double speed). The
    per-output-frame source position is the integral of the speed curve
    over time, so a `start_speed` of 0.25 and `end_speed` of 1.0 ramps
    smoothly from a quarter-speed slow-mo intro up to normal speed by the
    end, using the same `linear`/`ease_in`/`ease_out`/`ease_in_out` curve
    shapes as Prompt Weight Scheduler. Missing source frames are filled by
    nearest-neighbour selection (no synthesized in-between frames — pair
    with Video Frame Interpolate afterward for smoother slow-motion
    sections instead of visible held frames).
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":        ("IMAGE",),
                "output_frames": ("INT",   {"default": 48, "min": 1, "max": 4096}),
                "start_speed":   ("FLOAT", {"default": 0.25, "min": 0.05, "max": 4.0, "step": 0.05}),
                "end_speed":     ("FLOAT", {"default": 1.0,  "min": 0.05, "max": 4.0, "step": 0.05}),
                "curve":         (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "ease_out"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",  "STRING")
    RETURN_NAMES  = ("images", "summary")
    FUNCTION      = "run"

    @staticmethod
    def _ease(t, curve):
        if curve == "linear":
            return t
        if curve == "ease_in":
            return t * t
        if curve == "ease_out":
            return 1 - (1 - t) * (1 - t)
        if curve == "ease_in_out":
            return 3 * t * t - 2 * t * t * t
        return t

    def run(self, images, output_frames, start_speed, end_speed, curve):
        n_source = images.shape[0]
        if n_source < 2 or output_frames < 1:
            return (images, "Batch too small or output_frames < 1 — passthrough.")

        # Build the speed value at each output step, eased between start/end
        t_vals = np.linspace(0.0, 1.0, output_frames)
        eased = np.array([self._ease(t, curve) for t in t_vals])
        speed_curve = start_speed + (end_speed - start_speed) * eased

        # Integrate the speed curve to get cumulative "source position" progress,
        # then rescale so it spans the full source frame range by the last output frame.
        cumulative = np.cumsum(speed_curve)
        cumulative -= cumulative[0]
        if cumulative[-1] > 1e-9:
            cumulative = cumulative / cumulative[-1]
        source_positions = cumulative * (n_source - 1)
        indices = np.clip(np.round(source_positions), 0, n_source - 1).astype(int)

        output = images[torch.from_numpy(indices)]

        avg_speed = float(speed_curve.mean())
        summary = (
            f"Source frames:  {n_source}\n"
            f"Output frames:  {output_frames}\n"
            f"Speed range:    {start_speed:.2f}x -> {end_speed:.2f}x ({curve})\n"
            f"Average speed:  {avg_speed:.2f}x\n"
            f"Note: frames are nearest-neighbour selected, not synthesized — "
            f"pair with Video Frame Interpolate for smoother slow-motion sections."
        )

        return (output, summary)


NODE_CLASS_MAPPINGS = {
    "VideoSpeedRampNode": VideoSpeedRampNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSpeedRampNode": "Video Speed Ramp 🐢",
}
