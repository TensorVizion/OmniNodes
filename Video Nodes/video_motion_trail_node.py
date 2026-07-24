"""
TensorVizion ComfyUI Nodes
video_motion_trail_node.py — Blends each frame with a decaying
accumulation of previous frames, producing a motion-trail / long-exposure
look across an IMAGE batch. Pure tensor math, no external dependencies —
a creative effect node rather than a utility, filling the gap where
Image Blend/Vignette & Glow exist for single images but nothing in the
pack works across a whole batch's temporal dimension for a stylized look.
"""

import torch


class VideoMotionTrailNode:
    """
    Walks through `images` in order, maintaining a running "trail" buffer
    that accumulates previous frames at `decay` (0 = no trail, each output
    frame is just its own input frame; close to 1.0 = very long-lived
    trails). Each output frame is a blend of the current input frame and
    the accumulated trail, mixed by `blend_strength`. `trail_mode` picks
    how the trail accumulates: `average` (soft, dreamy accumulation) or
    `lighten` (keeps only the brightest value per-pixel across the trail —
    classic star-trail/long-exposure look).
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "decay":           ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01}),
                "blend_strength":  ("FLOAT", {"default": 0.6,  "min": 0.0, "max": 1.0,  "step": 0.01}),
                "trail_mode":      (["average", "lighten"], {"default": "average"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    FUNCTION      = "run"

    def run(self, images, decay, blend_strength, trail_mode):
        n_frames = images.shape[0]
        if n_frames < 2:
            return (images,)

        output = torch.zeros_like(images)
        trail = images[0].clone()
        output[0] = images[0]

        for i in range(1, n_frames):
            current = images[i]

            if trail_mode == "average":
                trail = trail * decay + current * (1.0 - decay)
            else:  # lighten
                trail = torch.maximum(trail * decay, current)

            output[i] = current * (1.0 - blend_strength) + trail * blend_strength

        output = torch.clamp(output, 0.0, 1.0)

        return (output,)


NODE_CLASS_MAPPINGS = {
    "VideoMotionTrailNode": VideoMotionTrailNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMotionTrailNode": "Video Motion Trail 🌌",
}
