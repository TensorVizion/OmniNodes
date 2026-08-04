"""
TensorVizion ComfyUI Nodes
video_loop_composer_node.py — Turns a finite IMAGE batch into a
seamlessly loopable sequence: straight loop, pingpong (forward then
reverse), or crossfade-loop (blends the tail back into the head so a
straight loop doesn't have a visible seam). A common finishing step for
short AI-generated clips that isn't covered by any existing node —
Latent Interpolate produces a one-way batch, this node is what turns
that into something that can actually loop cleanly for a GIF/social post.
"""

import torch


class VideoLoopComposerNode:
    """
    `mode`: `pingpong` plays the batch forward then backward (the reversed
    pass excludes the duplicate endpoint frames so the loop doesn't stutter
    on a held frame); `straight_loop` just repeats the batch `repeats`
    times as-is (only truly seamless if the source was already designed to
    loop); `crossfade_loop` blends the last `crossfade_frames` of the
    batch into the first `crossfade_frames`, smoothing away a seam so an
    otherwise non-looping clip reads as a loop.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":           ("IMAGE",),
                "mode":             (["pingpong", "straight_loop", "crossfade_loop"], {"default": "pingpong"}),
                "repeats":          ("INT", {"default": 1, "min": 1, "max": 20}),
                "crossfade_frames": ("INT", {"default": 4, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES  = ("IMAGE",  "STRING")
    RETURN_NAMES  = ("images", "summary")
    FUNCTION      = "run"

    def run(self, images, mode, repeats, crossfade_frames):
        n_source = images.shape[0]

        if mode == "pingpong":
            if n_source < 3:
                looped = images
            else:
                forward = images
                reverse = images.flip(dims=[0])[1:-1]  # drop duplicate endpoints
                one_cycle = torch.cat([forward, reverse], dim=0)
                looped = one_cycle.repeat(repeats, 1, 1, 1)

        elif mode == "straight_loop":
            looped = images.repeat(repeats, 1, 1, 1)

        else:  # crossfade_loop
            cf = min(crossfade_frames, n_source // 2) if n_source >= 2 else 0
            if cf <= 0:
                looped = images.repeat(repeats, 1, 1, 1)
            else:
                head = images[:cf].clone()
                tail = images[-cf:].clone()
                blended_head = torch.zeros_like(head)
                for k in range(cf):
                    t = (k + 1) / (cf + 1)
                    blended_head[k] = tail[k] * (1 - t) + head[k] * t

                one_cycle = torch.cat([blended_head, images[cf:]], dim=0)
                looped = one_cycle.repeat(repeats, 1, 1, 1)

        summary = (
            f"Mode:             {mode}\n"
            f"Source frames:    {n_source}\n"
            f"Repeats:          {repeats}\n"
            f"Output frames:    {looped.shape[0]}\n"
        )
        if mode == "crossfade_loop":
            summary += f"Crossfade length: {min(crossfade_frames, n_source // 2) if n_source >= 2 else 0} frames"

        return (looped, summary)


NODE_CLASS_MAPPINGS = {
    "VideoLoopComposerNode": VideoLoopComposerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoopComposerNode": "Video Loop Composer 🔂",
}
