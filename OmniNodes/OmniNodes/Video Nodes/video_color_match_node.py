"""
TensorVizion ComfyUI Nodes
video_color_match_node.py — Matches the color/exposure statistics of an
IMAGE batch to a reference (a single reference image, or another clip's
average) using per-channel mean/std matching in LAB-like luminance space.
Built for consistency when stitching segments generated in separate runs
(e.g. multiple FrameForge passes, or clips joined with Video Concat/
Splice) where each segment can drift slightly in exposure or white
balance even from the same prompt/seed family.
"""

import torch


class VideoColorMatchNode:
    """
    Adjusts `images` so its per-channel mean and standard deviation match
    `reference`'s, frame-by-frame statistics computed once from the
    reference and applied uniformly across the whole clip (rather than
    per-frame, which would cause flicker).

    `reference` can be a single image (its own stats are used directly)
    or a batch (stats are averaged across all its frames first) — e.g.
    feed it the first few frames of a previously-generated segment to
    match a new segment's color to it before splicing them together.

    `strength` blends between the original (`0.0`) and the fully matched
    result (`1.0`), useful when a full match looks too aggressive.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "match_mode": (["mean_std", "mean_only"], {"default": "mean_std"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "summary")
    FUNCTION = "run"

    def run(self, images, reference, strength, match_mode):
        eps = 1e-6

        # Per-channel stats over all pixels and all reference frames
        ref_flat = reference.reshape(-1, reference.shape[-1])  # (N, C)
        ref_mean = ref_flat.mean(dim=0)  # (C,)
        ref_std = ref_flat.std(dim=0) + eps

        src_flat = images.reshape(-1, images.shape[-1])
        src_mean = src_flat.mean(dim=0)
        src_std = src_flat.std(dim=0) + eps

        if match_mode == "mean_std":
            scale = (ref_std / src_std).view(1, 1, 1, -1)
        else:
            scale = torch.ones(1, 1, 1, images.shape[-1], device=images.device)

        shift = (ref_mean - src_mean * scale.view(-1)).view(1, 1, 1, -1)

        matched = images * scale + shift
        matched = matched.clamp(0.0, 1.0)

        out = images * (1.0 - strength) + matched * strength
        out = out.clamp(0.0, 1.0)

        summary = (
            f"Match mode:   {match_mode}\n"
            f"Strength:     {strength:.2f}\n"
            f"Source mean:  {[round(v, 4) for v in src_mean.tolist()]}\n"
            f"Reference mean: {[round(v, 4) for v in ref_mean.tolist()]}\n"
            f"Frames matched: {images.shape[0]}"
        )

        return (out, summary)


NODE_CLASS_MAPPINGS = {
    "VideoColorMatchNode": VideoColorMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoColorMatchNode": "Video Color Match 🎨",
}
