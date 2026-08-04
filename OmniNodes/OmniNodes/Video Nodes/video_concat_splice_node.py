"""
TensorVizion ComfyUI Nodes
video_concat_splice_node.py — Joins two or more IMAGE batches end-to-end
into a single sequence, with an optional crossfade at each seam. The
counterpart to Video Loop Composer: that node takes one clip and makes it
loop on itself, this node takes separate clips (e.g. per-scene segments
cut out with Video Trim/Extract, or separate FrameForge/Latent Interpolate
runs) and assembles them into one timeline. Frames are auto-resized to
match clip_a's resolution if any input differs.
"""

import torch
import torch.nn.functional as F


class VideoConcatSpliceNode:
    """
    Concatenates `clip_a` through `clip_d` (unused optional slots are
    skipped) in order. `crossfade_frames` > 0 blends the tail of each clip
    into the head of the next instead of a hard cut, consuming
    `crossfade_frames` from the end of the earlier clip and the start of
    the next so the blended region isn't duplicated on top of a hard edit.

    If a later clip's resolution doesn't match `clip_a`'s, it's resized
    (bilinear) to match before splicing so batches concatenate cleanly.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE",),
                "crossfade_frames": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
            "optional": {
                "clip_b": ("IMAGE",),
                "clip_c": ("IMAGE",),
                "clip_d": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "summary")
    FUNCTION = "run"

    def _match_resolution(self, clip, target_h, target_w):
        if clip.shape[1] == target_h and clip.shape[2] == target_w:
            return clip
        chw = clip.permute(0, 3, 1, 2)
        resized = F.interpolate(chw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    def _splice_pair(self, prev, nxt, crossfade_frames):
        cf = min(crossfade_frames, prev.shape[0], nxt.shape[0])
        if cf <= 0:
            return torch.cat([prev, nxt], dim=0)

        prev_body = prev[:-cf]
        prev_tail = prev[-cf:]
        nxt_head = nxt[:cf]
        nxt_body = nxt[cf:]

        weights = torch.linspace(0.0, 1.0, cf, device=prev.device).view(cf, 1, 1, 1)
        blended = prev_tail * (1.0 - weights) + nxt_head * weights

        return torch.cat([prev_body, blended, nxt_body], dim=0)

    def run(self, clip_a, crossfade_frames, clip_b=None, clip_c=None, clip_d=None):
        clips = [c for c in (clip_a, clip_b, clip_c, clip_d) if c is not None]

        target_h, target_w = clip_a.shape[1], clip_a.shape[2]
        clips = [self._match_resolution(c, target_h, target_w) for c in clips]

        result = clips[0]
        seam_count = 0
        for nxt in clips[1:]:
            result = self._splice_pair(result, nxt, crossfade_frames)
            seam_count += 1

        summary = (
            f"Clips spliced:     {len(clips)}\n"
            f"Seams:             {seam_count}\n"
            f"Crossfade frames:  {crossfade_frames} per seam\n"
            f"Output resolution: {target_w}x{target_h}\n"
            f"Total frames:      {result.shape[0]}"
        )

        return (result, result.shape[0], summary)


NODE_CLASS_MAPPINGS = {
    "VideoConcatSpliceNode": VideoConcatSpliceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatSpliceNode": "Video Concat / Splice 🔗",
}
