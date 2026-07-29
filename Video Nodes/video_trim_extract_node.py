"""
TensorVizion ComfyUI Nodes
video_trim_extract_node.py — Extracts a contiguous frame range or a
specific list of frame indices from an IMAGE batch. The natural
downstream partner to Video Scene Detect: feed its cut_frame_indices
STRING straight into `explicit_indices` here to pull out a single scene
for isolated processing, or use range mode to trim dead frames off the
head/tail of a generated clip before Video Save.
"""

import torch


class VideoTrimExtractNode:
    """
    `mode`:
      range     — extracts frames from `start_frame` to `end_frame`
                  (inclusive; `end_frame` = -1 means "to the last frame")
      explicit  — extracts exactly the frames listed in
                  `explicit_indices`, a comma-separated string of frame
                  numbers (accepts Scene Detect's cut_frame_indices
                  output directly, or a hand-typed list like "0, 24, 48")

    Out-of-range indices are clamped/skipped rather than erroring, and the
    summary reports which requested indices (if any) fell outside the
    batch so a mismatched wiring mistake is easy to spot.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["range", "explicit"], {"default": "range"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 100000}),
                "explicit_indices": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "summary")
    FUNCTION = "run"

    def run(self, images, mode, start_frame, end_frame, explicit_indices):
        n = images.shape[0]

        if mode == "range":
            start = max(0, min(start_frame, n - 1))
            end = n - 1 if end_frame < 0 else min(end_frame, n - 1)
            if end < start:
                start, end = end, start
            out = images[start:end + 1]
            summary = (
                f"Mode: range\n"
                f"Source frames: {n}\n"
                f"Extracted:     frames {start}-{end} ({out.shape[0]} frames)"
            )
        else:
            skipped = []
            valid_idx = []
            for tok in explicit_indices.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    idx = int(float(tok))
                except ValueError:
                    skipped.append(tok)
                    continue
                if 0 <= idx < n:
                    valid_idx.append(idx)
                else:
                    skipped.append(str(idx))

            if not valid_idx:
                out = images[0:0]
            else:
                idx_tensor = torch.tensor(valid_idx, dtype=torch.long)
                out = images.index_select(0, idx_tensor)

            summary = (
                f"Mode: explicit\n"
                f"Source frames:    {n}\n"
                f"Requested:        {explicit_indices.strip() or '(empty)'}\n"
                f"Extracted:        {len(valid_idx)} frames\n"
                f"Skipped/invalid:  {', '.join(skipped) if skipped else 'none'}"
            )

        return (out, out.shape[0], summary)


NODE_CLASS_MAPPINGS = {
    "VideoTrimExtractNode": VideoTrimExtractNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTrimExtractNode": "Video Trim / Extract ✂️",
}
