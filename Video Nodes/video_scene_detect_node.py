"""
TensorVizion ComfyUI Nodes
video_scene_detect_node.py — Detects hard scene cuts in an IMAGE-batch
video sequence using frame-difference thresholding, and returns cut
timestamps/frame indices plus split-ready segment boundaries. Meant to run
before feeding raw footage into an I2V pipeline like FrameForge, so each
segment can be processed as its own consistent shot.
"""

import numpy as np
import torch


class VideoSceneDetectNode:
    """
    Computes a per-frame difference score (mean absolute pixel difference
    against the previous frame, downsampled first for speed) across
    `images` and flags a cut wherever that score exceeds `threshold`.

    `min_scene_frames` suppresses cuts that would create a scene shorter
    than that many frames (guards against false positives from motion
    blur or flash frames).

    Returns:
      cut_frame_indices — STRING, comma-separated frame indices where a
                           new scene starts (frame 0 always included)
      cut_timestamps     — STRING, comma-separated seconds (needs fps)
      scene_count         — INT
      report              — STRING summary
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_scene_frames": ("INT", {"default": 6, "min": 1, "max": 1000}),
                "downsample": ("INT", {"default": 32, "min": 4, "max": 256}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("cut_frame_indices", "cut_timestamps", "scene_count", "report")
    FUNCTION = "detect"

    def detect(self, images, fps, threshold, min_scene_frames, downsample):
        B = images.shape[0]

        if B < 2:
            return ("0", "0.000", 1, "[TensorVizion] Need 2+ frames to detect scene cuts; got 1.")

        # Downsample spatially for a fast, robust diff score
        small = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2), size=(downsample, downsample),
            mode="bilinear", align_corners=False
        )  # (B, C, ds, ds)

        gray = small.mean(dim=1)  # (B, ds, ds)
        frames_np = gray.cpu().numpy()

        diffs = np.abs(np.diff(frames_np, axis=0)).mean(axis=(1, 2))  # (B-1,)
        diffs = diffs / (diffs.max() + 1e-8) if diffs.max() > 0 else diffs

        raw_cuts = [0]
        last_cut = 0
        for i, d in enumerate(diffs):
            frame_idx = i + 1
            if d > threshold and (frame_idx - last_cut) >= min_scene_frames:
                raw_cuts.append(frame_idx)
                last_cut = frame_idx

        scene_count = len(raw_cuts)
        cut_timestamps_list = [round(idx / fps, 3) for idx in raw_cuts]

        cut_frame_indices = ", ".join(str(i) for i in raw_cuts)
        cut_timestamps = ", ".join(f"{t:.3f}" for t in cut_timestamps_list)

        scene_lengths = []
        for i in range(scene_count):
            start = raw_cuts[i]
            end = raw_cuts[i + 1] if i + 1 < scene_count else B
            scene_lengths.append(end - start)

        report_lines = [
            f"Total frames:  {B}",
            f"Scenes found:  {scene_count}",
            f"Threshold:     {threshold}",
            f"Min scene len: {min_scene_frames} frames",
            "",
            "Scenes:",
        ]
        for i, (start_idx, length) in enumerate(zip(raw_cuts, scene_lengths)):
            report_lines.append(
                f"  Scene {i}: frames {start_idx}-{start_idx + length - 1} "
                f"({length} frames, starts at {start_idx / fps:.2f}s)"
            )

        report = "\n".join(report_lines)

        return (cut_frame_indices, cut_timestamps, scene_count, report)


NODE_CLASS_MAPPINGS = {
    "VideoSceneDetectNode": VideoSceneDetectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSceneDetectNode": "Video Scene Detect 🎬",
}
