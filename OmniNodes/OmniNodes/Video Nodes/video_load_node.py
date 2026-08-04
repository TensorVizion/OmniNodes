"""
TensorVizion ComfyUI Nodes
video_load_node.py — Loads a video file from disk and extracts its
frames as a standard ComfyUI IMAGE batch, with fps/frame-count metadata
and optional resampling to a target frame rate or frame cap. The load-side
companion to Video Save, completing the round trip.
"""

import os
import numpy as np
import torch


class VideoLoadNode:
    """
    Reads `video_path` and returns every frame (or every Nth frame, or a
    capped count) as an IMAGE batch. `target_fps` set to 0 keeps the
    source's native frame rate; set to a positive value to resample by
    dropping/duplicating frames to approximate that rate without a full
    optical-flow interpolation pass (use Video Frame Interpolate afterward
    for smooth retiming instead of simple drop/duplicate). `max_frames`
    caps how many frames are returned (0 = no cap), useful for previewing
    or limiting VRAM use on a long source video.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path":  ("STRING", {"default": ""}),
                "target_fps":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "max_frames":  ("INT", {"default": 0, "min": 0, "max": 100000}),
                "skip_first":  ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES  = ("IMAGE",  "FLOAT",     "INT",         "STRING")
    RETURN_NAMES  = ("images", "source_fps", "frame_count", "summary")
    FUNCTION      = "run"

    def run(self, video_path, target_fps, max_frames, skip_first):
        if not video_path or not os.path.exists(video_path):
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0.0, 0, f"[TensorVizion] File not found: {video_path}")

        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                msg = (
                    "[TensorVizion] 'imageio' is not installed. Install it with "
                    "'pip install imageio imageio-ffmpeg' to enable Video Load."
                )
                return (empty, 0.0, 0, msg)

        try:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            source_fps = float(meta.get("fps", 24.0))
        except Exception as e:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, 0.0, 0, f"[TensorVizion] Failed to open video: {e}")

        frames = []
        for i, frame in enumerate(reader):
            if i < skip_first:
                continue
            frames.append(np.asarray(frame))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        reader.close()

        if not frames:
            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, source_fps, 0, "[TensorVizion] No frames extracted.")

        # Simple drop/duplicate resample to approximate target_fps
        if target_fps > 0 and abs(target_fps - source_fps) > 1e-3:
            ratio = target_fps / source_fps
            n_out = max(1, int(round(len(frames) * ratio)))
            indices = np.linspace(0, len(frames) - 1, n_out).round().astype(int)
            frames = [frames[i] for i in indices]
            reported_fps = target_fps
        else:
            reported_fps = source_fps

        arr = np.stack(frames).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            arr = arr[..., :3]  # drop alpha if present

        image_tensor = torch.from_numpy(arr)

        summary = (
            f"Source FPS:    {source_fps:.3f}\n"
            f"Output FPS:    {reported_fps:.3f}\n"
            f"Frames loaded: {len(frames)}\n"
            f"Resolution:    {image_tensor.shape[2]}x{image_tensor.shape[1]}\n"
            f"Path:          {video_path}"
        )

        return (image_tensor, reported_fps, len(frames), summary)


NODE_CLASS_MAPPINGS = {
    "VideoLoadNode": VideoLoadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoadNode": "Video Load 📹",
}
