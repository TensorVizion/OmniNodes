"""
TensorVizion ComfyUI Nodes
audio_waveform_node.py — Renders an audio waveform as an IMAGE tensor.
"""

import numpy as np
import torch


class AudioWaveformNode:
    """
    Renders an audio waveform visualisation as a ComfyUI IMAGE tensor.
    Accepts raw audio samples (1-D float tensor) and outputs a waveform
    plot that can be passed to any image node downstream.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "width":  ("INT",  {"default": 1024, "min": 256,  "max": 4096, "step": 64}),
                "height": ("INT",  {"default": 256,  "min": 64,   "max": 2048, "step": 64}),
                "line_color_r": ("INT", {"default": 0,   "min": 0, "max": 255}),
                "line_color_g": ("INT", {"default": 212, "min": 0, "max": 255}),
                "line_color_b": ("INT", {"default": 255, "min": 0, "max": 255}),
                "bg_color_r":   ("INT", {"default": 5,  "min": 0, "max": 255}),
                "bg_color_g":   ("INT", {"default": 10, "min": 0, "max": 255}),
                "bg_color_b":   ("INT", {"default": 20, "min": 0, "max": 255}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("waveform_image",)
    FUNCTION = "render_waveform"

    def render_waveform(
        self,
        audio_samples,
        width, height,
        line_color_r, line_color_g, line_color_b,
        bg_color_r, bg_color_g, bg_color_b,
        line_thickness,
    ):
        # Normalise input — accept dict (ComfyUI AUDIO) or raw tensor
        if isinstance(audio_samples, dict):
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
        else:
            waveform = audio_samples

        if isinstance(waveform, torch.Tensor):
            samples = waveform.squeeze().float().cpu().numpy()
        else:
            samples = np.asarray(waveform, dtype=np.float32).squeeze()

        # Flatten to mono if stereo
        if samples.ndim == 2:
            samples = samples.mean(axis=0)

        # Normalise amplitude to [-1, 1]
        peak = np.abs(samples).max()
        if peak > 0:
            samples = samples / peak

        # Build pixel canvas (H, W, 3) — uint8
        bg = np.array([bg_color_r, bg_color_g, bg_color_b], dtype=np.uint8)
        canvas = np.full((height, width, 3), bg, dtype=np.uint8)

        line_color = np.array([line_color_r, line_color_g, line_color_b], dtype=np.uint8)
        half_h = height / 2.0
        n = len(samples)

        # Draw waveform column by column
        for x in range(width):
            # Map pixel column -> sample index
            idx = int(x / width * n)
            idx = min(idx, n - 1)
            amp = float(samples[idx])

            # Pixel y position (0 = top)
            y_center = int(half_h)
            y_amp    = int(amp * (half_h - line_thickness))
            y_top    = max(0, y_center - abs(y_amp))
            y_bot    = min(height - 1, y_center + abs(y_amp))

            for t in range(line_thickness):
                for y in range(y_top, y_bot + 1):
                    yt = min(height - 1, y + t)
                    canvas[yt, x] = line_color

        # Centre line
        mid = height // 2
        canvas[mid, :] = (line_color * 0.4).astype(np.uint8)

        # Convert to ComfyUI IMAGE format: (1, H, W, 3) float32 [0,1]
        image = torch.from_numpy(canvas).float() / 255.0
        image = image.unsqueeze(0)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "AudioWaveformNode": AudioWaveformNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioWaveformNode": "Audio Waveform 🎵",
}
