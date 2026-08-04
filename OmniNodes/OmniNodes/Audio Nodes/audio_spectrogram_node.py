"""
TensorVizion ComfyUI Nodes
audio_spectrogram_node.py — Renders a Mel/FFT spectrogram as an IMAGE tensor.
"""

import numpy as np
import torch


class AudioSpectrogramNode:
    """
    Generates a spectrogram visualisation (Mel or linear FFT) from raw audio
    and outputs a ComfyUI IMAGE tensor. Pairs naturally with AudioWaveformNode.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "sample_rate":   ("INT",   {"default": 44100, "min": 8000,  "max": 192000, "step": 100}),
                "width":         ("INT",   {"default": 1024,  "min": 256,   "max": 4096,   "step": 64}),
                "height":        ("INT",   {"default": 512,   "min": 64,    "max": 2048,   "step": 64}),
                "fft_size":      ("INT",   {"default": 2048,  "min": 256,   "max": 8192,   "step": 256}),
                "hop_length":    ("INT",   {"default": 512,   "min": 64,    "max": 2048,   "step": 64}),
                "spectrogram_type": (["linear", "log_power"], {"default": "log_power"}),
                "colormap":      (["cyan_dark", "viridis", "hot", "grayscale"], {"default": "cyan_dark"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram_image",)
    FUNCTION = "render_spectrogram"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_colormap(self, magnitude_norm, colormap):
        """Map a (H, W) float32 [0,1] array to (H, W, 3) uint8."""
        m = magnitude_norm
        if colormap == "cyan_dark":
            r = (m * 0).astype(np.uint8)
            g = (m * 212).astype(np.uint8)
            b = (m * 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        elif colormap == "viridis":
            r = (np.clip(1.5 - np.abs(m * 4 - 3), 0, 1) * 255).astype(np.uint8)
            g = (np.clip(1.5 - np.abs(m * 4 - 2), 0, 1) * 255).astype(np.uint8)
            b = (np.clip(1.5 - np.abs(m * 4 - 1), 0, 1) * 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        elif colormap == "hot":
            r = (np.clip(m * 3,       0, 1) * 255).astype(np.uint8)
            g = (np.clip(m * 3 - 1,   0, 1) * 255).astype(np.uint8)
            b = (np.clip(m * 3 - 2,   0, 1) * 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)
        else:  # grayscale
            v = (m * 255).astype(np.uint8)
            return np.stack([v, v, v], axis=-1)

    # ------------------------------------------------------------------

    def render_spectrogram(
        self,
        audio_samples,
        sample_rate,
        width, height,
        fft_size, hop_length,
        spectrogram_type,
        colormap,
    ):
        # Unpack AUDIO dict or raw tensor
        if isinstance(audio_samples, dict):
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
        else:
            waveform = audio_samples

        if isinstance(waveform, torch.Tensor):
            samples = waveform.squeeze().float().cpu().numpy()
        else:
            samples = np.asarray(waveform, dtype=np.float32).squeeze()

        if samples.ndim == 2:
            samples = samples.mean(axis=0)

        # Hann window
        window = np.hanning(fft_size)
        n_frames = max(1, (len(samples) - fft_size) // hop_length + 1)

        # STFT
        spec = np.zeros((fft_size // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = samples[start: start + fft_size]
            if len(frame) < fft_size:
                frame = np.pad(frame, (0, fft_size - len(frame)))
            frame = frame * window
            fft_out = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft_out)

        if spectrogram_type == "log_power":
            spec = np.log1p(spec)

        # Normalise
        spec_min, spec_max = spec.min(), spec.max()
        if spec_max > spec_min:
            spec_norm = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = np.zeros_like(spec)

        # Resize to (height, width) using simple bilinear
        from PIL import Image as _PIL
        spec_img = _PIL.fromarray((spec_norm * 255).astype(np.uint8))
        spec_img = spec_img.resize((width, height), _PIL.BILINEAR)
        spec_resized = np.array(spec_img).astype(np.float32) / 255.0

        # Flip vertically so low freqs are at the bottom
        spec_resized = np.flipud(spec_resized)

        # Apply colour map
        rgb = self._apply_colormap(spec_resized, colormap)

        image = torch.from_numpy(rgb).float() / 255.0
        image = image.unsqueeze(0)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "AudioSpectrogramNode": AudioSpectrogramNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSpectrogramNode": "Audio Spectrogram 🎛️",
}
