"""
TensorVizion ComfyUI Nodes
audio_normalize_node.py — Normalises audio samples to a target peak or RMS
level, with optional DC offset removal and soft-clip protection.
"""

import numpy as np
import torch


class AudioNormalizeNode:
    """
    Normalises audio amplitude to a target dBFS level.
    Supports peak normalisation and RMS normalisation, with optional
    DC-offset removal and soft-clip limiting at the output stage.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples":    ("AUDIO",),
                "target_db":        ("FLOAT", {"default": -3.0,  "min": -60.0, "max": 0.0,   "step": 0.5}),
                "normalize_mode":   (["peak", "rms"],),
                "remove_dc_offset": ("BOOLEAN", {"default": True}),
                "soft_clip":        ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("AUDIO",  "FLOAT",  "FLOAT",  "STRING")
    RETURN_NAMES  = ("audio",  "peak_db", "rms_db", "summary")
    FUNCTION      = "normalize_audio"

    def normalize_audio(self, audio_samples, target_db, normalize_mode, remove_dc_offset, soft_clip):
        # Unpack
        is_dict = isinstance(audio_samples, dict)
        if is_dict:
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
        else:
            waveform = audio_samples

        if isinstance(waveform, torch.Tensor):
            samples_np = waveform.float().cpu().numpy()
        else:
            samples_np = np.asarray(waveform, dtype=np.float32)

        working = samples_np.copy()

        # DC offset removal
        if remove_dc_offset:
            working = working - working.mean()

        # Measure levels
        peak = float(np.abs(working).max())
        rms  = float(np.sqrt(np.mean(working ** 2)))

        peak_db = 20.0 * np.log10(max(peak, 1e-10))
        rms_db  = 20.0 * np.log10(max(rms, 1e-10))

        # Compute gain
        target_linear = 10.0 ** (target_db / 20.0)
        if normalize_mode == "peak":
            ref = max(peak, 1e-10)
        else:  # rms
            ref = max(rms, 1e-10)

        gain    = target_linear / ref
        working = working * gain

        # Soft clip (tanh) to prevent hard clipping
        if soft_clip:
            working = np.tanh(working)

        # Package output
        out_tensor = torch.from_numpy(working).float()
        if is_dict:
            audio_out = {**audio_samples, "waveform": out_tensor}
        else:
            audio_out = out_tensor

        summary = (
            f"Mode:       {normalize_mode}\n"
            f"Gain:       {20*np.log10(max(gain,1e-10)):.2f} dB\n"
            f"Peak in:    {peak_db:.2f} dBFS\n"
            f"RMS in:     {rms_db:.2f} dBFS\n"
            f"Target:     {target_db:.1f} dBFS\n"
            f"DC remove:  {remove_dc_offset}\n"
            f"Soft clip:  {soft_clip}"
        )

        return (audio_out, float(peak_db), float(rms_db), summary)


NODE_CLASS_MAPPINGS = {
    "AudioNormalizeNode": AudioNormalizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNormalizeNode": "Audio Normalize 🔊",
}
