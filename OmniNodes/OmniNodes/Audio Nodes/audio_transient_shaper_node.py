"""
TensorVizion ComfyUI Nodes
audio_transient_shaper_node.py — Boosts or attenuates the attack and
sustain portions of an audio signal's transients independently of overall
volume. Distinct from Audio Sidechain Duck (level driven by a second
signal) and Audio Reverb (adds tail) — this reshapes the envelope of the
signal's own transients for punch/snap control.
"""

import numpy as np
import torch


class AudioTransientShaperNode:
    """
    Splits `audio_samples` into a fast (attack) envelope and a slow
    (sustain) envelope using two exponential followers, derives a transient
    component as the difference between them, then rebuilds the signal
    with independently scaled attack and sustain gain.

    attack_amount:  >1.0 boosts transient punch/snap, <1.0 softens it
    sustain_amount: >1.0 boosts body/tail, <1.0 tightens it
    attack_ms / sustain_ms: envelope follower time constants
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000}),
                "attack_amount": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 4.0, "step": 0.05}),
                "sustain_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 50.0, "step": 0.5}),
                "sustain_ms": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 500.0, "step": 5.0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_out", "summary")
    FUNCTION = "shape"

    def _envelope_follower(self, rectified, sample_rate, time_ms):
        # Single-pole exponential follower; time_ms controls response speed
        tau = max(1e-4, time_ms / 1000.0)
        alpha = float(np.exp(-1.0 / (tau * sample_rate)))
        env = np.zeros_like(rectified)
        prev = 0.0
        for i in range(len(rectified)):
            prev = alpha * prev + (1.0 - alpha) * rectified[i]
            env[i] = prev
        return env

    def shape(self, audio_samples, sample_rate, attack_amount, sustain_amount, attack_ms, sustain_ms):
        if isinstance(audio_samples, dict):
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
            input_sr = audio_samples.get("sample_rate", sample_rate)
        else:
            waveform = audio_samples
            input_sr = sample_rate

        is_tensor = isinstance(waveform, torch.Tensor)
        if is_tensor:
            samples = waveform.squeeze().float().cpu().numpy()
        else:
            samples = np.asarray(waveform, dtype=np.float32).squeeze()

        orig_shape = samples.shape
        mono = samples if samples.ndim == 1 else samples.mean(axis=0)

        rectified = np.abs(mono)
        fast_env = self._envelope_follower(rectified, input_sr, attack_ms)
        slow_env = self._envelope_follower(rectified, input_sr, sustain_ms)

        # Transient = energy present in the fast envelope but not yet in
        # the slow one; sustain = the slow envelope itself.
        transient = np.maximum(0.0, fast_env - slow_env)
        eps = 1e-8

        gain = 1.0 + (attack_amount - 1.0) * (transient / (fast_env + eps)) \
                   + (sustain_amount - 1.0) * (slow_env / (fast_env + eps))
        gain = np.clip(gain, 0.0, 8.0)

        shaped_mono = mono * gain
        peak = np.max(np.abs(shaped_mono)) + eps
        if peak > 1.0:
            shaped_mono = shaped_mono / peak

        if samples.ndim == 1:
            shaped = shaped_mono.astype(np.float32)
        else:
            # Apply the same gain curve to each channel
            shaped = (samples * gain[np.newaxis, :]).astype(np.float32)
            ch_peak = np.max(np.abs(shaped)) + eps
            if ch_peak > 1.0:
                shaped = shaped / ch_peak

        if is_tensor:
            out_tensor = torch.from_numpy(shaped).float()
            if out_tensor.shape != waveform.shape:
                out_tensor = out_tensor.reshape(waveform.shape) if out_tensor.numel() == waveform.numel() else out_tensor
            audio_out = {"waveform": out_tensor.unsqueeze(0) if out_tensor.ndim == 1 else out_tensor,
                          "sample_rate": input_sr}
        else:
            audio_out = {"waveform": shaped, "sample_rate": input_sr}

        summary = (
            f"Attack amount:  {attack_amount:.2f}x ({attack_ms:.1f}ms window)\n"
            f"Sustain amount: {sustain_amount:.2f}x ({sustain_ms:.1f}ms window)\n"
            f"Peak after shaping: {float(np.max(np.abs(shaped))):.3f}"
        )

        return (audio_out, summary)


NODE_CLASS_MAPPINGS = {
    "AudioTransientShaperNode": AudioTransientShaperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTransientShaperNode": "Audio Transient Shaper 🥊",
}
