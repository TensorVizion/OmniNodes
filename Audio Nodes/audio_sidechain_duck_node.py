"""
TensorVizion ComfyUI Nodes
audio_sidechain_duck_node.py — Classic sidechain compression: ducks the
volume of one audio track based on the amplitude envelope of another
(e.g. duck a music bed whenever a voiceover track is active). Pure
NumPy envelope-follower and gain-riding, no external DSP libraries.
"""

import numpy as np
import torch


class AudioSidechainDuckNode:
    """
    Follows the amplitude envelope of `trigger_audio` and applies an
    inverse gain reduction to `target_audio` wherever the trigger is
    loud — the same effect used to duck a music bed under a voice track,
    or pump a pad under a kick drum. `attack_ms`/`release_ms` shape how
    fast the duck engages/recovers; `duck_amount` sets how much gain is
    removed at full trigger loudness; `threshold` sets the trigger level
    below which no ducking happens at all.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_audio":  ("AUDIO",),
                "trigger_audio": ("AUDIO",),
                "duck_amount":   ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold":     ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attack_ms":     ("FLOAT", {"default": 15.0, "min": 0.1, "max": 500.0, "step": 0.1}),
                "release_ms":    ("FLOAT", {"default": 200.0, "min": 1.0, "max": 2000.0, "step": 1.0}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING")
    RETURN_NAMES  = ("audio", "summary")
    FUNCTION      = "duck"

    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(audio):
        if isinstance(audio, dict):
            wav = audio.get("waveform", audio.get("samples"))
            sr = audio.get("sample_rate", 44100)
        else:
            wav, sr = audio, 44100

        if isinstance(wav, torch.Tensor):
            arr = wav.float().cpu().numpy()
        else:
            arr = np.asarray(wav, dtype=np.float32)

        arr = arr.squeeze()
        if arr.ndim == 1:
            arr = arr[None, :]  # mono -> (1, N)
        return arr, sr

    @staticmethod
    def _envelope_follower(mono_signal, sr, attack_ms, release_ms):
        """One-pole attack/release envelope follower on rectified signal."""
        rectified = np.abs(mono_signal)
        attack_coef = np.exp(-1.0 / (sr * (attack_ms / 1000.0) + 1e-9))
        release_coef = np.exp(-1.0 / (sr * (release_ms / 1000.0) + 1e-9))

        env = np.zeros_like(rectified)
        prev = 0.0
        for i in range(len(rectified)):
            x = rectified[i]
            coef = attack_coef if x > prev else release_coef
            prev = coef * prev + (1.0 - coef) * x
            env[i] = prev
        return env

    def duck(self, target_audio, trigger_audio, duck_amount, threshold, attack_ms, release_ms):
        target_arr, sr = self._unpack(target_audio)
        trigger_arr, _ = self._unpack(trigger_audio)

        n_target = target_arr.shape[-1]
        n_trigger = trigger_arr.shape[-1]
        n = min(n_target, n_trigger) if n_trigger > 0 else n_target

        if n_trigger == 0:
            gain_curve = np.ones(n_target, dtype=np.float32)
        else:
            trigger_mono = trigger_arr.mean(axis=0)[:n]
            env = self._envelope_follower(trigger_mono, sr, attack_ms, release_ms)

            env_max = float(env.max()) if env.max() > 0 else 1.0
            env_norm = env / env_max

            above = np.clip((env_norm - threshold) / max(1e-6, (1.0 - threshold)), 0.0, 1.0)
            reduction = above * duck_amount
            gain_curve = (1.0 - reduction).astype(np.float32)

            if n < n_target:
                pad = np.ones(n_target - n, dtype=np.float32) * gain_curve[-1]
                gain_curve = np.concatenate([gain_curve, pad])

        ducked = target_arr * gain_curve[None, :]
        ducked = np.tanh(ducked)  # soft-clip safety

        out_tensor = torch.from_numpy(ducked.astype(np.float32))
        audio_out = {"waveform": out_tensor, "sample_rate": sr}

        avg_reduction_db = 20 * np.log10(max(1e-6, float(gain_curve.mean())))
        min_gain_db = 20 * np.log10(max(1e-6, float(gain_curve.min())))
        summary = (
            f"Duck amount:     {duck_amount:.2f}\n"
            f"Threshold:       {threshold:.2f}\n"
            f"Attack/Release:  {attack_ms:.1f}ms / {release_ms:.1f}ms\n"
            f"Avg gain change: {avg_reduction_db:+.2f} dB\n"
            f"Deepest duck:    {min_gain_db:+.2f} dB"
        )

        return (audio_out, summary)


NODE_CLASS_MAPPINGS = {
    "AudioSidechainDuckNode": AudioSidechainDuckNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSidechainDuckNode": "Audio Sidechain Duck 🦆",
}
