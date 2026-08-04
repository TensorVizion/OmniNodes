"""
TensorVizion ComfyUI Nodes
audio_to_latent_modulator_node.py — Analyzes an audio track's amplitude
envelope and converts it into per-frame FLOAT values and a batch of
audio-reactive LATENT noise, letting an audio track directly drive
animation/generation parameters (e.g. denoise strength or LoRA weight
pulsing with the beat). The audio-to-latent bridge the pack was missing:
Latent Interpolate and Prompt Weight Scheduler already produce per-frame
values from a frame index — this node produces the same kind of per-frame
value, but derived from real audio instead.
"""

import numpy as np
import torch


class AudioToLatentModulatorNode:
    """
    Splits `audio` into `num_frames` equal time-slices, measures the RMS
    (or peak) amplitude of each slice, and normalizes the result into a
    0–1 envelope curve — the classic "audio reactive" driver used in music
    visualizers. Outputs that curve as a comma-separated STRING (for
    reading into other nodes or scripts) plus the value at `frame_index`
    as a single FLOAT for direct wiring into any FLOAT input (denoise
    strength, LoRA weight, Prompt Weight Scheduler's weight range, etc).

    Optionally also outputs a LATENT batch of shape (num_frames, ...)
    where each frame's noise is scaled by that frame's envelope value —
    connect a template LATENT (any correctly-shaped empty latent) to get
    a batch of ready-to-use, audio-scaled noise for direct sampling.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":         ("AUDIO",),
                "num_frames":    ("INT",   {"default": 24,  "min": 1,   "max": 4096}),
                "frame_index":   ("INT",   {"default": 0,   "min": 0,   "max": 4095}),
                "measure":       (["rms", "peak"], {"default": "rms"}),
                "smoothing":     ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.95, "step": 0.01}),
                "gamma":         ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0,  "step": 0.05}),
                "seed":          ("INT",   {"default": 0,   "min": 0,   "max": 2**32 - 1}),
            },
            "optional": {
                "latent_template": ("LATENT",),
                "noise_strength":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("FLOAT",           "STRING",        "LATENT")
    RETURN_NAMES  = ("value_at_frame",  "envelope_csv",  "audio_scaled_latent")
    FUNCTION      = "run"

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
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # downmix to mono for envelope purposes
        return arr, sr

    def run(self, audio, num_frames, frame_index, measure, smoothing, gamma, seed,
            latent_template=None, noise_strength=1.0):
        mono, sr = self._unpack(audio)
        n_samples = len(mono)
        frame_index = min(frame_index, num_frames - 1)

        # Slice into num_frames equal chunks
        bounds = np.linspace(0, n_samples, num_frames + 1).astype(int)
        raw_env = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            chunk = mono[bounds[i]:bounds[i + 1]]
            if len(chunk) == 0:
                raw_env[i] = 0.0
            elif measure == "rms":
                raw_env[i] = float(np.sqrt(np.mean(chunk ** 2)))
            else:  # peak
                raw_env[i] = float(np.max(np.abs(chunk)))

        # Normalize to 0-1
        max_val = raw_env.max()
        env = raw_env / max_val if max_val > 1e-9 else raw_env

        # Temporal smoothing (simple EMA across frames)
        if smoothing > 0:
            smoothed = np.zeros_like(env)
            prev = env[0]
            for i in range(len(env)):
                prev = smoothing * prev + (1.0 - smoothing) * env[i]
                smoothed[i] = prev
            env = smoothed

        # Gamma shaping — emphasize peaks (gamma<1) or flatten them (gamma>1)
        env = np.clip(env, 0.0, 1.0) ** gamma

        value_at_frame = float(env[frame_index])
        envelope_csv = ",".join(f"{v:.4f}" for v in env)

        # Build audio-scaled latent batch if a template was provided
        if latent_template is not None:
            template_samples = latent_template["samples"]
            _, C, H, W = template_samples.shape
            rng = np.random.default_rng(seed)

            batch = np.zeros((num_frames, C, H, W), dtype=np.float32)
            for i in range(num_frames):
                noise = rng.standard_normal((C, H, W)).astype(np.float32)
                batch[i] = noise * env[i] * noise_strength

            out_latent = {"samples": torch.from_numpy(batch)}
        else:
            # No template: emit a minimal 1x1 placeholder so the output
            # socket is always valid even if unused downstream.
            out_latent = {"samples": torch.zeros((num_frames, 4, 1, 1), dtype=torch.float32)}

        return (value_at_frame, envelope_csv, out_latent)


NODE_CLASS_MAPPINGS = {
    "AudioToLatentModulatorNode": AudioToLatentModulatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToLatentModulatorNode": "Audio-to-Latent Modulator 🎧",
}
