"""
TensorVizion ComfyUI Nodes
audio_mixer_node.py — Mixes up to 4 audio tracks with per-track gain (dB),
pan, mute, and a master output gain. All pure NumPy/Torch — no deps.
"""

import numpy as np
import torch


class AudioMixerNode:
    """
    A 4-channel audio mixer. Each slot has:
      - gain_db  : per-track level in dB  (-60 to +12)
      - pan      : stereo position (-1.0 = full left, 0 = centre, 1.0 = full right)
      - mute     : silence the track without removing it from the chain

    Outputs a stereo (2-channel) mix plus a summary string.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_gain_db": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 12.0, "step": 0.5}),
                # Track 1
                "gain_db_1": ("FLOAT",   {"default": 0.0,  "min": -60.0, "max": 12.0,  "step": 0.5}),
                "pan_1":     ("FLOAT",   {"default": 0.0,  "min": -1.0,  "max": 1.0,   "step": 0.01}),
                "mute_1":    ("BOOLEAN", {"default": False}),
                # Track 2
                "gain_db_2": ("FLOAT",   {"default": 0.0,  "min": -60.0, "max": 12.0,  "step": 0.5}),
                "pan_2":     ("FLOAT",   {"default": 0.0,  "min": -1.0,  "max": 1.0,   "step": 0.01}),
                "mute_2":    ("BOOLEAN", {"default": False}),
                # Track 3
                "gain_db_3": ("FLOAT",   {"default": 0.0,  "min": -60.0, "max": 12.0,  "step": 0.5}),
                "pan_3":     ("FLOAT",   {"default": 0.0,  "min": -1.0,  "max": 1.0,   "step": 0.01}),
                "mute_3":    ("BOOLEAN", {"default": False}),
                # Track 4
                "gain_db_4": ("FLOAT",   {"default": 0.0,  "min": -60.0, "max": 12.0,  "step": 0.5}),
                "pan_4":     ("FLOAT",   {"default": 0.0,  "min": -1.0,  "max": 1.0,   "step": 0.01}),
                "mute_4":    ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
            }
        }

    RETURN_TYPES  = ("AUDIO",  "STRING")
    RETURN_NAMES  = ("audio",  "summary")
    FUNCTION      = "mix_tracks"

    # ------------------------------------------------------------------
    def _to_stereo_np(self, audio, target_len):
        """Unpack an AUDIO input to a (2, target_len) float32 numpy array."""
        if audio is None:
            return np.zeros((2, target_len), dtype=np.float32)

        if isinstance(audio, dict):
            wav = audio.get("waveform", audio.get("samples"))
        else:
            wav = audio

        if isinstance(wav, torch.Tensor):
            arr = wav.float().cpu().numpy()
        else:
            arr = np.asarray(wav, dtype=np.float32)

        arr = arr.squeeze()
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)   # mono → stereo
        elif arr.shape[0] == 1:
            arr = np.repeat(arr, 2, axis=0)

        # Trim or pad to target_len
        if arr.shape[1] > target_len:
            arr = arr[:, :target_len]
        elif arr.shape[1] < target_len:
            arr = np.pad(arr, ((0, 0), (0, target_len - arr.shape[1])))

        return arr.astype(np.float32)

    def _pan_gains(self, pan):
        """Constant-power panning."""
        angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
        return float(np.cos(angle)), float(np.sin(angle))   # left, right

    # ------------------------------------------------------------------
    def mix_tracks(
        self,
        master_gain_db,
        gain_db_1, pan_1, mute_1,
        gain_db_2, pan_2, mute_2,
        gain_db_3, pan_3, mute_3,
        gain_db_4, pan_4, mute_4,
        audio_1=None, audio_2=None, audio_3=None, audio_4=None,
    ):
        tracks  = [audio_1, audio_2, audio_3, audio_4]
        gains   = [gain_db_1, gain_db_2, gain_db_3, gain_db_4]
        pans    = [pan_1,     pan_2,     pan_3,     pan_4]
        mutes   = [mute_1,   mute_2,   mute_3,   mute_4]

        # Find longest track length
        max_len = 1
        for t in tracks:
            if t is None:
                continue
            if isinstance(t, dict):
                w = t.get("waveform", t.get("samples"))
            else:
                w = t
            if isinstance(w, torch.Tensor):
                n = w.shape[-1]
            else:
                n = np.asarray(w).shape[-1]
            max_len = max(max_len, n)

        mix_l = np.zeros(max_len, dtype=np.float32)
        mix_r = np.zeros(max_len, dtype=np.float32)

        active_tracks = []
        for i, (track, gain_db, pan, mute) in enumerate(zip(tracks, gains, pans, mutes), 1):
            if track is None or mute:
                status = "muted" if (track is not None and mute) else "empty"
                active_tracks.append(f"  Track {i}: {status}")
                continue

            arr       = self._to_stereo_np(track, max_len)
            lin_gain  = 10.0 ** (gain_db / 20.0)
            pan_l, pan_r = self._pan_gains(pan)

            mix_l += arr[0] * lin_gain * pan_l
            mix_r += arr[1] * lin_gain * pan_r
            active_tracks.append(f"  Track {i}: gain={gain_db:+.1f}dB  pan={pan:+.2f}")

        # Master gain
        master_lin = 10.0 ** (master_gain_db / 20.0)
        mix_l *= master_lin
        mix_r *= master_lin

        # Soft-clip output
        mix_l = np.tanh(mix_l)
        mix_r = np.tanh(mix_r)

        out_arr    = np.stack([mix_l, mix_r], axis=0)
        out_tensor = torch.from_numpy(out_arr).float()
        audio_out  = {"waveform": out_tensor, "sample_rate": 44100}

        summary = (
            f"Master gain:  {master_gain_db:+.1f} dB\n"
            + "\n".join(active_tracks)
        )

        return (audio_out, summary)


NODE_CLASS_MAPPINGS = {
    "AudioMixerNode": AudioMixerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioMixerNode": "Audio Mixer 🎚️",
}
