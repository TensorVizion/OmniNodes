"""
TensorVizion ComfyUI Nodes
audio_pitch_shift_node.py — Shifts the pitch of audio by a given number of
semitones using a phase-vocoder approach (no external deps beyond numpy/torch).
"""

import numpy as np
import torch


class AudioPitchShiftNode:
    """
    Shifts audio pitch up or down by a semitone amount using a phase-vocoder
    implemented in pure NumPy. No librosa or soundfile dependency required.

    Positive semitones = pitch up. Negative = pitch down.
    Range: -24 to +24 semitones (2 octaves either direction).
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "semitones":     ("FLOAT", {"default": 0.0,  "min": -24.0, "max": 24.0, "step": 0.5}),
                "fft_size":      ("INT",   {"default": 2048, "min": 512,   "max": 8192, "step": 256}),
                "hop_length":    ("INT",   {"default": 512,  "min": 64,    "max": 2048, "step": 64}),
                "preserve_formants": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("AUDIO",  "STRING")
    RETURN_NAMES  = ("audio",  "summary")
    FUNCTION      = "pitch_shift"

    # ------------------------------------------------------------------
    def _phase_vocoder_shift(self, samples, shift_factor, fft_size, hop_length):
        """Time-stretch via phase vocoder then resample to achieve pitch shift."""
        n      = len(samples)
        window = np.hanning(fft_size)

        n_frames = (n - fft_size) // hop_length + 1
        stft = np.zeros((fft_size // 2 + 1, n_frames), dtype=complex)

        for i in range(n_frames):
            s = i * hop_length
            frame = samples[s: s + fft_size]
            if len(frame) < fft_size:
                frame = np.pad(frame, (0, fft_size - len(frame)))
            stft[:, i] = np.fft.rfft(frame * window)

        # Phase accumulation for time-stretch
        n_frames_out = int(np.ceil(n_frames / shift_factor))
        stft_out     = np.zeros((fft_size // 2 + 1, n_frames_out), dtype=complex)
        phase_acc    = np.angle(stft[:, 0])
        expect_phase = 2.0 * np.pi * hop_length * np.arange(fft_size // 2 + 1) / fft_size

        for i in range(n_frames_out):
            src_f = i * shift_factor
            src_i = int(src_f)
            src_i = min(src_i, n_frames - 1)

            mag         = np.abs(stft[:, src_i])
            phase_in    = np.angle(stft[:, src_i])
            delta_phase = phase_in - (phase_acc - expect_phase)
            delta_phase = delta_phase - 2.0 * np.pi * np.round(delta_phase / (2.0 * np.pi))
            phase_acc   = phase_acc + expect_phase + delta_phase
            stft_out[:, i] = mag * np.exp(1j * phase_acc)

        # ISTFT
        out_len    = (n_frames_out - 1) * hop_length + fft_size
        out_signal = np.zeros(out_len, dtype=np.float32)
        window_sum = np.zeros(out_len, dtype=np.float32)

        for i in range(n_frames_out):
            s     = i * hop_length
            frame = np.fft.irfft(stft_out[:, i])[:fft_size] * window
            out_signal[s: s + fft_size] += frame
            window_sum[s: s + fft_size] += window ** 2

        mask              = window_sum > 1e-8
        out_signal[mask] /= window_sum[mask]

        # Resample back to original length
        target_len    = int(len(out_signal) / shift_factor)
        indices       = np.linspace(0, len(out_signal) - 1, target_len)
        resampled     = np.interp(indices, np.arange(len(out_signal)), out_signal)

        return resampled.astype(np.float32)

    # ------------------------------------------------------------------
    def pitch_shift(self, audio_samples, semitones, fft_size, hop_length, preserve_formants):
        is_dict = isinstance(audio_samples, dict)
        if is_dict:
            waveform = audio_samples.get("waveform", audio_samples.get("samples"))
        else:
            waveform = audio_samples

        if isinstance(waveform, torch.Tensor):
            arr = waveform.float().cpu().numpy()
        else:
            arr = np.asarray(waveform, dtype=np.float32)

        original_shape = arr.shape
        # Work on each channel independently
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        shift_factor = 2.0 ** (semitones / 12.0)
        shifted_channels = []

        for ch in range(arr.shape[0]):
            ch_samples = arr[ch]
            shifted    = self._phase_vocoder_shift(ch_samples, shift_factor, fft_size, hop_length)
            # Match original length
            if len(shifted) > len(ch_samples):
                shifted = shifted[:len(ch_samples)]
            elif len(shifted) < len(ch_samples):
                shifted = np.pad(shifted, (0, len(ch_samples) - len(shifted)))
            shifted_channels.append(shifted)

        out_arr = np.stack(shifted_channels, axis=0)
        if len(original_shape) == 1:
            out_arr = out_arr.squeeze(0)

        out_tensor = torch.from_numpy(out_arr).float()

        if is_dict:
            audio_out = {**audio_samples, "waveform": out_tensor}
        else:
            audio_out = out_tensor

        direction = "up" if semitones > 0 else "down" if semitones < 0 else "unchanged"
        summary = (
            f"Pitch shift:      {semitones:+.1f} semitones ({direction})\n"
            f"Shift factor:     {shift_factor:.4f}x\n"
            f"FFT size:         {fft_size}\n"
            f"Hop length:       {hop_length}\n"
            f"Preserve formants:{preserve_formants}"
        )

        return (audio_out, summary)


NODE_CLASS_MAPPINGS = {
    "AudioPitchShiftNode": AudioPitchShiftNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioPitchShiftNode": "Audio Pitch Shift 🎼",
}
