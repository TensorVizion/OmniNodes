"""
TensorVizion ComfyUI Nodes
audio_stem_splitter_node.py — Splits a track into bass/mid/high stems via
FFT band-splitting. Not a source-separation model (no vocal/instrument
isolation) — a fast, dependency-free frequency-domain split for quick
remixing, isolating a frequency range, or feeding separate bands into
downstream effects (e.g. sidechain the bass only, reverb the highs only).
"""

import numpy as np
import torch


class AudioStemSplitterNode:
    """
    Runs an FFT on the input audio and splits it into three frequency
    bands using configurable crossover points, then inverse-FFTs each
    band back into its own audio stream. Crossovers are set as two
    frequencies in Hz: everything below `low_crossover_hz` is the bass
    band, everything between the two crossovers is the mid band, and
    everything above `high_crossover_hz` is the high band. A short
    raised-cosine taper is applied at each crossover to avoid the harsh
    ringing a hard brick-wall cutoff would introduce.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":            ("AUDIO",),
                "low_crossover_hz":  ("FLOAT", {"default": 250.0,  "min": 20.0,  "max": 2000.0,  "step": 10.0}),
                "high_crossover_hz": ("FLOAT", {"default": 4000.0, "min": 500.0, "max": 18000.0, "step": 10.0}),
                "taper_fraction":    ("FLOAT", {"default": 0.15,   "min": 0.0,   "max": 0.5,     "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES  = ("bass",  "mid",   "high",  "summary")
    FUNCTION      = "split"

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
            arr = arr[None, :]
        return arr, sr

    @staticmethod
    def _band_mask(freqs, lo, hi, taper_fraction):
        """Raised-cosine (Tukey-style) soft mask between lo and hi Hz."""
        mask = np.zeros_like(freqs, dtype=np.float32)
        band = (freqs >= lo) & (freqs <= hi) if hi is not None else (freqs >= lo)
        if hi is None:
            band = freqs >= lo
        mask[band] = 1.0

        width = max(1e-6, (hi - lo) if hi is not None else freqs.max() - lo)
        taper_hz = width * taper_fraction

        # Soften the low edge
        if lo > 0 and taper_hz > 0:
            edge = (freqs >= lo - taper_hz) & (freqs < lo)
            t = (freqs[edge] - (lo - taper_hz)) / taper_hz
            mask[edge] = 0.5 - 0.5 * np.cos(np.pi * t)

        # Soften the high edge
        if hi is not None and taper_hz > 0:
            edge = (freqs > hi) & (freqs <= hi + taper_hz)
            t = (freqs[edge] - hi) / taper_hz
            mask[edge] = 0.5 + 0.5 * np.cos(np.pi * t)

        return mask

    def split(self, audio, low_crossover_hz, high_crossover_hz, taper_fraction):
        arr, sr = self._unpack(audio)
        n_channels, n_samples = arr.shape

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

        mask_bass = self._band_mask(freqs, 0.0, low_crossover_hz, taper_fraction)
        mask_mid  = self._band_mask(freqs, low_crossover_hz, high_crossover_hz, taper_fraction)
        mask_high = self._band_mask(freqs, high_crossover_hz, None, taper_fraction)

        bass_out = np.zeros_like(arr)
        mid_out  = np.zeros_like(arr)
        high_out = np.zeros_like(arr)

        for c in range(n_channels):
            spectrum = np.fft.rfft(arr[c])
            bass_out[c] = np.fft.irfft(spectrum * mask_bass, n=n_samples)
            mid_out[c]  = np.fft.irfft(spectrum * mask_mid,  n=n_samples)
            high_out[c] = np.fft.irfft(spectrum * mask_high, n=n_samples)

        def pack(a):
            return {"waveform": torch.from_numpy(a.astype(np.float32)), "sample_rate": sr}

        summary = (
            f"Bass band:  0 Hz – {low_crossover_hz:.0f} Hz\n"
            f"Mid band:   {low_crossover_hz:.0f} Hz – {high_crossover_hz:.0f} Hz\n"
            f"High band:  {high_crossover_hz:.0f} Hz and up\n"
            f"Taper:      {taper_fraction * 100:.0f}% of each band's width\n"
            f"Note: FFT band-split, not ML source separation — "
            f"isolates frequency ranges, not instruments/vocals."
        )

        return (pack(bass_out), pack(mid_out), pack(high_out), summary)


NODE_CLASS_MAPPINGS = {
    "AudioStemSplitterNode": AudioStemSplitterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioStemSplitterNode": "Audio Stem Splitter (Freq Band) 🍰",
}
