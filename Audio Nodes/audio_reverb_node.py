"""
TensorVizion ComfyUI Nodes
audio_reverb_node.py — Applies a convolution reverb / algorithmic reverb
to audio using a pure NumPy implementation. No external deps required.
"""

import numpy as np
import torch


class AudioReverbNode:
    """
    Applies reverb to audio via either a simple algorithmic Schroeder reverb
    (comb + allpass filters) or a synthetic impulse-response convolution.

    Controls: room_size, damping, wet/dry mix, pre-delay, and reverb type.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "sample_rate":   ("INT",   {"default": 44100, "min": 8000, "max": 192000}),
                "reverb_type":   (["algorithmic", "convolution"],),
                "room_size":     ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "damping":       ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "wet_level":     ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dry_level":     ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pre_delay_ms":  ("FLOAT", {"default": 20.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                "stereo_width":  ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("AUDIO",  "STRING")
    RETURN_NAMES  = ("audio",  "summary")
    FUNCTION      = "apply_reverb"

    # ------------------------------------------------------------------
    def _comb_filter(self, signal, delay_samples, feedback, damping):
        out    = np.zeros_like(signal)
        buf    = np.zeros(delay_samples, dtype=np.float32)
        pos    = 0
        filt   = 0.0
        for i in range(len(signal)):
            buf_out = buf[pos]
            filt    = buf_out * (1.0 - damping) + filt * damping
            buf[pos] = signal[i] + filt * feedback
            pos      = (pos + 1) % delay_samples
            out[i]   = buf_out
        return out

    def _allpass_filter(self, signal, delay_samples, feedback=0.5):
        out = np.zeros_like(signal)
        buf = np.zeros(delay_samples, dtype=np.float32)
        pos = 0
        for i in range(len(signal)):
            buf_out  = buf[pos]
            buf[pos] = signal[i] + buf_out * feedback
            pos      = (pos + 1) % delay_samples
            out[i]   = buf_out - signal[i]
        return out

    def _algorithmic_reverb(self, samples, sample_rate, room_size, damping, pre_delay_ms):
        # Pre-delay
        pre_delay_n = int(pre_delay_ms * sample_rate / 1000.0)
        if pre_delay_n > 0:
            samples = np.concatenate([np.zeros(pre_delay_n, dtype=np.float32), samples])

        # Schroeder comb delays (prime-ish multiples scaled by room_size)
        base_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        scaled      = [max(1, int(d * (0.5 + room_size * 0.5))) for d in base_delays]
        feedback    = 0.5 + room_size * 0.28

        comb_sum = np.zeros(len(samples), dtype=np.float32)
        for d in scaled:
            comb_sum += self._comb_filter(samples, d, feedback, damping)
        comb_sum /= len(scaled)

        # Two allpass stages
        ap1 = self._allpass_filter(comb_sum, 225)
        ap2 = self._allpass_filter(ap1,      556)

        return ap2[:len(samples) - pre_delay_n if pre_delay_n > 0 else len(samples)]

    def _convolution_reverb(self, samples, sample_rate, room_size, damping):
        # Synthesise a decaying IR
        decay_time = 0.5 + room_size * 3.5        # 0.5 – 4 seconds
        ir_len     = int(decay_time * sample_rate)
        rng        = np.random.default_rng(42)
        ir         = rng.standard_normal(ir_len).astype(np.float32)

        # Exponential decay envelope
        t          = np.linspace(0, decay_time, ir_len)
        decay_rate = 3.0 + damping * 7.0
        envelope   = np.exp(-decay_rate * t).astype(np.float32)

        # High-freq damping via simple LP
        alpha = damping * 0.8
        lp    = np.zeros_like(ir)
        prev  = 0.0
        for i in range(ir_len):
            prev  = prev * alpha + ir[i] * (1.0 - alpha)
            lp[i] = prev

        ir_shaped = lp * envelope
        ir_shaped /= (np.abs(ir_shaped).max() + 1e-8)

        # FFT convolution
        n_fft   = len(samples) + ir_len - 1
        out_fft = np.fft.rfft(samples, n=n_fft) * np.fft.rfft(ir_shaped, n=n_fft)
        wet     = np.fft.irfft(out_fft)[:len(samples)].astype(np.float32)
        return wet

    # ------------------------------------------------------------------
    def apply_reverb(
        self,
        audio_samples, sample_rate,
        reverb_type, room_size, damping,
        wet_level, dry_level, pre_delay_ms, stereo_width,
    ):
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
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        n_ch    = arr.shape[0]
        wet_chs = []

        for ch in range(n_ch):
            s = arr[ch]
            if reverb_type == "algorithmic":
                wet = self._algorithmic_reverb(s, sample_rate, room_size, damping, pre_delay_ms)
            else:
                wet = self._convolution_reverb(s, sample_rate, room_size, damping)
            wet_chs.append(wet)

        wet_arr = np.stack(wet_chs, axis=0)

        # Stereo width processing (for 2-ch audio)
        if n_ch == 2 and stereo_width > 0.0:
            mid  = (wet_arr[0] + wet_arr[1]) * 0.5
            side = (wet_arr[0] - wet_arr[1]) * 0.5
            wet_arr[0] = mid + side * (1.0 + stereo_width)
            wet_arr[1] = mid - side * (1.0 + stereo_width)

        # Normalise wet
        peak = np.abs(wet_arr).max()
        if peak > 1e-8:
            wet_arr = wet_arr / peak

        # Mix
        out_arr = arr * dry_level + wet_arr * wet_level
        peak_out = np.abs(out_arr).max()
        if peak_out > 1.0:
            out_arr /= peak_out

        if len(original_shape) == 1:
            out_arr = out_arr.squeeze(0)

        out_tensor = torch.from_numpy(out_arr.astype(np.float32)).float()
        audio_out  = {**audio_samples, "waveform": out_tensor} if is_dict else out_tensor

        summary = (
            f"Reverb type:   {reverb_type}\n"
            f"Room size:     {room_size:.2f}\n"
            f"Damping:       {damping:.2f}\n"
            f"Wet level:     {wet_level:.2f}\n"
            f"Dry level:     {dry_level:.2f}\n"
            f"Pre-delay:     {pre_delay_ms:.1f}ms\n"
            f"Stereo width:  {stereo_width:.2f}"
        )

        return (audio_out, summary)


NODE_CLASS_MAPPINGS = {
    "AudioReverbNode": AudioReverbNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioReverbNode": "Audio Reverb 🏛️",
}
