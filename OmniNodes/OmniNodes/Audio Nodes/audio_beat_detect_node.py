"""
TensorVizion ComfyUI Nodes
audio_beat_detect_node.py — Detects beat/onset positions in audio and returns
timestamps, a beat-count, and estimated BPM.
"""

import numpy as np
import torch


class AudioBeatDetectNode:
    """
    Analyses audio samples to detect rhythmic onsets / beats.
    Returns beat timestamps (as a list string), beat count, and estimated BPM.
    Can drive frame-sync video effects by pairing with video nodes.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_samples": ("AUDIO",),
                "sample_rate":   ("INT",   {"default": 44100, "min": 8000,  "max": 192000}),
                "sensitivity":   ("FLOAT", {"default": 1.5,  "min": 0.5,   "max": 5.0,   "step": 0.1}),
                "min_bpm":       ("INT",   {"default": 60,   "min": 20,    "max": 200}),
                "max_bpm":       ("INT",   {"default": 200,  "min": 40,    "max": 300}),
            }
        }

    RETURN_TYPES  = ("STRING",      "INT",         "FLOAT",  "STRING")
    RETURN_NAMES  = ("beat_times",  "beat_count",  "bpm",    "summary")
    FUNCTION      = "detect_beats"

    def detect_beats(self, audio_samples, sample_rate, sensitivity, min_bpm, max_bpm):
        # Unpack
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

        # Energy-based onset detection
        hop      = int(sample_rate * 0.01)   # 10ms frames
        frame_n  = max(1, hop)
        n_frames = len(samples) // frame_n

        energy = np.array([
            np.sum(samples[i * frame_n: (i + 1) * frame_n] ** 2)
            for i in range(n_frames)
        ])

        # Local average energy in a ~430ms window
        win = max(1, int(0.43 * sample_rate / frame_n))
        local_avg = np.convolve(energy, np.ones(win) / win, mode="same")
        local_avg = np.maximum(local_avg, 1e-10)

        # Onset = energy significantly above local average
        ratio   = energy / local_avg
        onsets  = np.where(ratio > sensitivity)[0]

        # Merge onsets within min inter-beat interval
        min_ibi_frames = int((60.0 / max_bpm) * sample_rate / frame_n)
        filtered = []
        last     = -min_ibi_frames * 2
        for o in onsets:
            if o - last >= min_ibi_frames:
                filtered.append(o)
                last = o

        # Convert frame indices to seconds
        beat_times_sec = [float(o * frame_n / sample_rate) for o in filtered]
        beat_count     = len(beat_times_sec)

        # Estimate BPM from median inter-beat interval
        bpm = 0.0
        if beat_count > 1:
            ibi = np.diff(beat_times_sec)
            median_ibi = float(np.median(ibi))
            if median_ibi > 0:
                bpm = round(60.0 / median_ibi, 2)
                bpm = max(float(min_bpm), min(float(max_bpm), bpm))

        times_str = ", ".join(f"{t:.3f}" for t in beat_times_sec)
        summary = (
            f"Beats detected: {beat_count}\n"
            f"Estimated BPM:  {bpm:.2f}\n"
            f"Duration:       {len(samples)/sample_rate:.2f}s\n"
            f"Sensitivity:    {sensitivity}"
        )

        return (times_str, beat_count, bpm, summary)


NODE_CLASS_MAPPINGS = {
    "AudioBeatDetectNode": AudioBeatDetectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioBeatDetectNode": "Audio Beat Detect 🥁",
}
