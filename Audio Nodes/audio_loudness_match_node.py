"""
TensorVizion ComfyUI Nodes
audio_loudness_match_node.py — Matches `audio`'s perceived loudness to a
`reference` clip's loudness, rather than to a fixed absolute target the
way Audio Normalize does. The audio equivalent of Video Color Match
(match-to-reference-frame) — useful when muxing generated audio/dialogue
against a reference track so levels don't visibly jump between clips.
"""

import numpy as np
import torch


class AudioLoudnessMatchNode:
    """
    Measures `reference`'s loudness (RMS-based dBFS, optionally with a
    simple perceptual weighting pass — see `use_perceptual_weighting`)
    and applies gain to `audio` so its loudness matches. This differs
    from Audio Normalize in kind, not just degree: Normalize targets a
    number YOU pick; this node targets whatever ANOTHER clip's actual
    loudness happens to be, which is the more common need when
    combining multiple audio sources that should sit at the same
    perceived level.

    `use_perceptual_weighting` applies a simple A-weighting-style
    frequency emphasis (boosting the 1-5kHz range where human hearing
    is most sensitive) before measuring RMS — a closer approximation to
    perceived loudness than flat RMS, though still not a full ITU-R
    BS.1770 LUFS implementation. Good enough for "make these two clips
    feel similarly loud," not intended for broadcast loudness compliance.

    `max_gain_db` caps how much gain can be applied in either direction,
    since matching a near-silent reference against a loud clip (or vice
    versa) could otherwise call for an extreme, noise-amplifying gain.
    """

    CATEGORY = "TensorVizion/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "reference": ("AUDIO",),
                "use_perceptual_weighting": ("BOOLEAN", {"default": True}),
                "max_gain_db": ("FLOAT", {"default": 18.0, "min": 1.0, "max": 40.0, "step": 0.5}),
                "soft_clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "applied_gain_db", "summary")
    FUNCTION = "match"

    def _unpack(self, audio_dict_or_tensor):
        is_dict = isinstance(audio_dict_or_tensor, dict)
        if is_dict:
            waveform = audio_dict_or_tensor.get("waveform", audio_dict_or_tensor.get("samples"))
        else:
            waveform = audio_dict_or_tensor
        if isinstance(waveform, torch.Tensor):
            samples_np = waveform.float().cpu().numpy()
        else:
            samples_np = np.asarray(waveform, dtype=np.float32)
        return samples_np, is_dict

    def _perceptual_rms(self, samples_np, use_weighting):
        working = samples_np.copy()
        if use_weighting:
            # Simple 3-tap high-shelf-ish emphasis on higher-frequency
            # content as a cheap stand-in for full A-weighting — boosts
            # the difference between adjacent samples (a crude high-pass
            # proxy) and blends it back in, rather than a full FFT-based
            # frequency-weighting curve.
            diff = np.diff(working, axis=-1, prepend=working[..., :1])
            working = working + 0.3 * diff
        return float(np.sqrt(np.mean(working ** 2)) + 1e-10)

    def match(self, audio, reference, use_perceptual_weighting, max_gain_db, soft_clip):
        audio_np, is_dict = self._unpack(audio)
        ref_np, _ = self._unpack(reference)

        audio_level = self._perceptual_rms(audio_np, use_perceptual_weighting)
        ref_level = self._perceptual_rms(ref_np, use_perceptual_weighting)

        gain_db = 20.0 * np.log10(ref_level / audio_level)
        clamped_gain_db = float(np.clip(gain_db, -max_gain_db, max_gain_db))
        gain_linear = 10.0 ** (clamped_gain_db / 20.0)

        matched = audio_np * gain_linear
        if soft_clip:
            matched = np.tanh(matched)

        out_tensor = torch.from_numpy(matched).float()
        if is_dict:
            audio_out = {**audio, "waveform": out_tensor}
        else:
            audio_out = out_tensor

        clamped_note = " (clamped to max_gain_db)" if abs(gain_db - clamped_gain_db) > 0.01 else ""
        summary = (
            f"Audio level:     {20*np.log10(audio_level):.2f} dBFS (perceptual)\n"
            f"Reference level: {20*np.log10(ref_level):.2f} dBFS (perceptual)\n"
            f"Requested gain:  {gain_db:+.2f} dB\n"
            f"Applied gain:    {clamped_gain_db:+.2f} dB{clamped_note}\n"
            f"Perceptual weighting: {use_perceptual_weighting}"
        )
        return (audio_out, clamped_gain_db, summary)


NODE_CLASS_MAPPINGS = {
    "AudioLoudnessMatchNode": AudioLoudnessMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLoudnessMatchNode": "Audio Loudness Match 🎚️",
}
