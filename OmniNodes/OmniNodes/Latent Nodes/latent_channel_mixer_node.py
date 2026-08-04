"""
TensorVizion ComfyUI Nodes
latent_channel_mixer_node.py — Per-channel gain/offset remixing and channel
reordering for LATENT tensors. A latent-space analogue of an RGB channel mixer.
"""

import torch


class LatentChannelMixerNode:
    """
    Applies independent gain and offset to each channel of a latent tensor,
    with an optional channel reorder/swap pass afterwards.

    Works with any channel count (4 for SD1.5/SDXL, 16 for SD3/Flux, etc.) —
    gains/offsets are given as comma-separated strings and cycled to fit
    however many channels the latent actually has.

    channel_order lets you swap or duplicate channels, e.g. "1,0,2,3" swaps
    channels 0 and 1. Leave it blank to skip reordering.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "gains": ("STRING", {"default": "1.0,1.0,1.0,1.0", "multiline": False,
                                       "placeholder": "comma-separated per-channel gain, cycles if short"}),
                "offsets": ("STRING", {"default": "0.0,0.0,0.0,0.0", "multiline": False,
                                         "placeholder": "comma-separated per-channel offset"}),
                "channel_order": ("STRING", {"default": "", "multiline": False,
                                               "placeholder": "e.g. 1,0,2,3  (blank = no reorder)"}),
                "normalize_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "mix_info")
    FUNCTION = "mix_channels"

    # ------------------------------------------------------------------
    def _parse_floats(self, text, count, default_val):
        try:
            values = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
        except ValueError:
            values = []
        if not values:
            values = [default_val]
        # Cycle to fill exactly `count` entries
        return [values[i % len(values)] for i in range(count)]

    def _parse_order(self, text, count):
        if not text.strip():
            return None
        try:
            order = [int(v.strip()) for v in text.split(",") if v.strip() != ""]
        except ValueError:
            return None
        if len(order) != count or any(i < 0 or i >= count for i in order):
            return None
        return order

    # ------------------------------------------------------------------
    def mix_channels(self, latent, gains, offsets, channel_order, normalize_output):
        samples = latent["samples"].clone()  # (B, C, H, W)
        B, C, H, W = samples.shape

        gain_list = self._parse_floats(gains, C, 1.0)
        offset_list = self._parse_floats(offsets, C, 0.0)

        for c in range(C):
            samples[:, c] = samples[:, c] * gain_list[c] + offset_list[c]

        order = self._parse_order(channel_order, C)
        reordered = False
        if order is not None:
            idx = torch.tensor(order, device=samples.device, dtype=torch.long)
            samples = samples.index_select(1, idx)
            reordered = True

        if normalize_output:
            peak = samples.abs().amax()
            if peak > 1e-8:
                samples = samples / peak

        mix_info = (
            f"Channels:    {C}\n"
            f"Gains:       {['%.2f' % g for g in gain_list]}\n"
            f"Offsets:     {['%.2f' % o for o in offset_list]}\n"
            f"Reordered:   {reordered}"
            + (f" -> {order}" if reordered else "")
            + f"\nNormalized:  {normalize_output}"
        )

        return ({"samples": samples}, mix_info)


NODE_CLASS_MAPPINGS = {
    "LatentChannelMixerNode": LatentChannelMixerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentChannelMixerNode": "Latent Channel Mixer 🎚️",
}
