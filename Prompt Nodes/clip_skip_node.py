"""
TensorVizion ComfyUI Nodes
clip_skip_node.py — Stops CLIP text-encoding early at
`-stop_at_clip_layer` layers from the end, an option most SD1.5-era
community checkpoints (esp. anime/NovelAI-derived models) were trained
expecting, and a basic expected utility this pack was previously missing
entirely. A thin, TensorVizion-branded wrapper around ComfyUI's own core
CLIPSetLastLayer.
"""

from nodes import CLIPSetLastLayer as _CoreCLIPSetLastLayer


class ClipSkipNode:
    """
    Returns a copy of `clip` with its last `abs(stop_at_clip_layer)`
    layers skipped during text encoding. `-1` (the default) is "no skip" /
    use the full CLIP stack; `-2` is the common "clip skip 2" setting many
    anime-style checkpoints were trained with.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreCLIPSetLastLayer.INPUT_TYPES()

    RETURN_TYPES  = ("CLIP",  "STRING")
    RETURN_NAMES  = ("clip",  "summary")
    FUNCTION      = "run"

    def run(self, clip, stop_at_clip_layer):
        core = _CoreCLIPSetLastLayer()
        result = core.set_last_layer(clip, stop_at_clip_layer)
        summary = f"CLIP skip applied: stop_at_clip_layer={stop_at_clip_layer}"
        return (result[0], summary)


NODE_CLASS_MAPPINGS = {
    "ClipSkipNode": ClipSkipNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipSkipNode": "CLIP Skip ✂️ (TensorVizion)",
}
