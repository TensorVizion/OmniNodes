"""
TensorVizion ComfyUI Nodes
vae_encode_node.py — Encodes an IMAGE into a LATENT, the img2img entry
point. A thin, TensorVizion-branded wrapper around ComfyUI's own core
VAEEncode, so an existing image (loaded via Image Load, or the output of
any other node in this pack) can feed into Simple KSampler at a partial
`denoise` for img2img/refinement, instead of only starting from Empty
Latent Image.
"""

from nodes import VAEEncode as _CoreVAEEncode


class VAEEncodeNode:
    """
    Delegates straight to ComfyUI's own core `VAEEncode.encode()` — same
    behaviour as the standard node — with an added `summary` output
    reporting the resulting latent's batch size and spatial dimensions.
    Pair with Simple KSampler at `denoise` well below 1.0 (e.g. 0.3–0.6)
    to meaningfully refine/restyle the source image rather than
    regenerate it from scratch.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreVAEEncode.INPUT_TYPES()

    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("latent",  "summary")
    FUNCTION      = "run"

    def run(self, pixels, vae):
        core = _CoreVAEEncode()
        result = core.encode(vae, pixels)
        latent = result[0]

        samples = latent["samples"]
        b, c, h, w = samples.shape
        summary = f"Encoded to latent: batch={b}, channels={c}, {w}x{h} (latent space)"

        return (latent, summary)


NODE_CLASS_MAPPINGS = {
    "VAEEncodeNode": VAEEncodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEEncodeNode": "VAE Encode 🔒",
}
