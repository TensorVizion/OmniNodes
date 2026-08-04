"""
TensorVizion ComfyUI Nodes
vae_decode_node.py — Decodes a LATENT into a viewable/saveable IMAGE. A
thin, TensorVizion-branded wrapper around ComfyUI's own core VAEDecode,
completing the sampling loop: Empty Latent Image -> Simple KSampler ->
VAE Decode -> Image Save / Contact Sheet Maker / Video Save, entirely
within this pack.
"""

from nodes import VAEDecode as _CoreVAEDecode


class VAEDecodeNode:
    """
    Delegates straight to ComfyUI's own core `VAEDecode.decode()` — same
    behaviour as the standard node — with an added `summary` output
    reporting the resulting batch size and resolution.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreVAEDecode.INPUT_TYPES()

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image",  "summary")
    FUNCTION      = "run"

    def run(self, samples, vae):
        core = _CoreVAEDecode()
        result = core.decode(vae, samples)
        image = result[0]

        b, h, w, c = image.shape
        summary = f"Decoded {b} image(s) at {w}x{h}"

        return (image, summary)


NODE_CLASS_MAPPINGS = {
    "VAEDecodeNode": VAEDecodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeNode": "VAE Decode 🔓",
}
