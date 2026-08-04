"""
TensorVizion ComfyUI Nodes
vae_loader_node.py — Loads a standalone VAE file, for workflows that need
a specific external VAE regardless of what a checkpoint bundles (e.g.
pairing an SDXL checkpoint with a community fine-tuned VAE, or using
taesd/taesdxl for fast previews). A thin, TensorVizion-branded wrapper
around ComfyUI's own core VAELoader so behaviour is guaranteed identical
to the standard node, including taesd/taesdxl/taesd3 special-casing.
"""

from nodes import VAELoader as _CoreVAELoader


class VAELoaderNode:
    """
    Loads `vae_name` from the `vae/` model folder using ComfyUI's own core
    VAE-loading logic (delegated directly to `nodes.VAELoader`, not
    reimplemented) — so any VAE format ComfyUI itself supports, including
    the special-cased `taesd`/`taesdxl`/`taesd3` fast-preview VAEs, works
    here identically. Use this instead of Simple SDXL Loader's bundled VAE
    when you want a specific external VAE independent of whichever
    checkpoint is loaded.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreVAELoader.INPUT_TYPES()

    RETURN_TYPES  = ("VAE",   "STRING")
    RETURN_NAMES  = ("vae",   "summary")
    FUNCTION      = "run"

    def run(self, vae_name):
        core = _CoreVAELoader()
        result = core.load_vae(vae_name)
        vae = result[0]
        summary = f"VAE loaded: {vae_name}"
        return (vae, summary)


NODE_CLASS_MAPPINGS = {
    "VAELoaderNode": VAELoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAELoaderNode": "VAE Loader 🗝️",
}
