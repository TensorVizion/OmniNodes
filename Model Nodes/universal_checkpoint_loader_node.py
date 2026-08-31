"""
TensorVizion ComfyUI Nodes
universal_checkpoint_loader_node.py — Loads any checkpoint (SD1.5, SD2.x,
SDXL, SD3, Flux) from the standard `checkpoints/` folder in one node,
instead of the pack only offering the SDXL-specific Simple SDXL Loader.
Delegates to ComfyUI's own core CheckpointLoaderSimple for the actual
load — this node's value-add is a lightweight name-based family guess
surfaced in `summary` so users can sanity-check they picked the loader
appropriate for their model before wiring up sampling nodes downstream.
"""

from nodes import CheckpointLoaderSimple as _CoreCheckpointLoaderSimple


def _guess_family(name: str) -> str:
    lowered = name.lower()
    if "flux" in lowered:
        return "Flux"
    if "sd3" in lowered or "stable-diffusion-3" in lowered:
        return "SD3"
    if "xl" in lowered:
        return "SDXL"
    if "sd2" in lowered or "-v2" in lowered or "768" in lowered:
        return "SD2.x"
    return "SD1.5 (or unrecognized — verify manually)"


class UniversalCheckpointLoaderNode:
    """
    Loads `ckpt_name` from the `checkpoints/` model folder using ComfyUI's
    own core checkpoint-loading logic, returning MODEL/CLIP/VAE exactly
    like the stock Load Checkpoint node. `summary` includes a filename-
    based best-guess of the model family (SD1.5/SD2.x/SDXL/SD3/Flux) to
    help catch loading the wrong-family checkpoint before it hits a
    sampler expecting a different latent format.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreCheckpointLoaderSimple.INPUT_TYPES()

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES  = ("model", "clip", "vae", "summary")
    FUNCTION      = "run"

    def run(self, ckpt_name):
        core = _CoreCheckpointLoaderSimple()
        model, clip, vae = core.load_checkpoint(ckpt_name)[:3]
        family = _guess_family(ckpt_name)
        summary = f"Checkpoint loaded: {ckpt_name} (guessed family: {family})"
        return (model, clip, vae, summary)


NODE_CLASS_MAPPINGS = {
    "UniversalCheckpointLoaderNode": UniversalCheckpointLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalCheckpointLoaderNode": "Universal Checkpoint Loader 🌐 (TensorVizion)",
}
