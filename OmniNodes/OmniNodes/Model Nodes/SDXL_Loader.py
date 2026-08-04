"""
TensorVizion ComfyUI Nodes
SDXL_Loader.py — Loads an SDXL (or any) checkpoint and its MODEL/CLIP/VAE in
one node, with an option to auto-load a matching external VAE file when the
checkpoint doesn't bundle its own (common for some SDXL community finetunes).
"""

import folder_paths
import comfy.sd
import comfy.utils


class SimpleSDXLLoader:
    """
    Loads a checkpoint's MODEL, CLIP, and VAE in a single node — a thin,
    opinionated wrapper around ComfyUI's standard checkpoint loader that adds
    one convenience: if the checkpoint has no baked-in VAE, it can
    automatically look for a same-named `*.vae.safetensors` file in your
    vae/ folder instead of leaving VAE empty.

    Set `vae_name` to override auto-detection and force a specific external
    VAE regardless of what the checkpoint bundles.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_auto":   ("BOOLEAN", {"default": True}),
                "vae_name":   (["None"] + folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES  = ("model", "clip", "vae", "summary")
    FUNCTION      = "load_sdxl"

    def load_sdxl(self, model_name, vae_auto, vae_name):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        if model_path is None:
            raise FileNotFoundError(f"[SimpleSDXLLoader] Checkpoint not found: {model_name}")

        out = comfy.sd.load_checkpoint_guess_config(
            model_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        # load_checkpoint_guess_config returns (model, clip, vae, clipvision)
        model, clip, vae = out[0], out[1], out[2]

        vae_source = "checkpoint (bundled)"

        # Explicit override takes priority over auto-detection
        if vae_name != "None":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            if vae_path is not None:
                sd = comfy.utils.load_torch_file(vae_path)
                vae = comfy.sd.VAE(sd=sd)
                vae_source = f"explicit override ({vae_name})"

        elif vae_auto and vae is None:
            # Try to find a same-named external VAE next to the checkpoint's
            # own name, e.g. "MyModel.safetensors" -> "MyModel.vae.safetensors"
            guess_name = model_name.rsplit(".", 1)[0] + ".vae.safetensors"
            guess_path = folder_paths.get_full_path("vae", guess_name)
            if guess_path is not None:
                sd = comfy.utils.load_torch_file(guess_path)
                vae = comfy.sd.VAE(sd=sd)
                vae_source = f"auto-matched ({guess_name})"
            else:
                vae_source = "none found (checkpoint has no VAE and no match in vae/)"

        summary = (
            f"Checkpoint: {model_name}\n"
            f"VAE source: {vae_source}"
        )

        return (model, clip, vae, summary)


NODE_CLASS_MAPPINGS = {
    "SimpleSDXLLoader": SimpleSDXLLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleSDXLLoader": "Simple SDXL Loader 📀",
}
