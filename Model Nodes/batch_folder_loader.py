"""
TensorVizion ComfyUI Nodes
batch_folder_loader.py — Lists every checkpoint file inside a folder under
ComfyUI's registered checkpoints/ search paths, optionally loading the first
match. Useful for auditing what's actually available in a subfolder without
opening a file browser, or as a quick "load whatever's first" utility node.
"""

import os
import folder_paths
import comfy.sd


class BatchFolderLoader:
    """
    Scans a subfolder of ComfyUI's checkpoints/ directory (not an arbitrary
    filesystem path) and lists every matching file. Optionally loads the
    first match's MODEL/CLIP/VAE so this can sit directly in a workflow
    instead of being a pure inspection node.

    `subfolder` is relative to one of ComfyUI's registered checkpoint search
    paths (e.g. "sdxl" if your models live in
    `ComfyUI/models/checkpoints/sdxl/`). Leave blank to scan the checkpoints
    root.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subfolder":          ("STRING", {"default": "", "multiline": False,
                                                     "placeholder": "subfolder under checkpoints/, blank = root"}),
                "filter_extensions":  ("STRING", {"default": ".safetensors,.ckpt"}),
                "load_first":         ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("MODEL",  "CLIP",  "VAE",  "STRING")
    RETURN_NAMES  = ("model",  "clip",  "vae",  "found_models")
    FUNCTION      = "load_batch"

    def load_batch(self, subfolder, filter_extensions, load_first):
        extensions = tuple(e.strip().lower() for e in filter_extensions.split(",") if e.strip())

        all_checkpoints = folder_paths.get_filename_list("checkpoints")

        # Filter to the requested subfolder (folder_paths returns paths with
        # forward slashes relative to the checkpoints root, regardless of OS)
        sub_norm = subfolder.strip().strip("/\\").replace("\\", "/")
        if sub_norm:
            matches = [
                f for f in all_checkpoints
                if f.replace("\\", "/").startswith(sub_norm + "/")
            ]
        else:
            matches = list(all_checkpoints)

        if extensions:
            matches = [f for f in matches if f.lower().endswith(extensions)]

        matches.sort()

        preview = matches[:10]
        listing = "\n".join(preview) if preview else "(no matching files)"
        if len(matches) > 10:
            listing += f"\n... and {len(matches) - 10} more"

        model, clip, vae = None, None, None

        if load_first and matches:
            model_path = folder_paths.get_full_path("checkpoints", matches[0])
            if model_path is not None:
                out = comfy.sd.load_checkpoint_guess_config(
                    model_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
                model, clip, vae = out[0], out[1], out[2]

        summary = f"📂 Found {len(matches)} model(s) in '{sub_norm or '(root)'}':\n{listing}"
        return (model, clip, vae, summary)


NODE_CLASS_MAPPINGS = {
    "BatchFolderLoader": BatchFolderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchFolderLoader": "Batch Folder Loader 📂",
}
