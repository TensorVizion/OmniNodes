"""
TensorVizion ComfyUI Nodes
quick_save_image_node.py — Saves a single image into ComfyUI's own managed
output tree (prefix_00001_ convention, gallery-visible immediately) with
a handful of OmniNodes-specific fields — seed/model/notes — embedded as
PNG text chunks. Fills the gap between the stock Save Image node (no
custom metadata) and Model Nodes/metadata_embed_node.py + Image Nodes/
custom_folder_batch_saver_node.py (both write outside the managed output
tree) — this is the "just save it with a bit of context, into the normal
place" node the pack didn't have. Delegates path/counter logic to
ComfyUI's own core SaveImage so files show up in the UI gallery exactly
like any other save.
"""

import numpy as np
from PIL import Image, PngImagePlugin

from nodes import SaveImage as _CoreSaveImage


class QuickSaveImageNode:
    """
    Saves `image` (first frame of the batch) via ComfyUI's own core
    SaveImage under `filename_prefix`, then re-writes that same file with
    `seed`, `model_name`, and `notes` added as PNG text chunks (core
    SaveImage's own PNGInfo, which already carries prompt/workflow JSON
    when metadata isn't disabled in ComfyUI's settings, is preserved and
    extended rather than replaced).
    """

    CATEGORY = "TensorVizion/Image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        base = _CoreSaveImage.INPUT_TYPES()
        base["required"]["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
        base["required"]["model_name"] = ("STRING", {"default": ""})
        base["optional"] = dict(base.get("optional", {}))
        base["optional"]["notes"] = ("STRING", {"multiline": True, "default": ""})
        return base

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "run"

    def run(self, images, filename_prefix, seed, model_name, notes="", prompt=None, extra_pnginfo=None):
        core = _CoreSaveImage()
        core_fn = getattr(core, core.FUNCTION)
        result = core_fn(images, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)

        saved_path = ""
        try:
            ui_images = result.get("ui", {}).get("images", [])
            if ui_images:
                import folder_paths
                info = ui_images[0]
                full_dir = folder_paths.get_output_directory() if info.get("type") == "output" else folder_paths.get_temp_directory()
                if info.get("subfolder"):
                    import os
                    full_dir = os.path.join(full_dir, info["subfolder"])
                import os
                saved_path = os.path.join(full_dir, info["filename"])

                img = Image.open(saved_path)
                meta = PngImagePlugin.PngInfo()
                for k, v in (img.text.items() if hasattr(img, "text") else []):
                    meta.add_text(k, v)
                meta.add_text("tv_seed", str(seed))
                meta.add_text("tv_model", model_name)
                if notes.strip():
                    meta.add_text("tv_notes", notes)
                img.save(saved_path, pnginfo=meta)
        except Exception:
            # Metadata enrichment is best-effort; the core save above has
            # already succeeded regardless of whether this step works.
            pass

        return (saved_path,)


NODE_CLASS_MAPPINGS = {
    "QuickSaveImageNode": QuickSaveImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickSaveImageNode": "Quick Save Image 💾 (TensorVizion)",
}
