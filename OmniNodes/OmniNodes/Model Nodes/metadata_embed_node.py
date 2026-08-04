"""
TensorVizion ComfyUI Nodes
metadata_embed_node.py — Writes custom key/value metadata (plus an optional
JSON blob of extra fields) into a saved PNG's tEXt chunks, without touching
pixel data. Useful for tagging generations with model, LoRA, seed-bank id,
client/job name, or anything else worth tracing back later.
"""

import os
import json
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin


class MetadataEmbedNode:
    """
    Saves `image` to disk under `output_dir/filename.png` (auto-incrementing
    on collision), embedding `key`/`value` as a PNG text chunk along with a
    timestamp. If `extra_json` is a valid JSON object, each of its top-level
    keys is embedded as its own text chunk too.
    """

    CATEGORY = "TensorVizion/Model"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "tv_output"}),
                "output_dir": ("STRING", {"default": "output/tensorvizion"}),
                "key": ("STRING", {"default": "tv_notes"}),
                "value": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "extra_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION      = "run"

    @staticmethod
    def _tensor_to_pil(img_tensor):
        arr = img_tensor.cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def run(self, image, filename, output_dir, key, value, extra_json=""):
        os.makedirs(output_dir, exist_ok=True)
        img = self._tensor_to_pil(image[0])

        meta = PngImagePlugin.PngInfo()
        meta.add_text(key, value)
        meta.add_text("tv_saved_at", datetime.now().isoformat())

        if extra_json.strip():
            try:
                parsed = json.loads(extra_json)
                for k, v in parsed.items():
                    meta.add_text(str(k), str(v))
            except Exception as e:
                meta.add_text("tv_extra_json_error", str(e))

        counter = 0
        path = os.path.join(output_dir, f"{filename}.png")
        while os.path.exists(path):
            counter += 1
            path = os.path.join(output_dir, f"{filename}_{counter:03d}.png")

        img.save(path, pnginfo=meta)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "MetadataEmbedNode": MetadataEmbedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetadataEmbedNode": "Metadata Embed 🏷️",
}
