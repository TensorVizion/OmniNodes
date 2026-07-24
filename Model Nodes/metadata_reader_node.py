"""
TensorVizion ComfyUI Nodes
metadata_reader_node.py — Reads back tEXt/iTXt metadata from any PNG on
disk, whether written by Metadata Embed or produced by another tool
(ComfyUI's own workflow embed, A1111, etc). The read-side companion to
Metadata Embed, and useful alongside LoRA Info Inspector / Model Info
Inspector when auditing where a generation actually came from.
"""

import json
from PIL import Image


class MetadataReaderNode:
    """
    Opens `file_path` and returns every text-chunk key/value pair found,
    both as a human-readable multiline string and as a JSON string for
    downstream parsing.
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata_raw", "metadata_json")
    FUNCTION      = "run"

    def run(self, file_path):
        import os
        if not file_path or not os.path.exists(file_path):
            return (f"[TensorVizion] File not found: {file_path}", "{}")

        img = Image.open(file_path)
        info = dict(img.info) if hasattr(img, "info") else {}

        clean = {}
        for k, v in info.items():
            try:
                clean[k] = v if isinstance(v, str) else str(v)
            except Exception:
                continue

        raw_lines = [f"{k}: {v}" for k, v in clean.items()]
        raw = "\n".join(raw_lines) if raw_lines else "[TensorVizion] No text metadata found."

        return (raw, json.dumps(clean, indent=2))


NODE_CLASS_MAPPINGS = {
    "MetadataReaderNode": MetadataReaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetadataReaderNode": "Metadata Reader 🔖",
}
