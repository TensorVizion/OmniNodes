"""
TensorVizion ComfyUI Nodes
custom_folder_batch_saver_node.py — Saves a batch to an arbitrary directory
(not ComfyUI's managed output root) using subfolder + prefix + zero-padded
persistent counter naming. Unlike Image Save (which follows ComfyUI's own
output/prefix_00001_ convention and embeds workflow metadata), this node is
built for exporting straight into a client deliverable folder, a dataset
directory, or anywhere else outside the standard output tree.
"""

import os
import re
import numpy as np
from PIL import Image


class CustomFolderBatchSaverNode:
    """
    Writes each image in `images` to `output_dir/subfolder/`, named
    `prefix_NNNN.ext`. The starting counter is read from existing files
    in that folder so repeated runs never collide with previous batches.
    """

    CATEGORY = "TensorVizion/Image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"default": "output/tensorvizion/batches"}),
                "subfolder": ("STRING", {"default": "batch_01"}),
                "prefix": ("STRING", {"default": "img"}),
                "pad_digits": ("INT", {"default": 4, "min": 1, "max": 10}),
                "format": (["png", "jpg"], {"default": "png"}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("saved_paths", "count")
    FUNCTION      = "run"

    @staticmethod
    def _tensor_to_pil(img_tensor):
        arr = img_tensor.cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def run(self, images, output_dir, subfolder, prefix, pad_digits, format, jpg_quality):
        full_dir = os.path.join(output_dir, subfolder)
        os.makedirs(full_dir, exist_ok=True)

        existing = [f for f in os.listdir(full_dir) if f.startswith(prefix + "_")]
        max_n = 0
        pat = re.compile(re.escape(prefix) + r"_(\d+)\.")
        for f in existing:
            m = pat.match(f)
            if m:
                max_n = max(max_n, int(m.group(1)))

        saved = []
        n = max_n
        for img_tensor in images:
            n += 1
            pil_img = self._tensor_to_pil(img_tensor)
            num = str(n).zfill(pad_digits)
            ext = "png" if format == "png" else "jpg"
            path = os.path.join(full_dir, f"{prefix}_{num}.{ext}")

            if format == "png":
                pil_img.save(path)
            else:
                pil_img.convert("RGB").save(path, quality=jpg_quality)

            saved.append(path)

        return ("\n".join(saved), len(saved))


NODE_CLASS_MAPPINGS = {
    "CustomFolderBatchSaverNode": CustomFolderBatchSaverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomFolderBatchSaverNode": "Custom Folder Batch Saver 📁",
}
