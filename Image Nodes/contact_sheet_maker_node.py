"""
TensorVizion ComfyUI Nodes
contact_sheet_maker_node.py — Arranges a batch of images into a single grid
contact sheet, with configurable columns, padding, and background color.
Fast visual QA for large batches, LoRA training sets, or wildcard sweeps —
pairs well with Batch Counter and Wildcard Loader upstream.
"""

import numpy as np
from PIL import Image
import torch


class ContactSheetMakerNode:
    """
    Lays out every image in `images` into a `columns`-wide grid, scaling
    each to fit a `thumb_size` square (centered if not already square) with
    `padding` px between cells, on a solid `bg_color` background.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 4, "min": 1, "max": 32}),
                "thumb_size": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 128}),
                "bg_color": ("STRING", {"default": "#141414"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("contact_sheet",)
    FUNCTION      = "run"

    @staticmethod
    def _tensor_to_pil(img_tensor):
        arr = img_tensor.cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _pil_to_tensor(img):
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]

    @staticmethod
    def _parse_color(s):
        s = s.strip().lstrip("#")
        if len(s) == 6:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        return (20, 20, 20)

    def run(self, images, columns, thumb_size, padding, bg_color):
        n = images.shape[0]
        cols = max(1, columns)
        rows = (n + cols - 1) // cols

        bg = self._parse_color(bg_color)
        sheet_w = cols * thumb_size + (cols + 1) * padding
        sheet_h = rows * thumb_size + (rows + 1) * padding
        sheet = Image.new("RGB", (sheet_w, sheet_h), bg)

        for i in range(n):
            img = self._tensor_to_pil(images[i])
            img.thumbnail((thumb_size, thumb_size))

            col = i % cols
            row = i // cols
            x = padding + col * (thumb_size + padding)
            y = padding + row * (thumb_size + padding)

            offset_x = x + (thumb_size - img.width) // 2
            offset_y = y + (thumb_size - img.height) // 2
            sheet.paste(img, (offset_x, offset_y))

        return (self._pil_to_tensor(sheet),)


NODE_CLASS_MAPPINGS = {
    "ContactSheetMakerNode": ContactSheetMakerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContactSheetMakerNode": "Contact Sheet Maker 🗺️",
}
