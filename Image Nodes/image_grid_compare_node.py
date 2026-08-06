"""
TensorVizion ComfyUI Nodes
image_grid_compare_node.py — Lays out a batch of images side-by-side
with a text label under each, for comparing sampler settings, LoRA
strengths, seeds, or any other sweep. Genuinely different from Contact
Sheet Maker (unlabeled thumbnail tiling for browsing a batch) — this is
built specifically for labeled comparison, where knowing WHICH image is
WHICH setting is the entire point.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class ImageGridCompareNode:
    """
    Lays out every image in `images` into a `columns`-wide grid (same
    grid concept as Contact Sheet Maker), but draws `labels` underneath
    each cell — `labels` is a newline-separated list, one label per
    image, matched by position (label N under image N). If fewer
    labels than images are given, remaining cells get an auto-numbered
    label ("#4", "#5", ...) instead of failing.

    `highlight_index` optionally draws a colored border around one
    specific cell (0-indexed) — useful for marking "this is the
    setting I actually used" among a sweep of alternatives.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "labels": ("STRING", {"default": "", "multiline": True}),
                "columns": ("INT", {"default": 4, "min": 1, "max": 32}),
                "thumb_size": ("INT", {"default": 256, "min": 32, "max": 2048}),
                "padding": ("INT", {"default": 12, "min": 0, "max": 128}),
                "label_height": ("INT", {"default": 32, "min": 16, "max": 128}),
                "bg_color": ("STRING", {"default": "20,20,20"}),
                "label_color": ("STRING", {"default": "255,255,255"}),
                "highlight_index": ("INT", {"default": -1, "min": -1, "max": 4096}),
                "highlight_color": ("STRING", {"default": "0,255,120"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("grid_image", "summary")
    FUNCTION = "process"

    def _parse_color(self, s, default=(255, 255, 255)):
        try:
            parts = [max(0, min(255, int(p.strip()))) for p in s.split(",")]
            if len(parts) == 1:
                parts = parts * 3
            return tuple(parts[:3])
        except (ValueError, IndexError):
            return default

    def process(self, images, labels, columns, thumb_size, padding, label_height,
                bg_color, label_color, highlight_index, highlight_color):
        n = images.shape[0]
        cols = max(1, min(columns, n))
        rows = (n + cols - 1) // cols

        bg = self._parse_color(bg_color, default=(20, 20, 20))
        label_fill = self._parse_color(label_color, default=(255, 255, 255))
        hl_color = self._parse_color(highlight_color, default=(0, 255, 120))

        label_list = [l for l in labels.split("\n")] if labels.strip() else []
        while len(label_list) < n:
            label_list.append(f"#{len(label_list) + 1}")

        cell_w = thumb_size
        cell_h = thumb_size + label_height
        sheet_w = cols * cell_w + (cols + 1) * padding
        sheet_h = rows * cell_h + (rows + 1) * padding

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg)
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(size=max(12, label_height - 12))
        except TypeError:
            font = ImageFont.load_default()

        batch_np = (images.cpu().numpy() * 255.0).astype(np.uint8)

        for i in range(n):
            row, col = divmod(i, cols)
            x0 = padding + col * (cell_w + padding)
            y0 = padding + row * (cell_h + padding)

            frame = Image.fromarray(batch_np[i]).convert("RGB")
            frame.thumbnail((thumb_size, thumb_size), Image.LANCZOS)

            paste_x = x0 + (thumb_size - frame.width) // 2
            paste_y = y0 + (thumb_size - frame.height) // 2
            sheet.paste(frame, (paste_x, paste_y))

            if i == highlight_index:
                draw.rectangle(
                    [x0 - 2, y0 - 2, x0 + thumb_size + 2, y0 + thumb_size + 2],
                    outline=hl_color, width=4,
                )

            label_text = label_list[i]
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x0 + (thumb_size - text_w) // 2
            text_y = y0 + thumb_size + (label_height - (bbox[3] - bbox[1])) // 2
            draw.text((text_x, text_y), label_text, font=font, fill=label_fill)

        out = torch.from_numpy(np.array(sheet).astype(np.float32) / 255.0).unsqueeze(0)
        summary = f"{n} images in a {cols}x{rows} labeled comparison grid"
        return (out, summary)


NODE_CLASS_MAPPINGS = {
    "ImageGridCompareNode": ImageGridCompareNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGridCompareNode": "Image Grid Compare 🆚",
}
