"""
TensorVizion ComfyUI Nodes
text_overlay_node.py — Draws text onto an IMAGE: font size, color,
position (9-point anchor grid + pixel offset), stroke/outline, and
optional background box. No text-rendering capability existed anywhere
in the pack before this — useful for watermarking, caption burn-in on
contact sheets, or meme-style generation.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class TextOverlayNode:
    """
    `anchor` places the text at one of 9 positions (corners/edges/center)
    with `margin` pixels of padding from that anchor point; `offset_x`/
    `offset_y` then nudge it further for fine positioning.

    `font_path` is optional — leave blank to use PIL's built-in default
    font (always available, no extra files needed, but fairly plain and
    fixed-size). Point it at any .ttf/.otf file on disk for a real
    scalable font.

    `stroke_width` draws an outline in `stroke_color` behind the fill —
    the standard way to keep text readable over a busy/bright background
    without needing a semi-transparent box behind it. Set to 0 to disable.

    `background_box` draws a solid `background_color` rectangle (with
    `background_padding` pixels of margin around the text) behind the
    text before drawing it — useful when stroke alone isn't enough
    contrast, e.g. a caption bar at the bottom of a frame.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "Sample Text", "multiline": True}),
                "font_size": ("INT", {"default": 48, "min": 4, "max": 512, "step": 1}),
                "font_path": ("STRING", {"default": ""}),
                "text_color": ("STRING", {"default": "255,255,255"}),
                "anchor": ([
                    "top_left", "top_center", "top_right",
                    "middle_left", "middle_center", "middle_right",
                    "bottom_left", "bottom_center", "bottom_right",
                ], {"default": "bottom_center"}),
                "margin": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "stroke_width": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "stroke_color": ("STRING", {"default": "0,0,0"}),
                "background_box": ("BOOLEAN", {"default": False}),
                "background_color": ("STRING", {"default": "0,0,0"}),
                "background_padding": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "summary")
    FUNCTION = "process"

    def _parse_color(self, s, default=(255, 255, 255)):
        try:
            parts = [max(0, min(255, int(p.strip()))) for p in s.split(",")]
            if len(parts) == 1:
                parts = parts * 3
            return tuple(parts[:3])
        except (ValueError, IndexError):
            return default

    def _load_font(self, font_path, font_size):
        if font_path.strip():
            try:
                return ImageFont.truetype(font_path.strip(), font_size), True
            except (OSError, IOError):
                pass  # fall through to default below
        try:
            # Newer Pillow versions accept a size arg on load_default();
            # older ones don't — try the sized call first, fall back cleanly.
            return ImageFont.load_default(size=font_size), False
        except TypeError:
            return ImageFont.load_default(), False

    def _anchor_position(self, anchor, img_w, img_h, text_w, text_h, margin):
        if "left" in anchor:
            x = margin
        elif "right" in anchor:
            x = img_w - text_w - margin
        else:
            x = (img_w - text_w) // 2

        if "top" in anchor:
            y = margin
        elif "bottom" in anchor:
            y = img_h - text_h - margin
        else:
            y = (img_h - text_h) // 2

        return x, y

    def process(self, image, text, font_size, font_path, text_color, anchor, margin,
                offset_x, offset_y, stroke_width, stroke_color, background_box,
                background_color, background_padding):
        fill = self._parse_color(text_color)
        stroke_fill = self._parse_color(stroke_color, default=(0, 0, 0))
        bg_fill = self._parse_color(background_color, default=(0, 0, 0))

        font, used_custom_font = self._load_font(font_path, font_size)

        batch_np = (image.cpu().numpy() * 255.0).astype(np.uint8)
        out_frames = []

        for i in range(batch_np.shape[0]):
            pil_img = Image.fromarray(batch_np[i]).convert("RGB")
            draw = ImageDraw.Draw(pil_img)

            bbox = draw.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke_width)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            x, y = self._anchor_position(anchor, pil_img.width, pil_img.height, text_w, text_h, margin)
            x += offset_x
            y += offset_y

            if background_box:
                pad = background_padding
                draw.rectangle(
                    [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
                    fill=bg_fill,
                )

            draw.multiline_text(
                (x - bbox[0], y - bbox[1]),
                text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )

            out_frames.append(np.array(pil_img).astype(np.float32) / 255.0)

        out_tensor = torch.from_numpy(np.stack(out_frames))
        font_note = "custom TTF" if used_custom_font else "PIL default font"
        summary = f"Drew text at anchor={anchor} using {font_note}, size={font_size}"
        return (out_tensor, summary)


NODE_CLASS_MAPPINGS = {
    "TextOverlayNode": TextOverlayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextOverlayNode": "Text Overlay ✏️",
}
