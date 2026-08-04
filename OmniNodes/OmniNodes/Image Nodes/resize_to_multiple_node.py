"""
TensorVizion ComfyUI Nodes
resize_to_multiple_node.py — Resizes an IMAGE so both dimensions are a
multiple of N (8 for SD/SDXL VAE compatibility) without distorting aspect
ratio. A small but constantly-needed utility: every img2img/inpaint
workflow eventually needs to feed an arbitrary source image into a latent
pipeline that requires dimensions divisible by 8.
"""

import torch
import torch.nn.functional as F


class ResizeToMultipleNode:
    """
    mode:
      pad_to_multiple   — scales the image down/up only as needed to fit
                            within a multiple-of-N box, then pads the
                            remainder with `pad_color` (letterbox-style).
                            Never crops content; aspect ratio is fully
                            preserved, output may have solid-color borders.
      crop_to_multiple  — scales the image to cover a multiple-of-N box,
                            then center-crops the overflow. Never adds
                            borders; some edge content may be cut off.
      stretch_to_multiple — resizes both dimensions independently to the
                            nearest multiple of N. Fastest, but will distort
                            aspect ratio if the source isn't already close
                            to a multiple-of-N shape.

    `multiple` defaults to 8 (SD/SDXL VAE requirement); some newer models
    (e.g. certain video VAEs) expect 16 or 32 — adjustable here rather than
    hardcoding 8.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["pad_to_multiple", "crop_to_multiple", "stretch_to_multiple"],),
                "multiple": ("INT", {"default": 8, "min": 2, "max": 128, "step": 2}),
                "max_dimension": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 8}),
                "pad_color": ("STRING", {"default": "0,0,0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "summary")
    FUNCTION = "process"

    def _round_to_multiple(self, value, multiple):
        return max(multiple, round(value / multiple) * multiple)

    def _parse_color(self, s):
        try:
            parts = [max(0, min(255, int(p.strip()))) / 255.0 for p in s.split(",")]
            if len(parts) == 1:
                parts = parts * 3
            return parts[:3]
        except (ValueError, IndexError):
            return [0.0, 0.0, 0.0]

    def process(self, image, mode, multiple, max_dimension, pad_color):
        B, H, W, C = image.shape
        chw = image.permute(0, 3, 1, 2).float()

        # Cap the working size first so a very large source image doesn't
        # round up to an enormous multiple-of-N target.
        scale_cap = min(1.0, max_dimension / max(H, W))
        cap_h, cap_w = int(H * scale_cap), int(W * scale_cap)

        target_h = self._round_to_multiple(cap_h, multiple)
        target_w = self._round_to_multiple(cap_w, multiple)

        if mode == "stretch_to_multiple":
            out = F.interpolate(chw, size=(target_h, target_w), mode="bilinear", align_corners=False)

        elif mode == "crop_to_multiple":
            scale = max(target_h / H, target_w / W)
            scaled_h, scaled_w = max(target_h, int(H * scale)), max(target_w, int(W * scale))
            scaled = F.interpolate(chw, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
            top = (scaled_h - target_h) // 2
            left = (scaled_w - target_w) // 2
            out = scaled[:, :, top:top + target_h, left:left + target_w]

        else:  # pad_to_multiple
            scale = min(target_h / H, target_w / W)
            scaled_h, scaled_w = min(target_h, int(H * scale)), min(target_w, int(W * scale))
            scaled_h, scaled_w = max(1, scaled_h), max(1, scaled_w)
            scaled = F.interpolate(chw, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)

            color = self._parse_color(pad_color)
            out = torch.tensor(color, device=chw.device).view(1, 3, 1, 1).repeat(B, 1, target_h, target_w)
            top = (target_h - scaled_h) // 2
            left = (target_w - scaled_w) // 2
            out[:, :, top:top + scaled_h, left:left + scaled_w] = scaled

        out = out.clamp(0.0, 1.0)
        summary = f"{W}x{H} -> {target_w}x{target_h} (mode={mode}, multiple={multiple})"
        return (out.permute(0, 2, 3, 1), target_w, target_h, summary)


NODE_CLASS_MAPPINGS = {
    "ResizeToMultipleNode": ResizeToMultipleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeToMultipleNode": "Resize to Multiple 📏",
}
