"""
TensorVizion ComfyUI Nodes
image_aspect_ratio_bucket_node.py — Snaps an image to the nearest standard
SD1.5/SDXL training aspect-ratio bucket, then center-crops and resizes to
that bucket's exact resolution. A classic missing utility when prepping
fine-tuning datasets so every image matches a bucket the trainer expects
instead of getting squashed to a single fixed square size.
"""

import numpy as np
import torch


class ImageAspectRatioBucketNode:
    """
    Compares the input image's aspect ratio against a preset table of
    standard training buckets and snaps to the closest one, then
    center-crops to that exact ratio before resizing (so no distortion
    is introduced — excess content is cropped, not squeezed).

    bucket_set:
      sdxl  — the standard SDXL multi-aspect bucket set (1024-area buckets)
      sd15  — matching set scaled to 512-area buckets for SD1.5 training

    Returns the bucketed IMAGE, the bucket resolution as "WxH", and a
    STRING report of the chosen bucket plus how much was cropped.
    """

    CATEGORY = "TensorVizion/Image"

    SDXL_BUCKETS = [
        (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
        (1344, 768), (768, 1344), (1536, 640), (640, 1536),
    ]

    SD15_BUCKETS = [
        (512, 512), (576, 448), (448, 576), (608, 416), (416, 608),
        (672, 384), (384, 672), (768, 320), (320, 768),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bucket_set": (["sdxl", "sd15"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("bucketed_image", "bucket_resolution", "report")
    FUNCTION = "bucket"

    def _nearest_bucket(self, ratio, buckets):
        best = min(buckets, key=lambda wh: abs((wh[0] / wh[1]) - ratio))
        return best

    def bucket(self, image, bucket_set):
        buckets = self.SDXL_BUCKETS if bucket_set == "sdxl" else self.SD15_BUCKETS

        B, H, W, C = image.shape
        ratio = W / H
        target_w, target_h = self._nearest_bucket(ratio, buckets)
        target_ratio = target_w / target_h

        # Center-crop the source to match the target ratio exactly before
        # resizing, so we crop excess rather than distort.
        if ratio > target_ratio:
            # source is wider than target -> crop width
            new_w = int(round(H * target_ratio))
            new_w = min(new_w, W)
            x0 = (W - new_w) // 2
            cropped = image[:, :, x0:x0 + new_w, :]
        else:
            # source is taller than target -> crop height
            new_h = int(round(W / target_ratio))
            new_h = min(new_h, H)
            y0 = (H - new_h) // 2
            cropped = image[:, y0:y0 + new_h, :, :]

        chw = cropped.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            chw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        out = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        bucket_resolution = f"{target_w}x{target_h}"
        report = (
            f"Source:      {W}x{H} (ratio {ratio:.3f})\n"
            f"Bucket set:  {bucket_set}\n"
            f"Chosen bucket: {bucket_resolution} (ratio {target_ratio:.3f})\n"
            f"Cropped from: {cropped.shape[2]}x{cropped.shape[1]} before resize"
        )

        return (out, bucket_resolution, report)


NODE_CLASS_MAPPINGS = {
    "ImageAspectRatioBucketNode": ImageAspectRatioBucketNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAspectRatioBucketNode": "Aspect Ratio Bucket 📐",
}
