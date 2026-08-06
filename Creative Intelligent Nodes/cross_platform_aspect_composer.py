import torch
import torch.nn.functional as F


class CrossPlatformAspectComposer:
    """
    Crops a source image into multiple aspect ratios while keeping the
    essential content anchored in the safe-zone intersection of all targets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_ratios": ("STRING", {"default": "9:16,1:1,16:9,4:5", "multiline": False}),
                "anchor_strategy": (["center", "subject_track", "rule_of_thirds"],),
                "output_master": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crops", "safe_zone_info")
    FUNCTION = "compose"
    CATEGORY = "creative/director_toolkit"

    def compose(self, image, target_ratios, anchor_strategy, output_master):
        ratios = []
        for r in target_ratios.split(","):
            r = r.strip()
            if ":" in r:
                w, h = r.split(":")
                ratios.append((float(w), float(h), r))

        B, H, W, C = image.shape

        # Determine anchor point
        if anchor_strategy == "center":
            ax, ay = W / 2, H / 2
        elif anchor_strategy == "rule_of_thirds":
            ax, ay = W / 3, H / 3
        else:  # subject_track (approximate via brightness centroid)
            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            weights = gray - gray.min()
            weights = weights / (weights.sum() + 1e-8)
            ys = torch.arange(H, device=image.device, dtype=torch.float32)
            xs = torch.arange(W, device=image.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            ay = (yy * weights).sum()
            ax = (xx * weights).sum()
            ax = float(ax)
            ay = float(ay)

        # Find safe intersection: smallest width and smallest height across ratios
        # Each ratio gives us crop dims for given image
        crops = []
        info_lines = [f"Anchor: ({ax:.0f}, {ay:.0f}) via {anchor_strategy}"]

        for rw, rh, label in ratios:
            target_aspect = rw / rh
            source_aspect = W / H

            if target_aspect > source_aspect:
                crop_w = W
                crop_h = int(W / target_aspect)
            else:
                crop_h = H
                crop_w = int(H * target_aspect)

            x1 = int(max(0, min(W - crop_w, ax - crop_w / 2)))
            y1 = int(max(0, min(H - crop_h, ay - crop_h / 2)))
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            crop = image[:, y1:y2, x1:x2, :]
            crops.append(crop)
            info_lines.append(f"{label}: {crop_w}x{crop_h} at ({x1},{y1})")

        if output_master:
            crops.insert(0, image)

        stacked = torch.cat(crops, dim=0)
        return (stacked, "\n".join(info_lines))