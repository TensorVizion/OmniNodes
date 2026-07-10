"""
TensorVizion ComfyUI Nodes
image_color_grade_node.py — Lift/gamma/gain-style colour grading for IMAGE
tensors: exposure, contrast, saturation, temperature, tint, and gamma in a
single node.
"""

import torch


class ImageColorGradeNode:
    """
    Applies a standard photographic/colour-grading pipeline, in this order:

      1. exposure    — multiplicative stop adjustment (2^exposure)
      2. lift / gain — shadow lift and highlight gain (basic lift-gamma-gain)
      3. gamma       — midtone power curve
      4. contrast    — pivot around 0.5
      5. saturation  — blend toward/away from luminance
      6. temperature — warm (+) / cool (-) shift between red and blue
      7. tint        — magenta (+) / green (-) shift

    All parameters default to a no-op (exposure 0, contrast/saturation/
    gain/gamma 1, lift/temperature/tint 0) so you can dial in from a neutral
    starting point.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "lift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")
    FUNCTION = "grade"

    # ------------------------------------------------------------------
    def grade(self, image, exposure, contrast, saturation, gamma, lift, gain, temperature, tint):
        img = image.clone().float()  # (B, H, W, C)
        C = img.shape[-1]

        # 1. exposure
        img = img * (2.0 ** exposure)

        # 2. lift / gain
        img = img * (gain - lift) + lift

        # 3. gamma
        img = img.clamp(min=1e-6) ** (1.0 / gamma)

        # 4. contrast (pivot at mid-grey)
        img = (img - 0.5) * contrast + 0.5

        # 5. saturation
        if C >= 3:
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device)
            lum = (img[..., :3] * weights).sum(dim=-1, keepdim=True)
            img_rgb = lum + (img[..., :3] - lum) * saturation
            img = torch.cat([img_rgb, img[..., 3:]], dim=-1) if C > 3 else img_rgb

            # 6. temperature: push red up / blue down (or vice versa)
            img = img.clone()
            img[..., 0] = img[..., 0] + temperature * 0.15
            img[..., 2] = img[..., 2] - temperature * 0.15

            # 7. tint: push green vs magenta (red+blue)
            img[..., 1] = img[..., 1] + tint * 0.15
            img[..., 0] = img[..., 0] - tint * 0.075
            img[..., 2] = img[..., 2] - tint * 0.075

        out = img.clamp(0.0, 1.0)

        grade_info = (
            f"Exposure:    {exposure:+.2f} stops\n"
            f"Contrast:    {contrast:.2f}\n"
            f"Saturation:  {saturation:.2f}\n"
            f"Gamma:       {gamma:.2f}\n"
            f"Lift/Gain:   {lift:+.2f} / {gain:.2f}\n"
            f"Temp/Tint:   {temperature:+.2f} / {tint:+.2f}"
        )

        return (out, grade_info)


NODE_CLASS_MAPPINGS = {
    "ImageColorGradeNode": ImageColorGradeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColorGradeNode": "Image Color Grade 🎨",
}
