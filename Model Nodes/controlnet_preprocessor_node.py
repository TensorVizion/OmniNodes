"""
TensorVizion ComfyUI Nodes
controlnet_preprocessor_node.py — Converts an IMAGE into a ControlNet
conditioning image (canny edges, a lightweight depth estimate, or lineart)
without depending on a separate ControlNet-aux install. Pairs with
Model Nodes/controlnet_loader_node.py, which loads the ControlNet model
itself but has nothing upstream to build its conditioning image from —
this node is that missing upstream step.
"""

import torch
import torch.nn.functional as F


class ControlNetPreprocessorNode:
    """
    preprocessor:
      canny        — classic Canny edge detection (Sobel gradient + double
                       threshold), pure PyTorch, no OpenCV dependency.
      depth_lite   — a lightweight monocular depth ESTIMATE, not a real
                       depth model: uses luminance + a soft blur-based
                       defocus cue as a proxy for near/far. This is a rough
                       stand-in for scenes with a clear focal subject —
                       for real depth accuracy, use a proper MiDaS/Depth-
                       Anything ControlNet preprocessor node instead. Kept
                       here specifically for quick iteration without an
                       extra model download.
      lineart      — inverted, thresholded edge map tuned for a cleaner
                       "line drawing" look than raw canny (softer, thicker
                       lines from a wider Sobel kernel).

    All modes output a 3-channel IMAGE (edges/depth replicated across RGB)
    so it's directly wireable into any ControlNet apply node expecting a
    standard IMAGE conditioning input.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (["canny", "depth_lite", "lineart"],),
                "low_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "high_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("control_image", "summary")
    FUNCTION = "process"

    # ------------------------------------------------------------------
    def _to_luminance(self, chw):
        # Standard Rec. 601 luma weights.
        weights = torch.tensor([0.299, 0.587, 0.114], device=chw.device).view(1, 3, 1, 1)
        return (chw * weights).sum(dim=1, keepdim=True)

    def _sobel_gradients(self, luma, kernel_size=3):
        if kernel_size == 3:
            kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=luma.device)
        else:
            # A wider, softer 5x5 Sobel-like kernel for the lineart mode —
            # produces thicker, less noisy lines than the standard 3x3.
            kx = torch.tensor([
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
                [-3, -5, 0, 5, 3],
                [-2, -3, 0, 3, 2],
                [-1, -2, 0, 2, 1],
            ], dtype=torch.float32, device=luma.device)
        ky = kx.t()

        pad = kernel_size // 2
        kx = kx.view(1, 1, kernel_size, kernel_size)
        ky = ky.view(1, 1, kernel_size, kernel_size)

        padded = F.pad(luma, (pad, pad, pad, pad), mode="reflect")
        gx = F.conv2d(padded, kx)
        gy = F.conv2d(padded, ky)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-8)

    def _canny_like(self, luma, low, high):
        mag = self._sobel_gradients(luma, kernel_size=3)
        # Double-threshold: below low -> 0, above high -> 1, in between ->
        # scaled linearly. This is a simplified stand-in for true Canny's
        # hysteresis edge-tracking (no edge-linking pass), but produces a
        # visually similar result for ControlNet conditioning purposes.
        edges = torch.zeros_like(mag)
        strong = mag >= high
        weak = (mag >= low) & (mag < high)
        edges[strong] = 1.0
        if high > low:
            edges[weak] = (mag[weak] - low) / (high - low)
        return edges

    def _lineart_like(self, luma, low, high):
        mag = self._sobel_gradients(luma, kernel_size=5)
        edges = torch.zeros_like(mag)
        strong = mag >= high
        weak = (mag >= low) & (mag < high)
        edges[strong] = 1.0
        if high > low:
            edges[weak] = (mag[weak] - low) / (high - low)
        return edges

    def _depth_lite(self, chw, luma):
        # Rough proxy for depth: brighter + sharper (higher local gradient
        # energy) regions are treated as "nearer". This is NOT a trained
        # depth model — it's a cheap heuristic for scenes with a clear
        # focal subject against a softer background, useful for quick
        # iteration without pulling in a MiDaS/Depth-Anything checkpoint.
        blurred = F.avg_pool2d(luma, kernel_size=9, stride=1, padding=4, count_include_pad=False)
        sharpness = (luma - blurred).abs()
        sharpness = sharpness / (sharpness.amax(dim=(2, 3), keepdim=True) + 1e-8)
        depth_proxy = 0.6 * luma + 0.4 * sharpness
        return depth_proxy.clamp(0.0, 1.0)

    def process(self, image, preprocessor, low_threshold, high_threshold, invert):
        chw = image.permute(0, 3, 1, 2).float()  # (B,C,H,W)
        luma = self._to_luminance(chw)

        if low_threshold > high_threshold:
            low_threshold, high_threshold = high_threshold, low_threshold

        if preprocessor == "canny":
            result = self._canny_like(luma, low_threshold, high_threshold)
            summary = f"Canny edges (low={low_threshold:.2f}, high={high_threshold:.2f})"
        elif preprocessor == "lineart":
            result = self._lineart_like(luma, low_threshold, high_threshold)
            summary = f"Lineart edges (low={low_threshold:.2f}, high={high_threshold:.2f})"
        else:
            result = self._depth_lite(chw, luma)
            summary = "Depth-lite estimate (heuristic, not a trained depth model)"

        if invert:
            result = 1.0 - result

        result_rgb = result.repeat(1, 3, 1, 1).clamp(0.0, 1.0)
        return (result_rgb.permute(0, 2, 3, 1), summary)


NODE_CLASS_MAPPINGS = {
    "ControlNetPreprocessorNode": ControlNetPreprocessorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlNetPreprocessorNode": "ControlNet Preprocessor 🕹️",
}
