"""
TensorVizion ComfyUI Nodes
video_frame_interpolate_node.py — Inserts N smoothly-interpolated frames
between each consecutive pair in an IMAGE batch using Farneback dense
optical flow, turning a low-fps generated sequence into a much smoother
one without any ML frame-interpolation model. Pairs naturally with Latent
Interpolate (which already generates a batch of morph latents) — decode
that batch, then run it through this node before Video Save for a much
smoother final clip than the raw latent-space steps alone would give.
"""

import numpy as np
import torch


class VideoFrameInterpolateNode:
    """
    For each consecutive frame pair, computes a dense optical flow field
    with OpenCV's Farneback method and warps both frames toward each
    intermediate timestep, cross-fading the two warped results — a
    classic motion-compensated interpolation approach, much smoother than
    a plain cross-fade for anything with actual motion between frames.
    `interp_frames` sets how many new frames are inserted between each
    original pair (0 = passthrough, no interpolation). Requires OpenCV
    (`pip install opencv-python`); falls back to a linear cross-fade
    (no motion compensation) automatically if OpenCV isn't available, and
    reports which method was used.
    """

    CATEGORY = "TensorVizion/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":        ("IMAGE",),
                "interp_frames": ("INT", {"default": 1, "min": 0, "max": 8}),
            }
        }

    RETURN_TYPES  = ("IMAGE",  "STRING")
    RETURN_NAMES  = ("images", "summary")
    FUNCTION      = "run"

    @staticmethod
    def _to_uint8(frame_tensor):
        return np.clip(frame_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

    def _warp_flow(self, cv2, img, flow):
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    def _optical_flow_interp(self, cv2, frame_a, frame_b, n_between):
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

        flow_ab = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_ba = cv2.calcOpticalFlowFarneback(
            gray_b, gray_a, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        out = []
        for k in range(1, n_between + 1):
            t = k / (n_between + 1)
            warped_a = self._warp_flow(cv2, frame_a, flow_ab * t)
            warped_b = self._warp_flow(cv2, frame_b, flow_ba * (1 - t))
            blended = (warped_a.astype(np.float32) * (1 - t) + warped_b.astype(np.float32) * t)
            out.append(np.clip(blended, 0, 255).astype(np.uint8))
        return out

    def _linear_interp(self, frame_a, frame_b, n_between):
        out = []
        for k in range(1, n_between + 1):
            t = k / (n_between + 1)
            blended = frame_a.astype(np.float32) * (1 - t) + frame_b.astype(np.float32) * t
            out.append(np.clip(blended, 0, 255).astype(np.uint8))
        return out

    def run(self, images, interp_frames):
        if interp_frames <= 0 or images.shape[0] < 2:
            return (images, "interp_frames=0 or single-frame batch — passthrough, no interpolation.")

        try:
            import cv2
            method = "optical_flow"
        except ImportError:
            cv2 = None
            method = "linear_crossfade"

        n_frames = images.shape[0]
        frames_uint8 = [self._to_uint8(images[i]) for i in range(n_frames)]

        output = [frames_uint8[0]]
        for i in range(n_frames - 1):
            a, b = frames_uint8[i], frames_uint8[i + 1]
            if method == "optical_flow":
                between = self._optical_flow_interp(cv2, a, b, interp_frames)
            else:
                between = self._linear_interp(a, b, interp_frames)
            output.extend(between)
            output.append(b)

        arr = np.stack(output).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(arr)

        summary = (
            f"Method:          {method}\n"
            f"Original frames: {n_frames}\n"
            f"Output frames:   {len(output)}\n"
            f"Inserted/pair:   {interp_frames}\n"
        )
        if method == "linear_crossfade":
            summary += "Note: OpenCV not found — used linear cross-fade fallback (no motion compensation). Install opencv-python for smoother results."

        return (out_tensor, summary)


NODE_CLASS_MAPPINGS = {
    "VideoFrameInterpolateNode": VideoFrameInterpolateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameInterpolateNode": "Video Frame Interpolate 🎥",
}
