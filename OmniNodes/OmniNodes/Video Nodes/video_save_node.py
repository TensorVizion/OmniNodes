"""
TensorVizion ComfyUI Nodes
video_save_node.py — Encodes an IMAGE batch (ComfyUI's standard
"video = batch of images" convention) to an actual video file on disk:
MP4, WEBM, or animated GIF, at a chosen frame rate. The exit point video
workflows were missing — Latent Interpolate and Contact Sheet Maker both
produce or consume batches, but nothing closed the loop from "batch in
memory" to "shareable video file" until now.
"""

import os
import numpy as np
import torch


class VideoSaveNode:
    """
    Writes every frame in `images` to a video file at `output_dir` using
    `fps` as the playback rate. `format` selects the container/codec:
    `mp4` (H.264, widely compatible), `webm` (VP9, smaller/web-friendly),
    or `gif` (universal but larger file size, no audio). Requires the
    `imageio` package with its ffmpeg plugin (`pip install imageio[ffmpeg]`)
    for mp4/webm; GIF export works with plain `imageio` alone. If
    `imageio` isn't installed, this node returns a clear error in its
    summary output rather than crashing the queue.
    """

    CATEGORY = "TensorVizion/Video"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":      ("IMAGE",),
                "fps":         ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "filename":    ("STRING", {"default": "tv_video"}),
                "output_dir":  ("STRING", {"default": "output/tensorvizion/video"}),
                "format":      (["mp4", "webm", "gif"], {"default": "mp4"}),
                "quality":     ("INT", {"default": 8, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES  = ("STRING",     "STRING")
    RETURN_NAMES  = ("saved_path", "summary")
    FUNCTION      = "run"

    @staticmethod
    def _tensor_to_uint8(img_tensor):
        arr = img_tensor.cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def run(self, images, fps, filename, output_dir, format, quality):
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                msg = (
                    "[TensorVizion] 'imageio' is not installed. Install it with "
                    "'pip install imageio imageio-ffmpeg' to enable Video Save "
                    "(mp4/webm additionally require the ffmpeg plugin; gif works "
                    "with plain imageio)."
                )
                return ("", msg)

        os.makedirs(output_dir, exist_ok=True)

        ext = format
        counter = 0
        path = os.path.join(output_dir, f"{filename}.{ext}")
        while os.path.exists(path):
            counter += 1
            path = os.path.join(output_dir, f"{filename}_{counter:03d}.{ext}")

        frames = [self._tensor_to_uint8(images[i]) for i in range(images.shape[0])]

        try:
            if format == "gif":
                imageio.mimsave(path, frames, fps=fps)
            else:
                writer_kwargs = {"fps": fps, "quality": quality}
                if format == "mp4":
                    writer_kwargs["codec"] = "libx264"
                elif format == "webm":
                    writer_kwargs["codec"] = "libvpx-vp9"
                imageio.mimsave(path, frames, format=format, **writer_kwargs)
        except Exception as e:
            msg = (
                f"[TensorVizion] Failed to encode {format}: {e}\n"
                f"mp4/webm require the ffmpeg imageio plugin "
                f"('pip install imageio-ffmpeg'). GIF requires no extra plugin."
            )
            return ("", msg)

        summary = (
            f"Format:    {format}\n"
            f"Frames:    {len(frames)}\n"
            f"FPS:       {fps}\n"
            f"Duration:  {len(frames) / fps:.2f}s\n"
            f"Saved to:  {path}"
        )

        return (path, summary)


NODE_CLASS_MAPPINGS = {
    "VideoSaveNode": VideoSaveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSaveNode": "Video Save 🎬",
}
