"""
TensorVizion ComfyUI Nodes
image_face_detect_crop_node.py — Detects faces in an IMAGE batch using a
Haar cascade (OpenCV, CPU-only, no extra model download) and crops/pads each
detection to a square target size. Built for isolating subjects when
prepping character LoRA / fine-tuning datasets.
"""

import numpy as np
import torch


class ImageFaceDetectCropNode:
    """
    Runs OpenCV's built-in frontal-face Haar cascade on each image in the
    batch, then crops around the largest (or first) detected face with a
    configurable margin and resizes to `output_size` x `output_size`.

    If no face is found in an image, `on_no_face` controls the fallback:
      center_crop — crop a square from the image center instead
      skip        — return a blank placeholder frame for that slot
      pass_through — return the original image resized, uncropped

    `select` chooses which face to use when multiple are detected:
      largest — highest area (recommended for single-subject portraits)
      first   — first detection returned by the cascade

    Returns the cropped IMAGE batch, a MASK batch marking which frames had
    a detection (1.0 = face found), and a STRING summary of counts.
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "select": (["largest", "first"],),
                "on_no_face": (["center_crop", "skip", "pass_through"],),
                "min_neighbors": ("INT", {"default": 5, "min": 1, "max": 20}),
                "scale_factor": ("FLOAT", {"default": 1.1, "min": 1.01, "max": 1.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("cropped_images", "detected_mask", "summary")
    FUNCTION = "detect_and_crop"

    _cascade = None

    @classmethod
    def _get_cascade(cls):
        if cls._cascade is None:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cls._cascade = cv2.CascadeClassifier(cascade_path)
        return cls._cascade

    def _center_crop_resize(self, img_np, output_size):
        H, W = img_np.shape[:2]
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        crop = img_np[y0:y0 + side, x0:x0 + side]
        return self._resize(crop, output_size)

    def _resize(self, img_np, output_size):
        import cv2
        return cv2.resize(img_np, (output_size, output_size), interpolation=cv2.INTER_AREA)

    def detect_and_crop(self, image, output_size, margin, select, on_no_face, min_neighbors, scale_factor):
        try:
            import cv2
        except ImportError:
            blank = torch.zeros((image.shape[0], output_size, output_size, image.shape[-1]), dtype=torch.float32)
            zero_mask = torch.zeros((image.shape[0], output_size, output_size), dtype=torch.float32)
            return (blank, zero_mask,
                    "[TensorVizion] opencv-python not installed — run: pip install opencv-python")

        cascade = self._get_cascade()

        B, H, W, C = image.shape
        out_frames = []
        out_masks = []
        found_count = 0

        for i in range(B):
            frame = image[i].cpu().numpy()
            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(frame_u8[..., :3], cv2.COLOR_RGB2GRAY) if C >= 3 else frame_u8[..., 0]

            faces = cascade.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
            )

            if len(faces) == 0:
                if on_no_face == "center_crop":
                    cropped = self._center_crop_resize(frame, output_size)
                elif on_no_face == "pass_through":
                    cropped = self._resize(frame, output_size)
                else:  # skip
                    cropped = np.zeros((output_size, output_size, C), dtype=np.float32)
                out_frames.append(cropped)
                out_masks.append(np.zeros((output_size, output_size), dtype=np.float32))
                continue

            found_count += 1

            if select == "largest":
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            else:
                fx, fy, fw, fh = faces[0]

            # Expand the box by margin, keep it square, clamp to image bounds
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0
            side = max(fw, fh) * (1.0 + margin)
            half = side / 2.0

            x0 = int(max(0, cx - half))
            y0 = int(max(0, cy - half))
            x1 = int(min(W, cx + half))
            y1 = int(min(H, cy + half))

            # Re-square in case clamping made it rectangular
            crop_w = x1 - x0
            crop_h = y1 - y0
            side_final = min(crop_w, crop_h)
            x1 = x0 + side_final
            y1 = y0 + side_final

            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                crop = frame

            cropped = self._resize(crop, output_size)
            out_frames.append(cropped)
            out_masks.append(np.ones((output_size, output_size), dtype=np.float32))

        images_out = torch.from_numpy(np.stack(out_frames, axis=0)).float()
        masks_out = torch.from_numpy(np.stack(out_masks, axis=0)).float()

        summary = (
            f"Faces detected: {found_count}/{B}\n"
            f"Output size:    {output_size}x{output_size}\n"
            f"Fallback mode:  {on_no_face}"
        )

        return (images_out, masks_out, summary)


NODE_CLASS_MAPPINGS = {
    "ImageFaceDetectCropNode": ImageFaceDetectCropNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFaceDetectCropNode": "Face Detect & Crop 🙂",
}
