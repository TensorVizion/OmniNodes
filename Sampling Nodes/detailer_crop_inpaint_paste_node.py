"""
TensorVizion ComfyUI Nodes
detailer_crop_inpaint_paste_node.py — The full "detailer" loop: crop a
masked region (e.g. a face from Image Nodes/image_face_detect_crop_node.py)
out to its own bounding box, upscale that crop so a small region gets a
full sampling pass at proper resolution, re-sample it with fresh
conditioning, then paste the result back into the original image with a
feathered blend at the seam. The pack had detection (crop node) and
sampling (KSampler variants) as separate pieces with no node tying them
into one region-fix pass — this is that missing connective node, built by
delegating cropping/pasting to ComfyUI's own core ImageCrop/ImageCompositeMasked
and re-using VAEEncodeForInpaint + KSampler for the actual fix.
"""

import torch
import torch.nn.functional as F

from nodes import VAEEncodeForInpaint as _CoreInpaintEncode
from nodes import KSampler as _CoreKSampler
from nodes import VAEDecode as _CoreVAEDecode


class DetailerCropInpaintPasteNode:
    """
    1. Crops `image` to the bounding box of `mask`, padded by `padding`px.
    2. Upscales that crop so its longer edge equals `detail_size`px,
       giving the sampler a full-resolution pass on a region that may
       have been tiny in the source image (typical for a detected face).
    3. Re-samples the upscaled crop with `positive`/`negative`
       conditioning at `denoise`, using the (grown, resized) mask so only
       the masked area changes.
    4. Downscales the result back to the original crop size and composites
       it into `image` at the original location, feathering the mask edge
       by `feather`px to avoid a visible seam.

    Use downstream of a face/region detector that outputs a MASK — this
    node does not do detection itself.
    """

    CATEGORY = "TensorVizion/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "dpmpp_2m"}),
                "scheduler": ("STRING", {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 512, "step": 4}),
                "detail_size": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 64}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "summary")
    FUNCTION = "run"

    # ------------------------------------------------------------------
    def _mask_bbox(self, mask_hw, padding, img_h, img_w):
        ys, xs = torch.where(mask_hw > 0.05)
        if ys.numel() == 0:
            return 0, 0, img_w, img_h
        y0, y1 = int(ys.min()) - padding, int(ys.max()) + 1 + padding
        x0, x1 = int(xs.min()) - padding, int(xs.max()) + 1 + padding
        return max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)

    def _resize_chw(self, chw, h, w):
        return F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False)

    def run(self, model, vae, image, mask, positive, negative, seed, steps, cfg,
            sampler_name, scheduler, denoise, padding, detail_size, feather):
        img = image[0]                      # (H, W, C)
        m = mask[0] if mask.dim() == 3 else mask  # (H, W)
        img_h, img_w = img.shape[0], img.shape[1]

        x0, y0, x1, y1 = self._mask_bbox(m, padding, img_h, img_w)
        crop_h, crop_w = y1 - y0, x1 - x0

        crop_img = img[y0:y1, x0:x1, :].unsqueeze(0)          # (1,h,w,C)
        crop_mask = m[y0:y1, x0:x1].unsqueeze(0)              # (1,h,w)

        scale = detail_size / max(crop_h, crop_w)
        det_h, det_w = max(8, round(crop_h * scale)), max(8, round(crop_w * scale))
        det_h -= det_h % 8
        det_w -= det_w % 8

        crop_chw = crop_img.permute(0, 3, 1, 2)
        detail_img = self._resize_chw(crop_chw, det_h, det_w).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        detail_mask = self._resize_chw(crop_mask.unsqueeze(1), det_h, det_w).squeeze(1)

        encoder = _CoreInpaintEncode()
        encode_fn = getattr(encoder, encoder.FUNCTION)
        inpaint_latent = encode_fn(vae, detail_img, detail_mask, 6)[0]

        sampler = _CoreKSampler()
        sample_fn = getattr(sampler, sampler.FUNCTION)
        sampled_latent = sample_fn(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, inpaint_latent, denoise,
        )[0]

        decoder = _CoreVAEDecode()
        decode_fn = getattr(decoder, decoder.FUNCTION)
        detail_result = decode_fn(vae, sampled_latent)[0]      # (1, det_h, det_w, C)

        result_chw = detail_result.permute(0, 3, 1, 2)
        back_to_crop = self._resize_chw(result_chw, crop_h, crop_w).permute(0, 2, 3, 1).clamp(0.0, 1.0)[0]

        feather_mask = crop_mask[0].clone()
        if feather > 0:
            fm = feather_mask.unsqueeze(0).unsqueeze(0)
            k = feather * 2 + 1
            fm = F.avg_pool2d(F.pad(fm, (feather,) * 4, mode="reflect"), kernel_size=k, stride=1)
            feather_mask = fm.squeeze(0).squeeze(0).clamp(0.0, 1.0)

        out = img.clone()
        region = out[y0:y1, x0:x1, :]
        blend = feather_mask.unsqueeze(-1)
        out[y0:y1, x0:x1, :] = region * (1 - blend) + back_to_crop * blend

        summary = (
            f"Detailer region {crop_w}x{crop_h} at ({x0},{y0}), "
            f"sampled at {det_w}x{det_h}, denoise={denoise:.2f}, feather={feather}px"
        )
        return (out.unsqueeze(0), summary)


NODE_CLASS_MAPPINGS = {
    "DetailerCropInpaintPasteNode": DetailerCropInpaintPasteNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailerCropInpaintPasteNode": "Detailer (Crop-Inpaint-Paste) 🔎 (TensorVizion)",
}
