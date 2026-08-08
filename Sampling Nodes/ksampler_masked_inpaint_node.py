"""
TensorVizion ComfyUI Nodes
ksampler_masked_inpaint_node.py — An inpainting-focused sampler with
FEWER required inputs than the usual VAE Encode (for Inpainting) ->
KSampler two-node chain: this single node takes the source image and
mask directly and handles the encode-then-sample sequence internally.

Different in kind from Simple KSampler and Base+Refiner: this one takes
an IMAGE+MASK pair instead of a pre-built LATENT, since inpainting workflows
don't have a LATENT to hand it until AFTER the masked encode step happens
— building that latent is this node's job, not something wired in from
outside.
"""

from nodes import VAEEncodeForInpaint as _CoreInpaintEncode
from nodes import KSampler as _CoreKSampler


class MaskedInpaintKSamplerNode:
    """
    Encodes `image` + `mask` into an inpainting-ready latent (via
    ComfyUI's own `VAEEncodeForInpaint`, which handles the mask-grow and
    masked-region normalization correctly rather than reimplementing
    that math here), then samples it in one call.

    `grow_mask_by` expands the mask by that many pixels before encoding
    — the same parameter VAE Encode (for Inpainting) exposes, kept here
    so the mask-growing step doesn't need a separate upstream node. A
    small grow (4-8px) helps avoid a visible seam at the mask edge.

    `denoise` below 1.0 preserves more of the original masked-region
    content; 1.0 replaces it entirely within the mask according to the
    prompt.
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "summary")
    FUNCTION = "run"

    def run(self, model, vae, image, mask, positive, negative, seed, steps, cfg,
            sampler_name, scheduler, denoise, grow_mask_by):
        encoder = _CoreInpaintEncode()
        encoder_fn = getattr(encoder, encoder.FUNCTION)
        encode_result = encoder_fn(vae, image, mask, grow_mask_by)
        inpaint_latent = encode_result[0]

        sampler = _CoreKSampler()
        sampler_fn = getattr(sampler, sampler.FUNCTION)
        sample_result = sampler_fn(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, inpaint_latent, denoise,
        )
        final_latent = sample_result[0]

        summary = (
            f"Inpaint mask grown by: {grow_mask_by}px\n"
            f"Sampler/Scheduler: {sampler_name} / {scheduler}\n"
            f"Steps: {steps}   CFG: {cfg}   Denoise: {denoise}\n"
            f"Seed: {seed}"
        )
        return (final_latent, summary)


NODE_CLASS_MAPPINGS = {
    "MaskedInpaintKSamplerNode": MaskedInpaintKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskedInpaintKSamplerNode": "KSampler Masked Inpaint 🖌️",
}
