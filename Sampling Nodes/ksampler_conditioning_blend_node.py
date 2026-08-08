"""
TensorVizion ComfyUI Nodes
ksampler_conditioning_blend_node.py — Takes TWO positive conditionings
(e.g. two different CLIPTextEncode prompts) and a blend ratio, averages
them into one conditioning using the same weighted-average approach as
ComfyUI's own core ConditioningAverage node, then samples in one call.

Different in kind from the other three new samplers: this one has an
EXTRA CONDITIONING input (two positives instead of one) rather than an
extra model, an image+mask pair, or a seed-count widget — useful for
"50/50 between these two prompt ideas" or gradually sweeping blend_ratio
across a batch of renders to see where a concept transitions from one
prompt's influence to the other's.
"""

import torch

from nodes import KSampler as _CoreKSampler


class ConditioningBlendKSamplerNode:
    """
    `blend_ratio` = 1.0 uses `positive_a` only, 0.0 uses `positive_b`
    only, anything between weighted-averages the two conditioning
    tensors (and their pooled_output, if present) — the same math
    ComfyUI's own core ConditioningAverage node uses internally, applied
    here before handing the result straight to a sampler so the blend
    and the sample happen in one node instead of two.

    If `positive_a` and `positive_b` have different token-length
    conditioning tensors, the shorter one is zero-padded to match before
    blending — same handling ComfyUI's own core node uses, so a length
    mismatch doesn't silently produce a shape error.
    """

    CATEGORY = "TensorVizion/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_a": ("CONDITIONING",),
                "positive_b": ("CONDITIONING",),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "dpmpp_2m"}),
                "scheduler": ("STRING", {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "summary")
    FUNCTION = "run"

    def _blend_conditioning(self, positive_a, positive_b, ratio):
        # Same approach as ComfyUI's own core ConditioningAverage.addWeighted:
        # weighted-average the conditioning tensor and pooled_output,
        # zero-padding the shorter conditioning_from tensor to match the
        # longer one's token length before blending.
        if len(positive_b) > 1:
            positive_b = positive_b[:1]
        cond_from = positive_b[0][0]
        pooled_from = positive_b[0][1].get("pooled_output", None)

        out = []
        for cond_to, meta_to in positive_a:
            pooled_to = meta_to.get("pooled_output", pooled_from)

            t0 = cond_from[:, :cond_to.shape[1]]
            if t0.shape[1] < cond_to.shape[1]:
                pad = torch.zeros((1, cond_to.shape[1] - t0.shape[1], cond_to.shape[2]))
                t0 = torch.cat([t0, pad], dim=1)

            blended_cond = cond_to * ratio + t0 * (1.0 - ratio)
            new_meta = meta_to.copy()
            if pooled_from is not None and pooled_to is not None:
                new_meta["pooled_output"] = pooled_to * ratio + pooled_from * (1.0 - ratio)
            elif pooled_from is not None:
                new_meta["pooled_output"] = pooled_from

            out.append([blended_cond, new_meta])
        return out

    def run(self, model, positive_a, positive_b, blend_ratio, negative, latent_image,
            seed, steps, cfg, sampler_name, scheduler, denoise):
        if blend_ratio >= 0.999:
            blended_positive = positive_a
        elif blend_ratio <= 0.001:
            blended_positive = positive_b
        else:
            blended_positive = self._blend_conditioning(positive_a, positive_b, blend_ratio)

        sampler = _CoreKSampler()
        sampler_fn = getattr(sampler, sampler.FUNCTION)
        result = sampler_fn(
            model, seed, steps, cfg, sampler_name, scheduler,
            blended_positive, negative, latent_image, denoise,
        )
        final_latent = result[0]

        summary = (
            f"Blend ratio: {blend_ratio:.2f} (1.0=all prompt A, 0.0=all prompt B)\n"
            f"Sampler/Scheduler: {sampler_name} / {scheduler}\n"
            f"Steps: {steps}   CFG: {cfg}   Denoise: {denoise}\n"
            f"Seed: {seed}"
        )
        return (final_latent, summary)


NODE_CLASS_MAPPINGS = {
    "ConditioningBlendKSamplerNode": ConditioningBlendKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningBlendKSamplerNode": "KSampler Conditioning Blend 🔀",
}
