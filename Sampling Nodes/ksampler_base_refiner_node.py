"""
TensorVizion ComfyUI Nodes
ksampler_base_refiner_node.py — Bundles SDXL's own documented base+refiner
two-stage sampling pattern into a single node. Normally this needs two
separate KSamplerAdvanced nodes wired together with matching start/end
steps and a leftover-noise handoff — this node does that internally so
the workflow graph only needs one node and two model inputs, not four
manually-configured sampler nodes.

Different in kind from Simple KSampler (Sampling Nodes) — that wraps ONE
model; this one takes TWO models and performs a real mid-generation
handoff between them, not just two back-to-back full samples.
"""

import torch

from nodes import KSamplerAdvanced as _CoreKSamplerAdvanced


class BaseRefinerKSamplerNode:
    """
    Samples `steps` total steps, using `model_base` for steps
    `0..switch_step` and `model_refiner` for `switch_step..steps` — the
    same split SDXL's official base+refiner workflow uses, implemented
    via two internal KSamplerAdvanced calls:

      Stage 1 (base):     add_noise=True,  start=0,           end=switch_step, return_with_leftover_noise=True
      Stage 2 (refiner):  add_noise=False, start=switch_step, end=steps,       return_with_leftover_noise=False

    The `return_with_leftover_noise=True` on stage 1 is what makes this a
    correct handoff rather than two independent samples stitched
    together — it hands stage 2 a latent that still has the right amount
    of residual noise for its start step, exactly matching what a single
    continuous sampling run at that step would look like.

    `switch_fraction` (0.0-1.0) sets where the handoff happens as a
    fraction of total steps — 0.8 (the common SDXL default) means the
    base model does the first 80% of steps and the refiner finishes the
    last 20%, matching Stability AI's own published recommendation for
    the base+refiner pair.
    """

    CATEGORY = "TensorVizion/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_refiner": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "dpmpp_2m"}),
                "scheduler": ("STRING", {"default": "karras"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "switch_fraction": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 0.95, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "summary")
    FUNCTION = "run"

    def run(self, model_base, model_refiner, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, switch_fraction):
        switch_step = max(1, min(steps - 1, round(steps * switch_fraction)))

        # Called via getattr(instance, instance.FUNCTION) rather than a
        # hardcoded ".sample(...)" — every real ComfyUI node's FUNCTION
        # class attribute names its own callable by contract (that's how
        # ComfyUI's own engine invokes every node internally), so this
        # stays correct even if a future core version ever renamed the
        # method, without needing to guess/hardcode it here.
        #
        # All positional, matching KSamplerAdvanced's confirmed real
        # signature order: model, add_noise, noise_seed, steps, cfg,
        # sampler_name, scheduler, positive, negative, latent_image,
        # start_at_step, end_at_step, return_with_leftover_noise.
        stage1 = _CoreKSamplerAdvanced()
        stage1_fn = getattr(stage1, stage1.FUNCTION)
        stage1_result = stage1_fn(
            model_base, "enable", seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            0, switch_step, "enable",
        )
        handoff_latent = stage1_result[0]

        stage2 = _CoreKSamplerAdvanced()
        stage2_fn = getattr(stage2, stage2.FUNCTION)
        stage2_result = stage2_fn(
            model_refiner, "disable", seed, steps, cfg, sampler_name, scheduler,
            positive, negative, handoff_latent,
            switch_step, steps, "disable",
        )
        final_latent = stage2_result[0]

        summary = (
            f"Base model steps:    0 -> {switch_step}\n"
            f"Refiner model steps: {switch_step} -> {steps}\n"
            f"Switch fraction:     {switch_fraction:.2f}\n"
            f"Sampler/Scheduler:   {sampler_name} / {scheduler}\n"
            f"CFG: {cfg}   Seed: {seed}"
        )
        return (final_latent, summary)


NODE_CLASS_MAPPINGS = {
    "BaseRefinerKSamplerNode": BaseRefinerKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaseRefinerKSamplerNode": "KSampler Base+Refiner 🎭",
}
