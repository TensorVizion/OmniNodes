"""
TensorVizion ComfyUI Nodes
ksampler_seed_variator_node.py — Samples the SAME prompt/settings across
`num_variations` different seeds and returns them as one batched LATENT,
instead of needing N separate KSampler nodes wired in parallel to compare
seed variety. The inverse shape of Base+Refiner: one input configuration,
many outputs collapsed into a single batched result rather than one
input needing two models.
"""

import torch

from nodes import KSampler as _CoreKSampler


class SeedVariatorKSamplerNode:
    """
    Runs `num_variations` independent samples of the same
    model/prompt/latent_image at consecutive seeds (`base_seed`,
    `base_seed+1`, `base_seed+2`, ...) and concatenates the results into
    one batch, exactly like manually running a KSampler `num_variations`
    times at different seeds and batching the outputs — just without
    needing that many separate nodes in the graph.

    `latent_image` must be a batch-1 latent (a single EmptyLatentImage,
    not a pre-existing batch) — each variation gets its own independent
    noise draw at its own seed from that same starting shape.

    Useful for a quick "seed lottery" pass: sample 4-8 variations, VAE
    Decode the whole batch, then look through a contact sheet or
    Image Grid Compare to pick the one worth refining further.
    """

    CATEGORY = "TensorVizion/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "num_variations": ("INT", {"default": 4, "min": 1, "max": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "dpmpp_2m"}),
                "scheduler": ("STRING", {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent_batch", "seeds_used", "summary")
    FUNCTION = "run"

    def run(self, model, positive, negative, latent_image, base_seed, num_variations,
            steps, cfg, sampler_name, scheduler, denoise):
        single_latent = {"samples": latent_image["samples"][0:1]}
        if "batch_index" in latent_image:
            single_latent["batch_index"] = latent_image["batch_index"][:1]

        sampler = _CoreKSampler()
        sampler_fn = getattr(sampler, sampler.FUNCTION)

        batch_samples = []
        seeds_used = []
        for i in range(num_variations):
            this_seed = (base_seed + i) & 0xffffffffffffffff
            result = sampler_fn(
                model, this_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, single_latent, denoise,
            )
            batch_samples.append(result[0]["samples"])
            seeds_used.append(this_seed)

        combined = torch.cat(batch_samples, dim=0)
        out_latent = {"samples": combined}

        seeds_str = ", ".join(str(s) for s in seeds_used)
        summary = (
            f"Variations: {num_variations}\n"
            f"Seeds used: {seeds_str}\n"
            f"Sampler/Scheduler: {sampler_name} / {scheduler}\n"
            f"Steps: {steps}   CFG: {cfg}   Denoise: {denoise}\n"
            f"Output batch shape: {tuple(combined.shape)}"
        )
        return (out_latent, seeds_str, summary)


NODE_CLASS_MAPPINGS = {
    "SeedVariatorKSamplerNode": SeedVariatorKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVariatorKSamplerNode": "KSampler Seed Variator 🎲",
}
