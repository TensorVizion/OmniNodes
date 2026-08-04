"""
TensorVizion ComfyUI Nodes
simple_ksampler_node.py — The single most important node in any diffusion
workflow: denoises a latent using a model, positive/negative conditioning,
and standard sampler/scheduler/cfg/steps controls. A thin, TensorVizion-
branded wrapper around ComfyUI's own core KSampler (delegated directly,
not reimplemented), added so a complete workflow can be built without
reaching outside the pack for the one node every workflow needs.
"""

from nodes import KSampler as _CoreKSampler


class SimpleKSamplerNode:
    """
    Delegates straight to ComfyUI's own core `KSampler.sample()` — same
    sampler/scheduler lists, same math, same behaviour as the standard
    node — with an added `summary` output reporting exactly what ran.
    `denoise` at 1.0 is a full generation from noise; lower values (e.g.
    0.3–0.6) are for img2img or refinement passes starting from an
    already-meaningful latent (see VAE Encode for turning an image into
    that starting latent).
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreKSampler.INPUT_TYPES()

    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("latent",  "summary")
    FUNCTION      = "run"

    def run(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        core = _CoreKSampler()
        result = core.sample(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image, denoise
        )
        out_latent = result[0]

        mode = "full generation (from noise)" if denoise >= 0.999 else "partial denoise (img2img/refine)"
        summary = (
            f"Sampler:    {sampler_name} / {scheduler}\n"
            f"Steps:      {steps}\n"
            f"CFG:        {cfg}\n"
            f"Denoise:    {denoise} — {mode}\n"
            f"Seed:       {seed}"
        )

        return (out_latent, summary)


NODE_CLASS_MAPPINGS = {
    "SimpleKSamplerNode": SimpleKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleKSamplerNode": "Simple KSampler 🌡️",
}
