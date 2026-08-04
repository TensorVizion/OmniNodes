"""
TensorVizion ComfyUI Nodes
smart_unloader.py — Frees VRAM by unloading currently-loaded models, while
passing its own input straight through unchanged. Designed to sit inline in
a workflow (e.g. right after a heavy generation step, before a second heavy
step that needs the freed memory) without breaking the graph the way a node
that swallows its inputs and returns None would.
"""

import gc
import torch
import comfy.model_management


class AlwaysEqualProxy(str):
    """
    A string subclass that always compares equal/not-equal to anything,
    used as an input/output type marker so this node accepts ANY ComfyUI
    type (MODEL, IMAGE, LATENT, CONDITIONING, etc.) on its passthrough
    socket without ComfyUI's newer stricter type-checking rejecting the
    connection. This is the standard workaround used by other "any type"
    utility nodes in the ComfyUI ecosystem (e.g. passthrough/reroute nodes).
    """
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


class SmartUnloader:
    """
    Unloads models from VRAM at the point this node executes, then passes
    its `passthrough` input straight through untouched. Connect any output
    from earlier in your workflow (an IMAGE, LATENT, MODEL, or anything
    else) into `passthrough` and route this node's output onward — that
    dependency edge is what forces ComfyUI to run the unload at this exact
    point in the graph, the same trick used by other unload-model community
    nodes.

    `unload_all` additionally calls ComfyUI's full unload_all_models(),
    clearing every currently loaded model/CLIP/VAE from memory, not just
    freeing unused cache. Use this between two heavy, unrelated stages of a
    workflow (e.g. an upscale pass that doesn't need the base checkpoint
    anymore).
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (ANY_TYPE,),
                "unload_all":  ("BOOLEAN", {"default": True}),
                "empty_cache": ("BOOLEAN", {"default": True}),
                "run_gc":      ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = (ANY_TYPE, "STRING")
    RETURN_NAMES  = ("output", "summary")
    FUNCTION      = "unload"

    def unload(self, passthrough, unload_all, empty_cache, run_gc):
        freed_before = 0
        freed_after = 0
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            freed_before = torch.cuda.memory_allocated()

        actions = []

        if unload_all:
            comfy.model_management.unload_all_models()
            actions.append("unloaded all models")

        if run_gc:
            gc.collect()
            actions.append("ran Python GC")

        if empty_cache:
            comfy.model_management.soft_empty_cache()
            if cuda_available:
                torch.cuda.empty_cache()
            actions.append("emptied CUDA cache")

        if cuda_available:
            freed_after = torch.cuda.memory_allocated()
            delta_gb = (freed_before - freed_after) / 1e9
            mem_line = f"VRAM: {freed_before/1e9:.2f} GB -> {freed_after/1e9:.2f} GB (freed {delta_gb:.2f} GB)"
        else:
            mem_line = "CUDA not available — skipped VRAM accounting"

        summary = f"Actions: {', '.join(actions) if actions else 'none'}\n{mem_line}"

        return (passthrough, summary)


NODE_CLASS_MAPPINGS = {
    "SmartUnloader": SmartUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartUnloader": "Smart Unloader 🧹",
}
