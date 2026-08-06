"""
TensorVizion ComfyUI Nodes
lora_weight_sweep_node.py — Takes a MODEL/CLIP + one LoRA and a strength
range, and outputs a REAL ComfyUI list of (model, clip, label) at each
strength step — using OUTPUT_IS_LIST so downstream nodes (KSampler, VAE
Decode, etc.) automatically execute once per strength value in the
sweep, no manual copy-pasting of the same sampler chain at different
strengths. Pairs naturally with Image Grid Compare: collect the
resulting images into a batch and wire the `labels` output straight
into its `labels` field for an automatically-labeled strength-sweep grid.
"""

import comfy.sd
import comfy.utils
import folder_paths


class LoRAWeightSweepNode:
    """
    Applies `lora_name` to `model`/`clip` at each strength from
    `start_strength` to `end_strength` in `steps` increments (inclusive
    of both ends), producing one patched (model, clip) pair per step.

    Both `model` output and `clip` output are REAL ComfyUI output lists
    (OUTPUT_IS_LIST) — connecting either into a downstream node (e.g.
    KSampler) causes that node to run once per strength value
    automatically, collecting results as it goes, rather than needing
    `steps` separate copies of your sampler chain wired in parallel.

    `labels` is a matching list of strings ("strength=0.20", etc.) for
    exactly this reason — wire it into Image Grid Compare's `labels`
    input (which also needs List processing enabled on that connection,
    ComfyUI will prompt for this automatically) to get an automatically
    labeled comparison grid across the whole sweep in one wire-up.
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (loras,),
                "start_strength": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "end_strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "steps": ("INT", {"default": 5, "min": 2, "max": 50}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "labels")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "sweep"

    def sweep(self, model, clip, lora_name, start_strength, end_strength, steps):
        path = folder_paths.get_full_path("loras", lora_name)
        if not path:
            # Return a single-item list so downstream List processing
            # still gets exactly one (unpatched) run rather than a
            # length-0 list, which some nodes handle poorly.
            return ([model], [clip], [f"ERROR: LoRA not found — {lora_name}"])

        lora_sd = comfy.utils.load_torch_file(path, safe_load=True)

        models, clips, labels = [], [], []
        if steps <= 1:
            strength_values = [start_strength]
        else:
            step_size = (end_strength - start_strength) / (steps - 1)
            strength_values = [start_strength + i * step_size for i in range(steps)]

        for strength in strength_values:
            m_out, c_out = comfy.sd.load_lora_for_models(model, clip, lora_sd, strength, strength)
            models.append(m_out)
            clips.append(c_out)
            labels.append(f"strength={strength:.2f}")

        return (models, clips, labels)


NODE_CLASS_MAPPINGS = {
    "LoRAWeightSweepNode": LoRAWeightSweepNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAWeightSweepNode": "Multi-LoRA Weight Sweep 📶",
}
