"""
TensorVizion ComfyUI Nodes
quick_lora_stacker.py — A lightweight 3-slot LoRA stacker with a single
shared weight per slot (rather than separate model/clip weights), for quick
everyday stacking where you don't need LoRA Stack's full 5-slot,
dual-weight control surface.
"""

import folder_paths
import comfy.utils
import comfy.sd


class QuickLoRAStacker:
    """
    Applies up to 3 LoRAs in sequence, each with one combined strength value
    applied to both model and clip. A faster-to-configure alternative to
    LoRA Stack for the common case where you don't need independent
    model/clip weighting per LoRA.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip":  ("CLIP",),
                "lora_1":    (lora_list,),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_2":    (lora_list,),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_3":    (lora_list,),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES  = ("model", "clip", "stack_summary")
    FUNCTION      = "apply_stack"

    def apply_stack(
        self,
        model, clip,
        lora_1, strength_1,
        lora_2, strength_2,
        lora_3, strength_3,
    ):
        slots = [
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
        ]

        applied = []
        model_out = model
        clip_out  = clip

        for lora_name, strength in slots:
            if lora_name == "None":
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                print(f"[QuickLoRAStacker] WARNING: LoRA not found — {lora_name}")
                continue
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out, clip_out, lora, strength, strength
            )
            applied.append(f"{lora_name}  (strength: {strength:.2f})")

        summary = "\n".join(applied) if applied else "No LoRAs applied"
        return (model_out, clip_out, summary)


NODE_CLASS_MAPPINGS = {
    "QuickLoRAStacker": QuickLoRAStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuickLoRAStacker": "Quick LoRA Stacker ⚡",
}
