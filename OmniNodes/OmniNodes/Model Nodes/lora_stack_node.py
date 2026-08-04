"""
TensorVizion ComfyUI Nodes
lora_stack_node.py — Builds a stack of up to 5 LoRAs with individual
model/clip weights, then applies them to a model+clip pair in one pass.
"""

import folder_paths


class LoRAStackNode:
    """
    Lets you chain up to 5 LoRAs with individual model_weight and clip_weight
    controls, outputting a patched MODEL and CLIP. Acts as a cleaner alternative
    to chaining multiple Load LoRA nodes.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip":  ("CLIP",),
                # LoRA slot 1
                "lora_1":         (lora_list,),
                "model_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_1":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # LoRA slot 2
                "lora_2":         (lora_list,),
                "model_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_2":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # LoRA slot 3
                "lora_3":         (lora_list,),
                "model_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_3":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # LoRA slot 4
                "lora_4":         (lora_list,),
                "model_weight_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_4":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                # LoRA slot 5
                "lora_5":         (lora_list,),
                "model_weight_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_5":  ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES  = ("model", "clip", "stack_summary")
    FUNCTION      = "apply_lora_stack"

    def apply_lora_stack(
        self,
        model, clip,
        lora_1, model_weight_1, clip_weight_1,
        lora_2, model_weight_2, clip_weight_2,
        lora_3, model_weight_3, clip_weight_3,
        lora_4, model_weight_4, clip_weight_4,
        lora_5, model_weight_5, clip_weight_5,
    ):
        import comfy.utils

        slots = [
            (lora_1, model_weight_1, clip_weight_1),
            (lora_2, model_weight_2, clip_weight_2),
            (lora_3, model_weight_3, clip_weight_3),
            (lora_4, model_weight_4, clip_weight_4),
            (lora_5, model_weight_5, clip_weight_5),
        ]

        applied = []
        model_out = model
        clip_out  = clip

        for lora_name, mw, cw in slots:
            if lora_name == "None":
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                print(f"[LoRAStackNode] WARNING: LoRA not found — {lora_name}")
                continue
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_out, clip_out = comfy.sd.load_lora_for_models(
                model_out, clip_out, lora, mw, cw
            )
            applied.append(f"{lora_name}  (m:{mw:.2f} | c:{cw:.2f})")

        summary = "\n".join(applied) if applied else "No LoRAs applied"
        return (model_out, clip_out, summary)


NODE_CLASS_MAPPINGS = {
    "LoRAStackNode": LoRAStackNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAStackNode": "LoRA Stack 🗂️",
}
