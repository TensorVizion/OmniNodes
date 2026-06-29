"""
TensorVizion ComfyUI Nodes
lora_info_node.py — Inspects a LoRA file and returns metadata: key count,
rank, alpha, target modules, and estimated parameter count.
"""

import folder_paths
import torch


class LoRAInfoNode:
    """
    Loads a LoRA file and extracts structural metadata without applying it to
    any model. Useful for auditing LoRAs before use and building debug pipelines.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES  = ("STRING", "INT",   "INT",   "FLOAT",  "STRING")
    RETURN_NAMES  = ("summary", "key_count", "rank", "alpha", "target_modules")
    FUNCTION      = "inspect_lora"

    def inspect_lora(self, lora_name):
        import comfy.utils

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora      = comfy.utils.load_torch_file(lora_path, safe_load=True)

        keys       = list(lora.keys())
        key_count  = len(keys)

        # Detect rank from lora_down keys
        rank  = 0
        alpha = 0.0
        for k in keys:
            if "lora_down" in k and lora[k].ndim >= 2:
                rank = int(lora[k].shape[0])
                break
        for k in keys:
            if "alpha" in k:
                try:
                    alpha = float(lora[k].item())
                    break
                except Exception:
                    pass

        # Collect unique target module prefixes
        module_set = set()
        for k in keys:
            parts = k.split(".")
            if len(parts) >= 2:
                module_set.add(parts[0])

        target_modules = ", ".join(sorted(module_set))

        # Estimate parameter count
        param_count = sum(v.numel() for v in lora.values() if isinstance(v, torch.Tensor))

        summary = (
            f"LoRA: {lora_name}\n"
            f"Keys:       {key_count}\n"
            f"Rank:       {rank}\n"
            f"Alpha:      {alpha:.4f}\n"
            f"Params:     ~{param_count:,}\n"
            f"Modules:    {target_modules}"
        )

        return (summary, key_count, rank, alpha, target_modules)


NODE_CLASS_MAPPINGS = {
    "LoRAInfoNode": LoRAInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAInfoNode": "LoRA Info Inspector 🔬",
}
