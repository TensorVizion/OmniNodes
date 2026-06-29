"""
TensorVizion ComfyUI Nodes
model_block_freeze_node.py — Selectively freezes (zeroes patch weights for)
specified block ranges of a diffusion model, enabling partial fine-tune
style inference where only certain depth layers are active.
"""


class ModelBlockFreezeNode:
    """
    Applies zero-weight patches to selected input, middle, or output blocks
    of a diffusion model, effectively 'freezing' their contribution during
    inference. Useful for style experiments and ablation workflows.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("MODEL",),
                "freeze_input":    ("BOOLEAN", {"default": False}),
                "freeze_middle":   ("BOOLEAN", {"default": False}),
                "freeze_output":   ("BOOLEAN", {"default": False}),
                "input_block_start":  ("INT", {"default": 0,  "min": 0, "max": 11, "step": 1}),
                "input_block_end":    ("INT", {"default": 3,  "min": 0, "max": 11, "step": 1}),
                "output_block_start": ("INT", {"default": 8,  "min": 0, "max": 11, "step": 1}),
                "output_block_end":   ("INT", {"default": 11, "min": 0, "max": 11, "step": 1}),
                "freeze_strength":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("MODEL", "STRING")
    RETURN_NAMES  = ("model", "freeze_summary")
    FUNCTION      = "freeze_blocks"

    def freeze_blocks(
        self,
        model,
        freeze_input, freeze_middle, freeze_output,
        input_block_start, input_block_end,
        output_block_start, output_block_end,
        freeze_strength,
    ):
        m = model.clone()
        kp = model.get_key_patches("diffusion_model.")

        frozen_keys = []

        for key in kp.keys():
            should_freeze = False

            if freeze_input and "input_blocks" in key:
                # Extract block index
                try:
                    block_idx = int(key.split("input_blocks.")[1].split(".")[0])
                    if input_block_start <= block_idx <= input_block_end:
                        should_freeze = True
                except (IndexError, ValueError):
                    pass

            if freeze_middle and "middle_block" in key:
                should_freeze = True

            if freeze_output and "output_blocks" in key:
                try:
                    block_idx = int(key.split("output_blocks.")[1].split(".")[0])
                    if output_block_start <= block_idx <= output_block_end:
                        should_freeze = True
                except (IndexError, ValueError):
                    pass

            if should_freeze:
                # Add patch with freeze_strength weighting (0 = fully frozen)
                m.add_patches({key: kp[key]}, freeze_strength, 1.0 - freeze_strength)
                frozen_keys.append(key)

        lines = [f"Frozen {len(frozen_keys)} key(s)  (strength={freeze_strength:.2f})"]
        if freeze_input:
            lines.append(f"  Input blocks {input_block_start}–{input_block_end}: frozen")
        if freeze_middle:
            lines.append("  Middle block: frozen")
        if freeze_output:
            lines.append(f"  Output blocks {output_block_start}–{output_block_end}: frozen")
        if not frozen_keys:
            lines.append("  (nothing frozen — enable at least one block group)")

        return (m, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "ModelBlockFreezeNode": ModelBlockFreezeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelBlockFreezeNode": "Model Block Freeze 🧊",
}
