"""
TensorVizion ComfyUI Nodes
model_info_inspector.py — Reads a checkpoint file and returns structural
metadata (key count, detected architecture, precision, parameter count)
without fully loading it into a MODEL/CLIP/VAE. Useful for auditing an
unfamiliar checkpoint before committing to a full load.
"""

import folder_paths
import comfy.utils
import torch


class ModelInfoInspector:
    """
    Loads a checkpoint's raw state dict (not a full ComfyUI MODEL/CLIP/VAE)
    and reports structural metadata: total key count, an architecture guess
    based on key-name signatures, detected weight precision, and an
    estimated parameter count.

    This is intentionally lightweight — it does not run
    load_checkpoint_guess_config, so it works even on checkpoints ComfyUI
    can't fully load (e.g. missing a config match), and it's much faster
    than a full load since no submodules are constructed.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES  = ("STRING", "INT",       "STRING",       "STRING")
    RETURN_NAMES  = ("summary", "key_count", "architecture", "precision")
    FUNCTION      = "inspect_model"

    @staticmethod
    def _guess_architecture(keys):
        key_set = set(keys)
        joined = " ".join(list(key_set)[:2000])  # cap for speed on huge dicts

        if any("double_blocks" in k for k in key_set):
            return "FLUX (double_blocks detected)"
        if any("model.diffusion_model.joint_blocks" in k for k in key_set):
            return "SD3 (joint_blocks detected)"
        if any("conditioner.embedders.1" in k for k in key_set):
            return "SDXL (dual text encoder detected)"
        if any("cond_stage_model.transformer" in k for k in key_set) and \
           any("model.diffusion_model.input_blocks" in k for k in key_set):
            return "SD 1.x / 2.x (single text encoder + UNet input_blocks)"
        if any("model.diffusion_model" in k for k in key_set):
            return "Stable Diffusion-family UNet (architecture unconfirmed)"
        return "Unrecognized / non-diffusion checkpoint"

    @staticmethod
    def _guess_precision(state_dict):
        dtypes = set()
        for v in state_dict.values():
            if isinstance(v, torch.Tensor):
                dtypes.add(str(v.dtype))
            if len(dtypes) > 3:
                break
        if not dtypes:
            return "unknown"
        return ", ".join(sorted(dtypes))

    def inspect_model(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path is None:
            raise FileNotFoundError(f"[ModelInfoInspector] Checkpoint not found: {ckpt_name}")

        state_dict = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        keys = list(state_dict.keys())
        key_count = len(keys)

        architecture = self._guess_architecture(keys)
        precision = self._guess_precision(state_dict)

        param_count = sum(
            v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
        )
        param_b = param_count / 1e9

        summary = (
            f"Checkpoint:    {ckpt_name}\n"
            f"Keys:          {key_count:,}\n"
            f"Architecture:  {architecture}\n"
            f"Precision:     {precision}\n"
            f"Params:        ~{param_b:.2f}B ({param_count:,})"
        )

        return (summary, key_count, architecture, precision)


NODE_CLASS_MAPPINGS = {
    "ModelInfoInspector": ModelInfoInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelInfoInspector": "Model Info Inspector 🔬",
}
