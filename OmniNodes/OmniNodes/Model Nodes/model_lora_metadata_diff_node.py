"""
TensorVizion ComfyUI Nodes
model_lora_metadata_diff_node.py — Loads two LoRA files and diffs their
structural metadata (rank, alpha, target modules, key/param counts) and
embedded safetensors header metadata (base model, trigger words, training
params where present) side-by-side. LoRA Info Inspector reports one LoRA
in isolation; this reports two and highlights where they differ — built
for auditing a growing LoRA library before batching uploads to CivitAI.
"""

import json

import folder_paths
import torch


class ModelLoRAMetadataDiffNode:
    """
    Loads `lora_name_a` and `lora_name_b` and produces a side-by-side diff:
      - structural: rank, alpha, target module prefixes, key count, param count
      - header metadata: any embedded safetensors metadata keys (base model
        tag, trigger words, network args, etc.) that CivitAI / kohya-style
        trainers commonly write, flagging keys present in one file but not
        the other, or present in both with different values.

    Useful before uploading a batch of LoRAs to catch cases like "these two
    were trained on different base models" or "one lost its trigger-word
    metadata".
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name_a": (loras,),
                "lora_name_b": (loras,),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("diff_report", "structurally_compatible")
    FUNCTION = "diff"

    def _structural_info(self, lora):
        keys = list(lora.keys())
        key_count = len(keys)

        rank = 0
        alpha = 0.0
        for k in keys:
            if "lora_down" in k and isinstance(lora[k], torch.Tensor) and lora[k].ndim >= 2:
                rank = int(lora[k].shape[0])
                break
        for k in keys:
            if "alpha" in k:
                try:
                    alpha = float(lora[k].item())
                    break
                except Exception:
                    pass

        module_set = set()
        for k in keys:
            parts = k.split(".")
            if len(parts) >= 2:
                module_set.add(parts[0])

        param_count = sum(v.numel() for v in lora.values() if isinstance(v, torch.Tensor))

        return {
            "key_count": key_count,
            "rank": rank,
            "alpha": alpha,
            "modules": module_set,
            "param_count": param_count,
        }

    def _read_header_metadata(self, path):
        """Reads the raw safetensors JSON header's top-level '__metadata__'
        block without pulling in a dependency beyond what comfy.utils
        already needs -- safetensors files store this as a plain JSON
        header at the start of the file."""
        metadata = {}
        try:
            with open(path, "rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode("utf-8"))
                metadata = header.get("__metadata__", {}) or {}
        except Exception:
            pass
        return metadata

    def diff(self, lora_name_a, lora_name_b):
        import comfy.utils

        path_a = folder_paths.get_full_path("loras", lora_name_a)
        path_b = folder_paths.get_full_path("loras", lora_name_b)

        lora_a = comfy.utils.load_torch_file(path_a, safe_load=True)
        lora_b = comfy.utils.load_torch_file(path_b, safe_load=True)

        info_a = self._structural_info(lora_a)
        info_b = self._structural_info(lora_b)

        meta_a = self._read_header_metadata(path_a)
        meta_b = self._read_header_metadata(path_b)

        lines = [f"LoRA A: {lora_name_a}", f"LoRA B: {lora_name_b}", ""]
        lines.append("-- Structural --")

        struct_ok = True
        for field in ("key_count", "rank", "alpha", "param_count"):
            va, vb = info_a[field], info_b[field]
            match = "==" if va == vb else "!="
            if field in ("rank",) and va != vb:
                struct_ok = False
            lines.append(f"  {field:12s}: A={va}  {match}  B={vb}")

        only_a = info_a["modules"] - info_b["modules"]
        only_b = info_b["modules"] - info_a["modules"]
        lines.append(f"  target_modules match: {'yes' if not only_a and not only_b else 'no'}")
        if only_a:
            lines.append(f"    only in A: {', '.join(sorted(only_a))}")
        if only_b:
            lines.append(f"    only in B: {', '.join(sorted(only_b))}")

        lines.append("")
        lines.append("-- Embedded header metadata --")
        all_keys = sorted(set(meta_a.keys()) | set(meta_b.keys()))
        if not all_keys:
            lines.append("  (no __metadata__ block found in either file)")
        for k in all_keys:
            va = meta_a.get(k, "<missing>")
            vb = meta_b.get(k, "<missing>")
            marker = "==" if va == vb else "!="
            lines.append(f"  {k}: A={va}  {marker}  B={vb}")

        base_a = meta_a.get("ss_sd_model_name") or meta_a.get("ss_base_model_version") or "unknown"
        base_b = meta_b.get("ss_sd_model_name") or meta_b.get("ss_base_model_version") or "unknown"
        if base_a != "unknown" and base_b != "unknown" and base_a != base_b:
            lines.append("")
            lines.append(f"  WARNING: base models differ ({base_a} vs {base_b})")
            struct_ok = False

        report = "\n".join(lines)
        return (report, struct_ok)


NODE_CLASS_MAPPINGS = {
    "ModelLoRAMetadataDiffNode": ModelLoRAMetadataDiffNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelLoRAMetadataDiffNode": "LoRA Metadata Diff 🆚",
}
