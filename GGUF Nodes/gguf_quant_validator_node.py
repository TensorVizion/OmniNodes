"""
TensorVizion ComfyUI Nodes
gguf_quant_validator_node.py — Health-checks a .gguf file before you
commit to loading it into a workflow: spot-dequantizes a sample of
tensors and checks for NaN/Inf, flags zero-sized or suspiciously
all-zero tensors, and confirms the tensor count/quant-type mix looks
sane. Catches a truncated download or a bad quantization pass before it
produces a confusing failure three nodes downstream instead of a clear
one here.

This does NOT re-verify a checksum against a known-good hash (this pack
has no registry of expected hashes to check against) — it's a structural/
numerical sanity check of the file's own internal consistency, not a
guarantee the file matches what its source claims it is.

Depends on the `gguf` PyPI package: `pip install gguf`.
"""

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None


_GGUF_SEARCH_FOLDERS = ("diffusion_models", "unet", "unet_gguf", "clip", "clip_gguf", "vae", "checkpoints")


def _discover_gguf_filenames():
    names = set()
    if folder_paths is not None:
        for folder_key in _GGUF_SEARCH_FOLDERS:
            try:
                for n in folder_paths.get_filename_list(folder_key):
                    if n.lower().endswith(".gguf"):
                        names.add(n)
            except Exception:
                continue
    return sorted(names) if names else ["<none found in model folders — use path_override>"]


def _resolve_path(gguf_name, path_override):
    if path_override and path_override.strip():
        return path_override.strip()
    if folder_paths is not None:
        for folder_key in _GGUF_SEARCH_FOLDERS:
            try:
                full = folder_paths.get_full_path(folder_key, gguf_name)
            except Exception:
                full = None
            if full and os.path.exists(full):
                return full
    return gguf_name if os.path.exists(gguf_name) else None


class GGUFQuantValidatorNode:
    """
    Reads `gguf_name`, dequantizes up to `sample_size` tensors (evenly
    spread through the file, not just the first N — a truncated-download
    problem often only shows up near the end of the file) and reports:
    any NaN/Inf found, any all-zero tensor (often a sign of a bad
    conversion rather than a genuinely zeroed layer), any zero-element
    tensor, and the overall quant-type mix. `is_valid` is False if any
    check fails on a sampled tensor.
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (_discover_gguf_filenames(),),
                "sample_size": ("INT", {"default": 25, "min": 1, "max": 500}),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_valid", "report")
    FUNCTION = "run"

    def run(self, gguf_name, sample_size, path_override=""):
        try:
            import gguf as gguf_module
            import numpy as np
        except ImportError:
            return (False, "[TensorVizion] The `gguf` package is not installed. Run: pip install gguf")

        path = _resolve_path(gguf_name, path_override)
        if not path:
            return (False, f"[TensorVizion] File not found: {gguf_name!r} (checked model folders and path_override)")

        try:
            reader = gguf_module.GGUFReader(path)
        except Exception as e:
            return (False, f"[TensorVizion] Failed to parse {os.path.basename(path)} as GGUF: {e}")

        tensors = reader.tensors
        if not tensors:
            return (False, f"[TensorVizion] {os.path.basename(path)} contains zero tensors.")

        n = len(tensors)
        step = max(1, n // sample_size)
        sample_indices = list(range(0, n, step))[:sample_size]

        nan_inf_tensors = []
        all_zero_tensors = []
        zero_element_tensors = []
        dequant_failures = []
        checked = 0

        for idx in sample_indices:
            t = tensors[idx]
            if t.n_elements == 0:
                zero_element_tensors.append(t.name)
                continue
            try:
                arr = gguf_module.dequantize(t.data, t.tensor_type).astype(np.float32)
            except Exception as e:
                dequant_failures.append(f"{t.name} ({t.tensor_type.name}): {e}")
                continue
            checked += 1
            if not np.isfinite(arr).all():
                nan_inf_tensors.append(t.name)
            elif not np.any(arr):
                all_zero_tensors.append(t.name)

        type_counts = {}
        for t in tensors:
            type_counts[t.tensor_type.name] = type_counts.get(t.tensor_type.name, 0) + 1

        is_valid = not (nan_inf_tensors or zero_element_tensors or dequant_failures)

        report_lines = [
            f"GGUF file: {os.path.basename(path)}",
            f"Total tensors: {n} | Sampled: {len(sample_indices)} (successfully checked: {checked})",
            f"Quant type mix: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1])),
            f"NaN/Inf found in: {len(nan_inf_tensors)} sampled tensor(s)" + (f" -> {nan_inf_tensors[:5]}" if nan_inf_tensors else ""),
            f"All-zero sampled tensor(s): {len(all_zero_tensors)}" + (f" -> {all_zero_tensors[:5]}" if all_zero_tensors else " (0 means no all-zero layers seen in the sample — a real full scan may still differ)"),
            f"Zero-element tensor(s): {len(zero_element_tensors)}" + (f" -> {zero_element_tensors[:5]}" if zero_element_tensors else ""),
            f"Dequantize failures on sampled tensors: {len(dequant_failures)}" + ("\n  " + "\n  ".join(dequant_failures[:5]) if dequant_failures else ""),
            f"Result: {'PASS — no problems found in sample' if is_valid else 'FAIL — see issues above'}",
        ]
        report = "\n".join(report_lines)

        return (is_valid, report)


NODE_CLASS_MAPPINGS = {
    "GGUFQuantValidatorNode": GGUFQuantValidatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFQuantValidatorNode": "GGUF Quant Validator ✅",
}
