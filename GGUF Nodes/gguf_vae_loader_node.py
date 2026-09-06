"""
TensorVizion ComfyUI Nodes
gguf_vae_loader_node.py — Loads a quantized VAE checkpoint in GGUF
format and returns a standard ComfyUI VAE. VAE weights are small enough
that GGUF quantization of them is uncommon in practice (most GGUF
release packages ship the VAE as plain safetensors and only quantize the
much larger UNet/text-encoder), but this node exists for the rare case
of a fully-GGUF-packaged model set, and for symmetry with the other
GGUF Nodes so one file format doesn't need two different loader
philosophies.

Same dequantize-to-fp32-at-load approach as the other GGUF loaders in
this pack — see gguf_diffusion_model_loader_node.py's docstring for the
VRAM trade-off vs a dedicated quantized-inference extension. VAEs are
run in fp32 here rather than fp16 by default since ComfyUI's own VAE
decode path is more precision-sensitive than UNet/CLIP inference.

Depends on the `gguf` PyPI package: `pip install gguf`.
"""

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import torch
except ImportError:
    torch = None

try:
    import comfy.sd
except ImportError:
    comfy = None


_GGUF_SEARCH_FOLDERS = ("vae", "vae_gguf", "checkpoints")


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


def _dequantize_gguf_state_dict(path, gguf_module, target_dtype):
    import numpy as np

    reader = gguf_module.GGUFReader(path)
    state_dict = {}
    skipped = []
    for t in reader.tensors:
        try:
            arr = gguf_module.dequantize(t.data, t.tensor_type).astype(np.float32)
        except NotImplementedError:
            skipped.append((t.name, t.tensor_type.name))
            continue
        state_dict[t.name] = torch.from_numpy(np.ascontiguousarray(arr)).to(target_dtype)
    return state_dict, skipped


class GGUFVaeLoaderNode:
    """
    Loads `gguf_name` (a VAE .gguf file), dequantizes every tensor to
    `weight_dtype`, and constructs a `comfy.sd.VAE` from the resulting
    state dict — output plugs into VAE Decode/Encode exactly like any
    other VAE loader in this pack.
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (_discover_gguf_filenames(),),
                "weight_dtype": (["fp32", "fp16"], {"default": "fp32"}),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "summary")
    FUNCTION = "run"

    def run(self, gguf_name, weight_dtype, path_override=""):
        if torch is None or comfy is None:
            return (None, "[TensorVizion] This node must run inside a ComfyUI environment "
                          "(torch / comfy.sd not importable).")

        try:
            import gguf as gguf_module
        except ImportError:
            return (None, "[TensorVizion] The `gguf` package is not installed. Run: pip install gguf")

        path = _resolve_path(gguf_name, path_override)
        if not path:
            return (None, f"[TensorVizion] File not found: {gguf_name!r} (checked model folders and path_override)")

        target_dtype = torch.float16 if weight_dtype == "fp16" else torch.float32

        try:
            state_dict, skipped = _dequantize_gguf_state_dict(path, gguf_module, target_dtype)
        except Exception as e:
            return (None, f"[TensorVizion] Failed to read/dequantize {os.path.basename(path)}: {e}")

        if not state_dict:
            return (None, f"[TensorVizion] No usable tensors decoded from {os.path.basename(path)}.")

        try:
            vae = comfy.sd.VAE(sd=state_dict)
        except Exception as e:
            return (None, f"[TensorVizion] comfy.sd.VAE rejected the dequantized state dict "
                          f"(this usually means the tensor key names don't match a VAE architecture "
                          f"ComfyUI recognizes): {e}")

        summary = (
            f"GGUF VAE loaded: {os.path.basename(path)}\n"
            f"Tensors dequantized: {len(state_dict)}\n"
            f"Tensors skipped (unsupported quant type): {len(skipped)}\n"
            f"Loaded weight dtype: {weight_dtype}"
        )
        return (vae, summary)


NODE_CLASS_MAPPINGS = {
    "GGUFVaeLoaderNode": GGUFVaeLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFVaeLoaderNode": "GGUF VAE Loader 🗝️",
}
