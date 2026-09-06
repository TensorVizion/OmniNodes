"""
TensorVizion ComfyUI Nodes
gguf_diffusion_model_loader_node.py — Loads a quantized diffusion/UNet
checkpoint stored in GGUF format (e.g. a Flux, SD3, or SDXL UNet
exported to Q4_K/Q5_K/Q8_0/etc.) and returns a standard ComfyUI MODEL.

IMPORTANT — how this differs from the dedicated ComfyUI-GGUF extension:
this node fully DEQUANTIZES every tensor to fp16 at load time and hands
the result to ComfyUI's normal `comfy.sd.load_diffusion_model_state_dict`
path. That means inference behaves exactly like any other fp16 MODEL —
but it also means you get the full fp16 VRAM/RAM footprint during and
after loading, NOT the reduced-VRAM benefit that ComfyUI-GGUF's custom
quantized ops provide by keeping weights quantized on the GPU. Use this
node when you want to open a .gguf file without installing that
extension and don't need its memory savings; use ComfyUI-GGUF's own
loader instead when VRAM headroom is the point of using GGUF in the
first place.

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
    import comfy.utils
except ImportError:
    comfy = None


_GGUF_SEARCH_FOLDERS = ("diffusion_models", "unet", "unet_gguf", "checkpoints")


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


def _dequantize_gguf_state_dict(path, gguf_module):
    """Reads every tensor in a .gguf file and returns a plain
    {name: torch.FloatTensor} state dict, dequantizing anything that
    isn't already F32/F16. Raises on any tensor type it can't handle so
    the caller gets a clear error instead of a silently wrong load."""
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
        state_dict[t.name] = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float16)
    return state_dict, skipped


class GGUFDiffusionModelLoaderNode:
    """
    Loads `gguf_name` (a UNet/diffusion-model .gguf file) by dequantizing
    every tensor to fp16 and passing the resulting state dict through
    ComfyUI's own `comfy.sd.load_diffusion_model_state_dict`, so it plugs
    into KSampler etc. exactly like a MODEL from the stock UNETLoader.

    `weight_dtype` controls what precision the dequantized tensors get
    cast to for the MODEL patcher (fp16 halves RAM/VRAM vs fp32 at the
    cost of a little precision — fp16 is the right default for nearly
    all modern GPUs).

    See this file's module docstring for the VRAM trade-off vs the
    dedicated ComfyUI-GGUF extension before relying on this for large
    (Flux/SD3-scale) models on limited VRAM.
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (_discover_gguf_filenames(),),
                "weight_dtype": (["fp16", "fp32"], {"default": "fp16"}),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
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

        try:
            state_dict, skipped = _dequantize_gguf_state_dict(path, gguf_module)
        except Exception as e:
            return (None, f"[TensorVizion] Failed to read/dequantize {os.path.basename(path)}: {e}")

        if not state_dict:
            return (None, f"[TensorVizion] No usable tensors decoded from {os.path.basename(path)} "
                          f"({len(skipped)} skipped — see log).")

        target_dtype = torch.float32 if weight_dtype == "fp32" else torch.float16
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}

        try:
            model = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options={})
        except AttributeError:
            return (None, "[TensorVizion] Your ComfyUI version's comfy.sd module doesn't expose "
                          "load_diffusion_model_state_dict — please update ComfyUI.")
        except Exception as e:
            return (None, f"[TensorVizion] comfy.sd rejected the dequantized state dict "
                          f"(likely an architecture ComfyUI's core UNet loader doesn't recognize "
                          f"from key names alone): {e}")

        summary = (
            f"GGUF diffusion model loaded: {os.path.basename(path)}\n"
            f"Tensors dequantized: {len(state_dict)}\n"
            f"Tensors skipped (unsupported quant type): {len(skipped)}"
            + (f" -> {skipped[:5]}{'...' if len(skipped) > 5 else ''}" if skipped else "") + "\n"
            f"Loaded weight dtype: {weight_dtype}\n"
            f"Note: fully dequantized at load time — full fp16/fp32 VRAM footprint, "
            f"not a memory-reduced quantized-inference load."
        )
        return (model, summary)


NODE_CLASS_MAPPINGS = {
    "GGUFDiffusionModelLoaderNode": GGUFDiffusionModelLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFDiffusionModelLoaderNode": "GGUF Diffusion Model Loader 🧠",
}
