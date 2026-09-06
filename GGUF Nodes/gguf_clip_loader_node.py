"""
TensorVizion ComfyUI Nodes
gguf_clip_loader_node.py — Loads one or two quantized text-encoder
checkpoints in GGUF format (CLIP-L, CLIP-G, T5-XXL, etc. — e.g. a T5-XXL
GGUF export used to cut Flux/SD3 text-encoder VRAM) and returns a
standard ComfyUI CLIP object.

Same trade-off as GGUF Diffusion Model Loader: tensors are fully
dequantized to fp16 at load time and handed to ComfyUI's own
`comfy.sd.load_text_encoder_state_dicts`, so downstream behaviour matches
a normal CLIP loader exactly, but without the reduced-VRAM benefit a
quantized-inference-native extension (like ComfyUI-GGUF) provides — see
gguf_diffusion_model_loader_node.py's docstring for the full rationale.

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


_GGUF_SEARCH_FOLDERS = ("clip", "clip_gguf", "text_encoders", "checkpoints")

# Mirrors the type combo ComfyUI's own core CLIPLoader/DualCLIPLoader
# expose, so a GGUF text encoder slots into the same mental model.
_CLIP_TYPES = ["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "flux", "hunyuan_video", "mochi", "ltxv"]


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
    if not gguf_name or gguf_name in ("none", "<none found in model folders — use path_override>"):
        return None
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


class GGUFClipLoaderNode:
    """
    Loads `clip_name1` (required) and optionally `clip_name2` — both
    GGUF text-encoder files — dequantizes them to fp16, and passes the
    resulting state dict(s) to `comfy.sd.load_text_encoder_state_dicts`
    along with `clip_type`, the same parameter ComfyUI's own CLIPLoader/
    DualCLIPLoader use to pick the right tokenizer/encoder wiring for the
    target model family. Leave `clip_name2` as the placeholder entry to
    load a single encoder (e.g. just T5-XXL).
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        names = _discover_gguf_filenames()
        names_with_none = ["none"] + names
        return {
            "required": {
                "clip_name1": (names,),
                "clip_type": (_CLIP_TYPES, {"default": "stable_diffusion"}),
            },
            "optional": {
                "clip_name2": (names_with_none, {"default": "none"}),
                "path_override1": ("STRING", {"default": ""}),
                "path_override2": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "summary")
    FUNCTION = "run"

    def run(self, clip_name1, clip_type, clip_name2="none", path_override1="", path_override2=""):
        if torch is None or comfy is None:
            return (None, "[TensorVizion] This node must run inside a ComfyUI environment "
                          "(torch / comfy.sd not importable).")

        try:
            import gguf as gguf_module
        except ImportError:
            return (None, "[TensorVizion] The `gguf` package is not installed. Run: pip install gguf")

        path1 = _resolve_path(clip_name1, path_override1)
        if not path1:
            return (None, f"[TensorVizion] File not found: {clip_name1!r} (checked model folders and path_override1)")

        paths = [path1]
        path2 = _resolve_path(clip_name2, path_override2)
        if path2:
            paths.append(path2)

        state_dicts = []
        total_skipped = 0
        for p in paths:
            try:
                sd, skipped = _dequantize_gguf_state_dict(p, gguf_module, torch.float16)
            except Exception as e:
                return (None, f"[TensorVizion] Failed to read/dequantize {os.path.basename(p)}: {e}")
            if not sd:
                return (None, f"[TensorVizion] No usable tensors decoded from {os.path.basename(p)}.")
            state_dicts.append(sd)
            total_skipped += len(skipped)

        try:
            clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip = comfy.sd.load_text_encoder_state_dicts(state_dicts, clip_type=clip_type_enum, model_options={})
        except AttributeError:
            return (None, "[TensorVizion] Your ComfyUI version's comfy.sd module doesn't expose "
                          "load_text_encoder_state_dicts/CLIPType the way this node expects — please update ComfyUI.")
        except Exception as e:
            return (None, f"[TensorVizion] comfy.sd rejected the dequantized state dict(s): {e}")

        summary = (
            f"GGUF CLIP loaded: {', '.join(os.path.basename(p) for p in paths)}\n"
            f"clip_type: {clip_type}\n"
            f"Encoders loaded: {len(paths)}\n"
            f"Tensors skipped across all files (unsupported quant type): {total_skipped}\n"
            f"Note: fully dequantized to fp16 at load time, not a memory-reduced quantized load."
        )
        return (clip, summary)


NODE_CLASS_MAPPINGS = {
    "GGUFClipLoaderNode": GGUFClipLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFClipLoaderNode": "GGUF CLIP Loader 📝",
}
