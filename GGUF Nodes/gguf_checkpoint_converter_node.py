"""
TensorVizion ComfyUI Nodes
gguf_checkpoint_converter_node.py — Converts a standard checkpoint
(.safetensors/.ckpt/.pt loaded via comfy.utils) into a .gguf file,
quantizing every tensor to the chosen GGML type. The offline-conversion
counterpart to the other GGUF Nodes here — use this to make your own
GGUF files instead of only consuming ones you downloaded.

Quantization is delegated to the `gguf` package's own `gguf.quants`
implementations, so files this node writes use the real GGML block
formats (not a custom scheme only this pack understands) and should
round-trip correctly through this pack's own GGUF loaders as well as
other GGUF-aware tools.

GGML block quantization requires each tensor's last dimension to be
divisible by that type's block size (32 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0;
256 for the K-quants). Tensors that don't satisfy this — commonly small
1-D bias/norm vectors — are automatically kept at F16 instead of
skipped, so the output file is always complete and loadable; the
skip/fallback count is reported in `report` so you know how much of the
model actually got quantized.

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
    import comfy.utils
except ImportError:
    comfy = None


_SOURCE_FOLDERS = ("checkpoints", "diffusion_models", "unet", "clip", "vae")

# Block-quant types with a fixed 32-element block that a straightforward
# tensor (no special calibration data needed) can always be quantized
# to, in addition to F16/F32. K-quants are intentionally left out here —
# they require a rounder file-level convert pipeline (importance-matrix
# aware block selection) than a single generic tensor-by-tensor pass can
# do faithfully, so exposing them would risk silently-worse quality
# without ever telling the user.
_QUANT_CHOICES = ["F16", "F32", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0"]
_BLOCK_SIZE = 32


def _discover_source_filenames():
    names = set()
    if folder_paths is not None:
        for folder_key in _SOURCE_FOLDERS:
            try:
                names.update(folder_paths.get_filename_list(folder_key))
            except Exception:
                continue
    return sorted(names) if names else ["<none found in model folders — use path_override>"]


def _resolve_source_path(name, path_override):
    if path_override and path_override.strip():
        return path_override.strip()
    if folder_paths is not None:
        for folder_key in _SOURCE_FOLDERS:
            try:
                full = folder_paths.get_full_path(folder_key, name)
            except Exception:
                full = None
            if full and os.path.exists(full):
                return full
    return name if os.path.exists(name) else None


def _output_dir():
    if folder_paths is not None:
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    return os.getcwd()


class GGUFCheckpointConverterNode:
    """
    Loads `source_name` (any checkpoint comfy.utils.load_torch_file can
    read) and writes every tensor into a new .gguf file at
    `output_filename` (saved under ComfyUI's output directory, or an
    absolute path if you supply one), quantized to `quant_type`.

    `key_filter` (optional) keeps only tensor names containing this
    substring — e.g. `model.diffusion_model.` to export just the UNet
    out of a full checkpoint that also bundles CLIP/VAE, cutting output
    size when you only need one component in GGUF form.
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_name": (_discover_source_filenames(),),
                "quant_type": (_QUANT_CHOICES, {"default": "Q8_0"}),
                "output_filename": ("STRING", {"default": "converted_model.gguf"}),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
                "key_filter": ("STRING", {"default": ""}),
                "architecture_name": ("STRING", {"default": "comfyui-export"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "run"

    def run(self, source_name, quant_type, output_filename, path_override="", key_filter="", architecture_name="comfyui-export"):
        if comfy is None or torch is None:
            return ("", "[TensorVizion] This node must run inside a ComfyUI environment (torch / comfy.utils not importable).")

        try:
            import gguf as gguf_module
            import numpy as np
        except ImportError:
            return ("", "[TensorVizion] The `gguf` package is not installed. Run: pip install gguf")

        src_path = _resolve_source_path(source_name, path_override)
        if not src_path:
            return ("", f"[TensorVizion] Source file not found: {source_name!r} (checked model folders and path_override)")

        try:
            state_dict = comfy.utils.load_torch_file(src_path, safe_load=True)
        except Exception as e:
            return ("", f"[TensorVizion] Failed to load {os.path.basename(src_path)}: {e}")

        if key_filter.strip():
            state_dict = {k: v for k, v in state_dict.items() if key_filter.strip() in k}
            if not state_dict:
                return ("", f"[TensorVizion] key_filter {key_filter!r} matched no tensors in {os.path.basename(src_path)}.")

        if not output_filename.strip():
            output_filename = "converted_model.gguf"
        if not output_filename.lower().endswith(".gguf"):
            output_filename += ".gguf"
        out_dir = self._resolve_out_dir(output_filename)
        out_path = output_filename if os.path.isabs(output_filename) else os.path.join(out_dir, os.path.basename(output_filename))

        try:
            qtype_enum = getattr(gguf_module.GGMLQuantizationType, quant_type)
        except AttributeError:
            return ("", f"[TensorVizion] Unknown quant_type {quant_type!r}.")

        writer = gguf_module.GGUFWriter(out_path, arch=architecture_name or "comfyui-export")
        writer.add_name(os.path.splitext(os.path.basename(src_path))[0])

        quantized_count = 0
        fallback_count = 0
        error_count = 0
        errors = []

        for name, tensor in state_dict.items():
            try:
                arr = tensor.detach().cpu().to(dtype=torch.float32).numpy()
            except Exception as e:
                error_count += 1
                errors.append(f"{name}: could not read tensor ({e})")
                continue

            chosen = qtype_enum
            if qtype_enum not in (gguf_module.GGMLQuantizationType.F16, gguf_module.GGMLQuantizationType.F32):
                if arr.ndim < 1 or arr.shape[-1] % _BLOCK_SIZE != 0:
                    chosen = gguf_module.GGMLQuantizationType.F16
                    fallback_count += 1

            try:
                if chosen == gguf_module.GGMLQuantizationType.F32:
                    payload = arr.astype(np.float32)
                elif chosen == gguf_module.GGMLQuantizationType.F16:
                    payload = arr.astype(np.float16)
                else:
                    payload = gguf_module.quants.quantize(arr.astype(np.float32), chosen)
                writer.add_tensor(name, payload, raw_dtype=chosen)
                if chosen == qtype_enum and chosen not in (gguf_module.GGMLQuantizationType.F16, gguf_module.GGMLQuantizationType.F32):
                    quantized_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"{name}: {e}")
                continue

        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        except Exception as e:
            return ("", f"[TensorVizion] Failed while writing {out_path}: {e}")

        file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        report = (
            f"Converted: {os.path.basename(src_path)} -> {out_path}\n"
            f"Requested quant type: {quant_type}\n"
            f"Total tensors written: {len(state_dict) - error_count}\n"
            f"Quantized to {quant_type}: {quantized_count}\n"
            f"Fallback to F16 (shape not divisible by block size {_BLOCK_SIZE}): {fallback_count}\n"
            f"Tensors with errors (dropped): {error_count}\n"
            f"Output file size: {file_size / (1024 ** 2):.1f} MB\n"
        )
        if errors:
            report += "First errors:\n" + "\n".join(f"  {e}" for e in errors[:5])

        return (out_path, report)

    def _resolve_out_dir(self, output_filename):
        if os.path.isabs(output_filename):
            return os.path.dirname(output_filename)
        return _output_dir()


NODE_CLASS_MAPPINGS = {
    "GGUFCheckpointConverterNode": GGUFCheckpointConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFCheckpointConverterNode": "GGUF Checkpoint Converter 🔄",
}
