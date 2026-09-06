"""
TensorVizion ComfyUI Nodes
gguf_file_info_node.py — Inspects a .gguf file (quantized diffusion
model, text encoder, or VAE exported in GGUF format) and reports its
architecture metadata plus a per-tensor quantization breakdown, without
loading it into a pipeline. Run this before any of the other five GGUF
Nodes to sanity-check a file — especially useful for confirming which
quant type a download actually is (filenames are not reliable) and
spotting mixed-precision files before they hit a loader.

Depends on the `gguf` PyPI package (the same Python bindings used by
llama.cpp and ComfyUI-GGUF): `pip install gguf`. This is NOT bundled
with ComfyUI — see requirements.txt.
"""

import json
import os

try:
    import folder_paths
except ImportError:
    folder_paths = None


# Folders that commonly hold .gguf checkpoint files across different
# GGUF-supporting extensions/conventions — checked in this order so this
# node works regardless of which one a user's setup uses.
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


class GGUFFileInfoNode:
    """
    Reads `gguf_name` (or `path_override`, for files outside ComfyUI's
    registered model folders) using `gguf.GGUFReader` and reports:
    architecture/name/quant-version key-value metadata, a count of
    tensors per GGML quantization type (the actual per-tensor mix — many
    "Q4" files are really Q4_K for most weights with a handful of F32/F16
    layers kept full precision), total tensor count, and total on-disk
    tensor byte size. This is read-only inspection — it never dequantizes
    or loads tensor data, so it works even on files no loader here
    supports yet.
    """

    CATEGORY = "TensorVizion/GGUF"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (_discover_gguf_filenames(),),
            },
            "optional": {
                "path_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "metadata_json")
    FUNCTION = "run"

    def run(self, gguf_name, path_override=""):
        try:
            import gguf
        except ImportError:
            msg = "[TensorVizion] The `gguf` package is not installed. Run: pip install gguf"
            return (msg, "{}")

        path = _resolve_path(gguf_name, path_override)
        if not path:
            msg = f"[TensorVizion] File not found: {gguf_name!r} (checked model folders and path_override)"
            return (msg, "{}")

        try:
            reader = gguf.GGUFReader(path)
        except Exception as e:
            msg = f"[TensorVizion] Failed to parse {os.path.basename(path)} as GGUF: {e}"
            return (msg, "{}")

        # Key/value metadata (architecture, name, quant version, etc.)
        # ReaderField.contents() is the library's own accessor for the
        # decoded scalar/array/string value of a field — safer than
        # poking at .parts/.data directly, which are documented as
        # internal/subject to change.
        kv = {}
        for key, field in reader.fields.items():
            if key.startswith("GGUF."):
                continue
            try:
                kv[key] = field.contents()
            except Exception:
                continue

        # Per-tensor quantization type breakdown + size totals
        type_counts = {}
        total_bytes = 0
        total_elements = 0
        for t in reader.tensors:
            type_name = t.tensor_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            total_bytes += int(t.n_bytes)
            total_elements += int(t.n_elements)

        file_size = os.path.getsize(path)
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        breakdown_lines = "\n".join(
            f"  {name}: {count} tensor(s)" for name, count in sorted(type_counts.items(), key=lambda kv_: -kv_[1])
        )
        summary = (
            f"GGUF file: {os.path.basename(path)}\n"
            f"File size: {file_size / (1024 ** 2):.1f} MB\n"
            f"Architecture: {kv.get('general.architecture', 'unknown')}\n"
            f"Name: {kv.get('general.name', 'unknown')}\n"
            f"Quantization version: {kv.get('general.quantization_version', 'n/a')}\n"
            f"Total tensors: {len(reader.tensors)}\n"
            f"Total elements: {total_elements:,}\n"
            f"Tensor data on disk: {total_bytes / (1024 ** 2):.1f} MB\n"
            f"Dominant quant type: {dominant_type}\n"
            f"Quant type breakdown:\n{breakdown_lines}\n"
            f"Note: dominant type is a per-file majority vote, not a guarantee "
            f"every tensor shares it — mixed-precision files are common and the "
            f"breakdown above shows the real mix."
        )

        json_safe_kv = {}
        for k, v in kv.items():
            try:
                json.dumps(v)
                json_safe_kv[k] = v
            except TypeError:
                json_safe_kv[k] = str(v)

        metadata_json = json.dumps(
            {
                "file": os.path.basename(path),
                "file_size_bytes": file_size,
                "tensor_count": len(reader.tensors),
                "total_elements": total_elements,
                "tensor_bytes": total_bytes,
                "quant_type_counts": type_counts,
                "kv_metadata": json_safe_kv,
            },
            indent=2,
            default=str,
        )

        return (summary, metadata_json)


NODE_CLASS_MAPPINGS = {
    "GGUFFileInfoNode": GGUFFileInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFFileInfoNode": "GGUF File Info 🔍",
}
