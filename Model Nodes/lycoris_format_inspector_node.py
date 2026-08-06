"""
TensorVizion ComfyUI Nodes
lycoris_format_inspector_node.py — Reads a LoRA-family file's actual
tensor key names (not just its file extension or folder) and identifies
which real format it is: standard LoRA, LoHa, LoKr, or LoCon, and
whether it carries a DoRA magnitude-decomposition marker on top of any
of those. Every LoHa/LoKr/DoRA merge node below depends on this
detection being right, since misidentifying the format means applying
the wrong math entirely — a real, documented failure mode (ComfyUI
issue #8683: LyCORIS format auto-detection can silently mis-route a
LoHa/LoKr file through the plain-LoRA path, producing a technically-
running but wrong result rather than an error).

Detection is based on the real, distinct key-naming conventions each
format actually uses on disk:
  LoRA / LoCon : *.lora_down.weight, *.lora_up.weight, *.alpha
  LoHa         : *.hada_w1_a, *.hada_w1_b, *.hada_w2_a, *.hada_w2_b
  LoKr         : *.lokr_w1 (or *.lokr_w1_a + *.lokr_w1_b if factored),
                 *.lokr_w2 (or *.lokr_w2_a + *.lokr_w2_b if factored)
  DoRA marker  : *.dora_scale present alongside any of the above
"""

import struct
import json
import os

import folder_paths


class LycorisFormatInspectorNode:
    """
    Reports, per unique layer prefix found in the file:
      - detected_format: "lora", "loha", "lokr", "unknown", or "mixed"
        (mixed = the file contains more than one format's key pattern,
        which is unusual but not impossible for hand-merged files)
      - has_dora: whether any *.dora_scale key was found
      - layer_count: how many distinct layers/modules have adapter weights
      - rank: for LoRA/LoCon, the down-projection's rank (its output dim);
               for LoHa/LoKr, the rank of the factored low-rank pieces if
               present, or None if the format doesn't expose a single rank
               number the way LoRA does

    Reads ONLY the safetensors header (tensor names + shapes + dtype),
    never the tensor data itself, so this is fast and low-memory even on
    large files — the header stores this as a plain JSON block at the
    start of the file, no need to load actual weights to answer "what
    format is this."
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (loras,),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = ("detected_format", "has_dora", "layer_count", "report")
    FUNCTION = "inspect"

    # ------------------------------------------------------------------
    def _read_header(self, path):
        """
        Reads the safetensors header directly: an 8-byte little-endian
        length prefix, then that many bytes of JSON describing every
        tensor's name/dtype/shape plus an optional '__metadata__' block.
        Matches the byte-parsing convention already used by
        model_lora_metadata_diff_node.py elsewhere in this pack, but
        keeps the full per-tensor header (not just '__metadata__') since
        format detection needs every tensor's NAME, not just the
        embedded metadata block.
        """
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode("utf-8"))
        header.pop("__metadata__", None)
        return header  # {tensor_name: {"dtype":..., "shape":[...], ...}}

    def _classify_key(self, key):
        if key.endswith(".dora_scale") or ".dora_scale" in key:
            return "dora_marker"
        if any(s in key for s in (".lora_down.weight", ".lora_up.weight", ".lora.down", ".lora.up")) or key.endswith(".alpha"):
            return "lora"
        if any(s in key for s in (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2")):
            return "loha"
        if any(s in key for s in (".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2")):
            return "lokr"
        return None

    def _layer_prefix(self, key):
        # Strip the format-specific suffix to get the module/layer this
        # key belongs to, e.g. "lora_unet_...blocks_0_attn.lora_down.weight"
        # -> "lora_unet_...blocks_0_attn"
        for suffix in (
            ".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale",
            ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
            ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
        ):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    def inspect(self, lora_name):
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not os.path.isfile(path):
            return ("unknown", False, 0, f"File not found: {lora_name}")

        try:
            header = self._read_header(path)
        except Exception as e:
            return ("unknown", False, 0, f"Could not read header: {e}")

        layer_formats = {}   # layer_prefix -> set of formats found for it
        has_dora = False
        rank_samples = []

        for key, info in header.items():
            kind = self._classify_key(key)
            if kind is None:
                continue
            if kind == "dora_marker":
                has_dora = True
                continue

            prefix = self._layer_prefix(key)
            layer_formats.setdefault(prefix, set()).add(kind)

            if kind == "lora" and key.endswith(".lora_down.weight"):
                shape = info.get("shape", [])
                if shape:
                    rank_samples.append(shape[0])

        if not layer_formats:
            return ("unknown", has_dora, 0, "No recognizable LoRA/LoHa/LoKr keys found in this file.")

        all_formats_seen = set()
        for formats in layer_formats.values():
            all_formats_seen |= formats

        if len(all_formats_seen) > 1:
            detected_format = "mixed"
        else:
            detected_format = next(iter(all_formats_seen))

        layer_count = len(layer_formats)
        rank = None
        if rank_samples:
            rank = rank_samples[0]
            rank_consistent = all(r == rank for r in rank_samples)
        else:
            rank_consistent = True

        format_note = ""
        if detected_format == "mixed":
            format_counts = {}
            for formats in layer_formats.values():
                for f in formats:
                    format_counts[f] = format_counts.get(f, 0) + 1
            format_note = (
                f"\nWARNING: multiple adapter formats detected in one file "
                f"({dict(format_counts)}). This is unusual — double-check "
                f"this file wasn't produced by concatenating unrelated LoRAs, "
                f"which some tools will do without warning."
            )
        elif rank_samples and not rank_consistent:
            format_note = "\nNote: rank varies across layers (not all lora_down.weight shapes match)."

        report = (
            f"File: {lora_name}\n"
            f"Detected format: {detected_format}\n"
            f"DoRA magnitude marker present: {has_dora}\n"
            f"Layers with adapter weights: {layer_count}\n"
            f"Rank: {rank if rank is not None else 'N/A (not exposed as a single number for this format)'}"
            f"{format_note}"
        )

        return (detected_format, has_dora, layer_count, report)


NODE_CLASS_MAPPINGS = {
    "LycorisFormatInspectorNode": LycorisFormatInspectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LycorisFormatInspectorNode": "LyCORIS Format Inspector 🔬",
}
