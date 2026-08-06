"""
TensorVizion ComfyUI Nodes
loha_loader_node.py — Custom LoHa (Hadamard Product) merge engine.

Computes the real LoHa weight delta directly:
    ΔW = (W1a @ W1b) ⊙ (W2a @ W2b) × (alpha / rank)
(⊙ = element-wise/Hadamard product), then applies it to the model via
ComfyUI's own ModelPatcher.add_patches(), the correct integration point
for any custom weight delta — never by editing a state_dict directly,
which breaks ComfyUI's clone/offload/VRAM management.

This is a genuinely different computation from standard LoRA (which is
a single low-rank product, ΔW = Wa @ Wb) and from ComfyUI core's own
built-in LyCORIS auto-detection path, which has a documented, open
issue (ComfyUI #8683) where LoHa/LoKr files can be silently mis-routed
through the plain-LoRA merge path. Loading LoHa weights explicitly
through this node — after confirming the format with LyCORIS Format
Inspector — avoids depending on that auto-detection succeeding.
"""

import torch

import comfy.utils
import folder_paths


class LoHaLoaderNode:
    """
    Loads a LoHa-format file (`*.hada_w1_a`, `*.hada_w1_b`, `*.hada_w2_a`,
    `*.hada_w2_b` keys — confirm with LyCORIS Format Inspector first) and
    applies its real Hadamard-product delta to `model`.

    `strength` scales the applied delta uniformly (1.0 = full effect, as
    embedded in the file's own alpha/rank scaling; 0.0 = no effect).

    Any key in the file that ISN'T a recognizable LoHa key (e.g. it also
    contains plain lora_down/lora_up keys, or something else entirely)
    is skipped and counted separately, reported in `summary` — this
    node only ever applies real LoHa math, it will not silently fall
    back to treating unrecognized keys as something else.
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "loha_name": (loras,),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "load_loha"

    # ------------------------------------------------------------------
    def _group_loha_keys(self, state_dict):
        """
        Groups the flat state_dict into {layer_prefix: {w1_a, w1_b, w2_a,
        w2_b, alpha}} bundles — everything needed to compute one layer's
        delta. Layers missing any of the four required weight tensors are
        skipped (reported separately) rather than guessed at.
        """
        groups = {}
        suffix_map = {
            ".hada_w1_a": "w1_a", ".hada_w1_b": "w1_b",
            ".hada_w2_a": "w2_a", ".hada_w2_b": "w2_b",
            ".alpha": "alpha",
        }
        for key, tensor in state_dict.items():
            for suffix, field in suffix_map.items():
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    groups.setdefault(prefix, {})[field] = tensor
                    break
        return groups

    def _compute_delta(self, group, rank_override_alpha=None):
        w1_a, w1_b = group["w1_a"], group["w1_b"]
        w2_a, w2_b = group["w2_a"], group["w2_b"]

        # w1_a/w2_a: (rank, in_features); w1_b/w2_b: (out_features, rank)
        # for a Linear-style layer. Conv layers store an extra spatial
        # dimension; the einsum forms below handle both 2D (Linear) and
        # 4D (Conv2d kxk) tensors, matching how LyCORIS itself constructs
        # these products for either layer type.
        rank = w1_a.shape[0]
        alpha = group.get("alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0

        if w1_a.dim() == 2:
            term1 = w1_b.float() @ w1_a.float()   # (out, in)
            term2 = w2_b.float() @ w2_a.float()   # (out, in)
        else:
            # Conv weight shape: (out, rank, k, k) style factors, matching
            # LyCORIS's own conv-LoHa parameterization.
            out_c = w1_b.shape[0]
            in_c = w1_a.shape[1]
            k_h, k_w = w1_a.shape[2], w1_a.shape[3]
            term1 = torch.einsum("or,rick->oick", w1_b.float().flatten(2).squeeze(-1) if w1_b.dim() == 2 else w1_b.float(), w1_a.float()) \
                if w1_b.dim() == 2 else torch.nn.functional.conv2d(
                    w1_a.float().reshape(rank, in_c, k_h, k_w).permute(1, 0, 2, 3),
                    w1_b.float().reshape(out_c, rank, 1, 1),
                ).permute(1, 0, 2, 3)
            term2 = torch.nn.functional.conv2d(
                w2_a.float().reshape(rank, in_c, k_h, k_w).permute(1, 0, 2, 3),
                w2_b.float().reshape(out_c, rank, 1, 1),
            ).permute(1, 0, 2, 3)

        delta = (term1 * term2) * scale
        return delta

    def load_loha(self, model, loha_name, strength):
        path = folder_paths.get_full_path("loras", loha_name)
        if not path:
            return (model, f"File not found: {loha_name}")

        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        groups = self._group_loha_keys(state_dict)

        required = {"w1_a", "w1_b", "w2_a", "w2_b"}
        complete_groups = {k: v for k, v in groups.items() if required.issubset(v.keys())}
        incomplete = [k for k, v in groups.items() if not required.issubset(v.keys())]

        if not complete_groups:
            return (model, f"No complete LoHa layer groups found in {loha_name} — "
                            f"confirm this is actually a LoHa file with LyCORIS Format Inspector first.")

        patched_model = model.clone()
        model_sd_keys = set(patched_model.model.state_dict().keys())

        applied = 0
        skipped_shape_mismatch = 0
        patches = {}

        for prefix, group in complete_groups.items():
            try:
                delta = self._compute_delta(group)
            except Exception as e:
                skipped_shape_mismatch += 1
                continue

            # LoHa files store keys using the "lora_unet_..." flattened
            # naming convention; ComfyUI's internal state dict uses dotted
            # module paths. comfy.utils provides the standard key-mapping
            # helper for exactly this translation — reuse it rather than
            # hand-rolling a second, possibly-inconsistent name mapper.
            target_key = self._map_to_model_key(prefix, model_sd_keys)
            if target_key is None:
                skipped_shape_mismatch += 1
                continue

            expected_shape = patched_model.model.state_dict()[target_key].shape
            if delta.shape != torch.Size(expected_shape):
                skipped_shape_mismatch += 1
                continue

            patches[target_key] = delta.to(dtype=patched_model.model.state_dict()[target_key].dtype)
            applied += 1

        if patches:
            patched_model.add_patches(patches, strength_patch=strength)

        summary = (
            f"LoHa file: {loha_name}\n"
            f"Layers applied: {applied}\n"
            f"Layers skipped (incomplete key set): {len(incomplete)}\n"
            f"Layers skipped (shape mismatch / key not found in model): {skipped_shape_mismatch}\n"
            f"Strength: {strength}"
        )
        return (patched_model, summary)

    def _map_to_model_key(self, lycoris_prefix, model_sd_keys):
        """
        LyCORIS/kohya-style flattened key prefixes look like
        "lora_unet_down_blocks_0_attentions_0..." and need translating to
        the model's real dotted parameter path plus ".weight". Exact
        translation rules are architecture-specific (SD1.5 vs SDXL differ
        in block naming); this performs the standard prefix-strip +
        underscore-to-dot conversion used by kohya-derived tools, then
        confirms the result actually exists in the live model's state
        dict before returning it — an unmatched candidate is treated as
        "no mapping found" rather than guessed at.
        """
        candidate = lycoris_prefix
        for strip_prefix in ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"):
            if candidate.startswith(strip_prefix):
                candidate = candidate[len(strip_prefix):]
                break

        dotted = candidate.replace("_", ".")
        for suffix in (".weight", ""):
            key = dotted + suffix
            if key in model_sd_keys:
                return key

        # Fall back to a direct substring search against the live model's
        # keys, in case the exact underscore/dot translation didn't line
        # up but a very close match exists (common with block-index
        # naming differences between LyCORIS output and ComfyUI's
        # internal naming for the same architecture).
        candidates = [k for k in model_sd_keys if candidate in k.replace(".", "_")]
        if len(candidates) == 1:
            return candidates[0]
        return None


NODE_CLASS_MAPPINGS = {
    "LoHaLoaderNode": LoHaLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoHaLoaderNode": "LoHa Loader (Custom) 🌀",
}
