"""
TensorVizion ComfyUI Nodes
lokr_loader_node.py — Custom LoKr (Kronecker Product) merge engine.

Computes the real LoKr weight delta directly:
    ΔW = (W1) ⊗ (W2) × (alpha / rank)
(⊗ = Kronecker product), where W1 and/or W2 may themselves be factored
into a low-rank pair (W2 = W2a @ W2b) rather than stored as a single
dense matrix — LoKr files commonly factor the LARGER of the two
Kronecker factors to keep the file small, while leaving the smaller
factor dense. This node handles both the fully-dense and the factored
case for either factor, detected per-layer from whichever keys are
actually present.

Applied via ComfyUI's ModelPatcher.add_patches(), same integration
point as the LoHa Loader and for the same reason: never edit a
state_dict directly, since that breaks ComfyUI's clone/offload/VRAM
management.
"""

import torch

import comfy.utils
import folder_paths


class LoKrLoaderNode:
    """
    Loads a LoKr-format file (`*.lokr_w1` / `*.lokr_w2`, or their
    factored `*_a`/`*_b` variants — confirm with LyCORIS Format
    Inspector first) and applies its real Kronecker-product delta to
    `model`.

    A Kronecker product ⊗ takes two matrices of shape (p,q) and (r,s)
    and produces one of shape (p*r, q*s) by tiling a scaled copy of the
    second matrix into every entry of the first — this is what actually
    lets LoKr represent a full-size weight update from two much smaller
    factor matrices, structurally different from LoRA's simple low-rank
    product or LoHa's element-wise Hadamard product.

    `strength` scales the applied delta uniformly (1.0 = full effect).
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "lokr_name": (loras,),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "load_lokr"

    # ------------------------------------------------------------------
    def _group_lokr_keys(self, state_dict):
        groups = {}
        suffix_map = {
            ".lokr_w1": "w1", ".lokr_w1_a": "w1_a", ".lokr_w1_b": "w1_b",
            ".lokr_w2": "w2", ".lokr_w2_a": "w2_a", ".lokr_w2_b": "w2_b",
            ".alpha": "alpha",
        }
        for key, tensor in state_dict.items():
            for suffix, field in suffix_map.items():
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    groups.setdefault(prefix, {})[field] = tensor
                    break
        return groups

    def _resolve_factor(self, group, factor_num):
        """
        Returns the dense (p,q) matrix for factor 1 or 2, whether it was
        stored as a single dense tensor (f"w{factor_num}") or as a
        factored low-rank pair (f"w{factor_num}_a" @ f"w{factor_num}_b").
        Returns None if neither form is present for this factor.
        """
        dense_key = f"w{factor_num}"
        a_key, b_key = f"w{factor_num}_a", f"w{factor_num}_b"

        if dense_key in group:
            return group[dense_key].float()
        if a_key in group and b_key in group:
            # Factored form: full matrix = B @ A (A: rank x q, B: p x rank),
            # matching the same down/up convention LoRA itself uses.
            return (group[b_key].float() @ group[a_key].float())
        return None

    def _compute_delta(self, group):
        w1 = self._resolve_factor(group, 1)
        w2 = self._resolve_factor(group, 2)
        if w1 is None or w2 is None:
            raise ValueError("incomplete LoKr factor group")

        if w1.dim() != 2 or w2.dim() != 2:
            raise ValueError("LoKr conv-layer (4D) support not implemented — only Linear-style 2D factors are handled")

        alpha = group.get("alpha")
        # LoKr's rank concept differs from LoRA's: when either factor is
        # itself factored (low-rank), that inner rank is what alpha
        # scales against; a fully-dense LoKr (no _a/_b keys at all) has
        # no meaningful "rank" and effectively uses scale=1.0 unless
        # alpha is explicitly given relative to one of the dense factor
        # dimensions — this matches how kohya/LyCORIS handle the same
        # ambiguity, defaulting to no extra scaling in the fully-dense case.
        if alpha is not None and ("w1_a" in group or "w2_a" in group):
            rank_source = group.get("w1_a", group.get("w2_a"))
            rank = rank_source.shape[0]
            scale = float(alpha) / rank
        else:
            scale = 1.0

        delta = torch.kron(w1, w2) * scale
        return delta

    def load_lokr(self, model, lokr_name, strength):
        path = folder_paths.get_full_path("loras", lokr_name)
        if not path:
            return (model, f"File not found: {lokr_name}")

        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        groups = self._group_lokr_keys(state_dict)

        complete_groups = {}
        incomplete = []
        for prefix, group in groups.items():
            has_w1 = "w1" in group or ("w1_a" in group and "w1_b" in group)
            has_w2 = "w2" in group or ("w2_a" in group and "w2_b" in group)
            if has_w1 and has_w2:
                complete_groups[prefix] = group
            else:
                incomplete.append(prefix)

        if not complete_groups:
            return (model, f"No complete LoKr layer groups found in {lokr_name} — "
                            f"confirm this is actually a LoKr file with LyCORIS Format Inspector first.")

        patched_model = model.clone()
        model_sd_keys = set(patched_model.model.state_dict().keys())

        applied = 0
        skipped_unsupported = 0
        skipped_shape_mismatch = 0
        patches = {}

        for prefix, group in complete_groups.items():
            try:
                delta = self._compute_delta(group)
            except ValueError:
                skipped_unsupported += 1
                continue
            except Exception:
                skipped_shape_mismatch += 1
                continue

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
            f"LoKr file: {lokr_name}\n"
            f"Layers applied: {applied}\n"
            f"Layers skipped (incomplete key set): {len(incomplete)}\n"
            f"Layers skipped (conv/4D factors — not yet supported): {skipped_unsupported}\n"
            f"Layers skipped (shape mismatch / key not found in model): {skipped_shape_mismatch}\n"
            f"Strength: {strength}"
        )
        return (patched_model, summary)

    def _map_to_model_key(self, lycoris_prefix, model_sd_keys):
        """Same translation strategy as LoHa Loader — see its docstring
        for the full rationale. Kept as an independent copy per this
        pack's per-file convention rather than a shared import, so each
        node file stays self-contained."""
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

        candidates = [k for k in model_sd_keys if candidate in k.replace(".", "_")]
        if len(candidates) == 1:
            return candidates[0]
        return None


NODE_CLASS_MAPPINGS = {
    "LoKrLoaderNode": LoKrLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoKrLoaderNode": "LoKr Loader (Custom) 🧩",
}
