"""
TensorVizion ComfyUI Nodes
dora_loader_node.py — Custom DoRA (Weight-Decomposed Low-Rank Adaptation)
merge engine.

Computes the real DoRA update directly, per the DoRA paper's
decomposition:
    W' = m * (W0 + BA) / ||W0 + BA||_c
where W0 is the base model's original weight, BA is the standard LoRA
low-rank delta (B=lora_up, A=lora_down), ||.||_c is the column-wise
L2 norm, and m is DoRA's learned per-output-channel magnitude vector
(stored in the file as *.dora_scale).

This is NOT the same as just applying a LoRA delta with extra scaling —
DoRA's whole point is that it decomposes the update into a MAGNITUDE
component (m) and a DIRECTION component (the normalized W0+BA), and the
learned magnitude vector only makes sense relative to that
normalization. A well-known community issue is treating dora_scale as
if it were an ordinary LoRA strength multiplier, which does not
reproduce DoRA's actual training-time behavior; this node instead
computes the full W0+BA -> normalize -> rescale-by-m -> delta-from-W0
pipeline before handing the result to ComfyUI's add_patches API.
"""

import torch

import comfy.utils
import folder_paths


class DoRALoaderNode:
    """
    Loads a DoRA file (standard LoRA keys — `*.lora_down.weight` /
    `*.lora_up.weight` / `*.alpha` — PLUS a `*.dora_scale` magnitude
    vector per layer; confirm with LyCORIS Format Inspector first,
    checking `has_dora=True`) and applies the real magnitude/direction
    decomposition to `model`.

    Because this requires the ORIGINAL base weight (W0) to compute the
    normalized direction, this node reads the live model's current
    weight for each affected layer before computing the update — it
    will not produce a correct result if `model` already has other
    patches applied that meaningfully changed those same weights, since
    DoRA's normalization is defined relative to a specific base weight.
    For predictable results, apply DoRA loaders before other model
    patches in your workflow graph, not after.

    `strength` scales the FINAL computed delta uniformly, same
    convention as every other loader in this pack — 1.0 reproduces the
    file's trained effect as closely as this node's math allows; values
    away from 1.0 blend toward (0.0) or exaggerate (>1.0) that effect.
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "dora_name": (loras,),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "load_dora"

    # ------------------------------------------------------------------
    def _group_dora_keys(self, state_dict):
        groups = {}
        suffix_map = {
            ".lora_down.weight": "down", ".lora_up.weight": "up",
            ".alpha": "alpha", ".dora_scale": "dora_scale",
        }
        for key, tensor in state_dict.items():
            for suffix, field in suffix_map.items():
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    groups.setdefault(prefix, {})[field] = tensor
                    break
        return groups

    def _compute_delta(self, group, base_weight):
        down = group["down"].float()   # (rank, in)
        up = group["up"].float()       # (out, rank)
        dora_scale = group["dora_scale"].float()  # (out, 1) typically

        rank = down.shape[0]
        alpha = group.get("alpha")
        lora_scale = (float(alpha) / rank) if alpha is not None else 1.0

        if base_weight.dim() != 2:
            raise ValueError("DoRA conv-layer (4D) support not implemented — only Linear-style 2D weights are handled")

        ba = (up @ down) * lora_scale                    # (out, in), the standard LoRA delta
        combined = base_weight.float() + ba               # W0 + BA

        # Column-wise L2 norm: DoRA normalizes each OUTPUT row (each
        # output neuron's full weight vector across all inputs), i.e.
        # norm over dim=1 for a (out, in) weight, keeping one norm value
        # per output channel — matching the DoRA paper's ||.||_c definition.
        col_norm = combined.norm(dim=1, keepdim=True)          # (out, 1)
        direction = combined / (col_norm + 1e-8)

        dora_scale_reshaped = dora_scale.reshape(-1, 1)
        new_weight = dora_scale_reshaped * direction           # (out, in)

        delta = new_weight - base_weight.float()
        return delta

    def load_dora(self, model, dora_name, strength):
        path = folder_paths.get_full_path("loras", dora_name)
        if not path:
            return (model, f"File not found: {dora_name}")

        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        groups = self._group_dora_keys(state_dict)

        required = {"down", "up", "dora_scale"}
        complete_groups = {k: v for k, v in groups.items() if required.issubset(v.keys())}
        incomplete = [k for k, v in groups.items() if not required.issubset(v.keys())]
        no_dora_marker = sum(1 for v in groups.values() if "down" in v and "up" in v and "dora_scale" not in v)

        if not complete_groups:
            if no_dora_marker > 0:
                return (model, f"{dora_name} has standard LoRA keys but NO dora_scale marker on any layer — "
                                f"this is a plain LoRA, not a DoRA file. Use your existing LoRA loader instead. "
                                f"Confirm with LyCORIS Format Inspector (check has_dora).")
            return (model, f"No complete DoRA layer groups found in {dora_name}.")

        patched_model = model.clone()
        live_state_dict = patched_model.model.state_dict()
        model_sd_keys = set(live_state_dict.keys())

        applied = 0
        skipped_unsupported = 0
        skipped_shape_mismatch = 0
        patches = {}

        for prefix, group in complete_groups.items():
            target_key = self._map_to_model_key(prefix, model_sd_keys)
            if target_key is None:
                skipped_shape_mismatch += 1
                continue

            base_weight = live_state_dict[target_key]

            try:
                delta = self._compute_delta(group, base_weight)
            except ValueError:
                skipped_unsupported += 1
                continue
            except Exception:
                skipped_shape_mismatch += 1
                continue

            if delta.shape != base_weight.shape:
                skipped_shape_mismatch += 1
                continue

            patches[target_key] = delta.to(dtype=base_weight.dtype)
            applied += 1

        if patches:
            patched_model.add_patches(patches, strength_patch=strength)

        summary = (
            f"DoRA file: {dora_name}\n"
            f"Layers applied: {applied}\n"
            f"Layers skipped (missing dora_scale — not a DoRA layer): {len(incomplete)}\n"
            f"Layers skipped (conv/4D weights — not yet supported): {skipped_unsupported}\n"
            f"Layers skipped (shape mismatch / key not found in model): {skipped_shape_mismatch}\n"
            f"Strength: {strength}\n"
            f"Note: computed via real magnitude/direction decomposition (||W0+BA||_c), "
            f"not a simple scaled LoRA delta."
        )
        return (patched_model, summary)

    def _map_to_model_key(self, lycoris_prefix, model_sd_keys):
        """Same translation strategy as LoHa/LoKr Loaders — see LoHa
        Loader's docstring for the full rationale."""
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
    "DoRALoaderNode": DoRALoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoRALoaderNode": "DoRA Loader (Custom) 🎯",
}
