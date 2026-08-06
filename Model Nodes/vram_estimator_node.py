"""
TensorVizion ComfyUI Nodes
vram_estimator_node.py — Estimates total VRAM footprint for a planned
checkpoint + LoRA/DoRA/LyCORIS stack BEFORE queuing a run, by reading
file sizes and parameter counts directly (no full model load required
— same lightweight-inspection approach as Model Info Inspector, applied
across a whole stack instead of one file). Catches a likely OOM before
it happens rather than after 40 minutes of setup.

This is an ESTIMATE, not a guarantee: actual VRAM use also depends on
resolution, batch size, attention implementation (xformers/sdpa/etc.),
gradient checkpointing, and ComfyUI's own dynamic offloading behavior —
all of which this node has no visibility into. Treat the output as a
sanity-check floor, not an exact prediction.
"""

import os
import json
import struct

import folder_paths


class VRAMEstimatorNode:
    """
    Reads `checkpoint_name` plus up to 4 optional LoRA/DoRA/LyCORIS
    files (any of Model Nodes' loaders can point at the same files —
    this node only reads file sizes/headers, it doesn't apply anything)
    and estimates:
      - base_weights_gb: raw on-disk size of everything combined,
                          converted to an in-VRAM estimate at the
                          selected `inference_dtype`
      - inference_estimate_gb: base_weights_gb plus a fixed overhead
                                 for activations/VAE/text-encoder,
                                 scaled by `resolution_factor`
      - training_estimate_gb: a full-fine-tune-style estimate
                                 (weights + gradients + optimizer
                                 states), using the same per-parameter
                                 memory math as this pack's earlier
                                 SDXL/SD1.5 fine-tuning guidance —
                                 useful even if you're not training,
                                 as a "how big is this model, really"
                                 sanity check.

    `resolution_factor` is a rough multiplier for activation memory at
    higher resolutions relative to a 512x512 baseline — 1.0 for 512,
    ~2.3 for 768, ~4.0 for 1024, matching how attention/activation
    memory scales roughly with pixel count for a fixed architecture.
    """

    CATEGORY = "TensorVizion/Model"

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "checkpoint_name": (checkpoints,),
                "inference_dtype": (["fp16_bf16", "fp32", "fp8"],),
                "resolution_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 8.0, "step": 0.1}),
            },
            "optional": {
                "lora_1": (loras,),
                "lora_2": (loras,),
                "lora_3": (loras,),
                "lora_4": (loras,),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("base_weights_gb", "inference_estimate_gb", "training_estimate_gb", "report")
    FUNCTION = "estimate"

    BYTES_PER_PARAM = {"fp16_bf16": 2, "fp32": 4, "fp8": 1}

    # Fixed overhead estimates (GB) for pieces beyond the raw UNet/DiT
    # weights at a 512x512 baseline — VAE, text encoder(s) activations,
    # and a modest activation buffer for a batch-1 forward pass. These
    # are round-number estimates from typical SD/SDXL-family footprints,
    # not measured for every possible architecture.
    INFERENCE_OVERHEAD_GB = 2.0
    TRAINING_BYTES_PER_PARAM = {
        # weights (fp16) + gradients (fp16) + adam optimizer states (fp32 m+v) + master weights (fp32)
        "full_finetune_adamw": 2 + 2 + 8 + 4,       # 16 bytes/param
        "full_finetune_8bit_adam": 2 + 2 + 2 + 4,   # 10 bytes/param
    }

    def _file_size_gb(self, path):
        try:
            return os.path.getsize(path) / (1024 ** 3)
        except OSError:
            return 0.0

    def _header_param_count(self, path):
        """Reads only the safetensors header to estimate parameter count
        from tensor shapes, without loading any actual weight data —
        same lightweight approach as the LyCORIS Format Inspector."""
        try:
            with open(path, "rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(header_len).decode("utf-8"))
            header.pop("__metadata__", None)
            total = 0
            for info in header.values():
                shape = info.get("shape", [])
                count = 1
                for s in shape:
                    count *= s
                total += count
            return total
        except Exception:
            return None  # not a safetensors file, or unreadable header — file size fallback used instead

    def estimate(self, checkpoint_name, inference_dtype, resolution_factor,
                 lora_1="None", lora_2="None", lora_3="None", lora_4="None"):
        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        if ckpt_path is None:
            return (0.0, 0.0, 0.0, f"Checkpoint not found: {checkpoint_name}")

        bytes_per_param = self.BYTES_PER_PARAM[inference_dtype]

        lines = [f"Checkpoint: {checkpoint_name}"]
        total_params = 0
        ckpt_params = self._header_param_count(ckpt_path)
        ckpt_size_gb = self._file_size_gb(ckpt_path)

        if ckpt_params:
            total_params += ckpt_params
            lines.append(f"  ~{ckpt_params/1e9:.2f}B params (from header), {ckpt_size_gb:.2f}GB on disk")
        else:
            lines.append(f"  (non-safetensors or unreadable header — using {ckpt_size_gb:.2f}GB on-disk size directly)")

        lora_names = [n for n in (lora_1, lora_2, lora_3, lora_4) if n and n != "None"]
        lora_total_params = 0
        for lora_name in lora_names:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                lines.append(f"  LoRA not found, skipped: {lora_name}")
                continue
            lora_params = self._header_param_count(lora_path)
            if lora_params:
                lora_total_params += lora_params
                lines.append(f"  + {lora_name}: ~{lora_params/1e6:.1f}M params")
            else:
                lines.append(f"  + {lora_name}: could not read header, size not counted")

        total_params += lora_total_params

        if ckpt_params:
            base_weights_gb = (total_params * bytes_per_param) / (1024 ** 3)
        else:
            # No reliable param count for the checkpoint (non-safetensors
            # format) — fall back to on-disk size as the base estimate,
            # since converting an unknown dtype mix to a target dtype
            # byte count isn't meaningful without knowing the source dtype.
            base_weights_gb = ckpt_size_gb + (lora_total_params * bytes_per_param) / (1024 ** 3)

        inference_estimate_gb = base_weights_gb + self.INFERENCE_OVERHEAD_GB * resolution_factor

        training_bytes_per_param_adamw = self.TRAINING_BYTES_PER_PARAM["full_finetune_adamw"]
        training_bytes_per_param_8bit = self.TRAINING_BYTES_PER_PARAM["full_finetune_8bit_adam"]
        training_estimate_gb = (total_params * training_bytes_per_param_adamw) / (1024 ** 3) if total_params else None
        training_estimate_8bit_gb = (total_params * training_bytes_per_param_8bit) / (1024 ** 3) if total_params else None

        report_lines = lines + [
            "",
            f"Total estimated params: ~{total_params/1e9:.2f}B" if total_params else "Total estimated params: unknown (non-safetensors checkpoint)",
            f"Base weights ({inference_dtype}): ~{base_weights_gb:.1f}GB",
            f"Inference estimate (resolution_factor={resolution_factor}): ~{inference_estimate_gb:.1f}GB",
        ]
        if training_estimate_gb is not None:
            report_lines.append(f"Full fine-tune estimate (AdamW, fp16+fp32 master): ~{training_estimate_gb:.1f}GB")
            report_lines.append(f"Full fine-tune estimate (8-bit Adam): ~{training_estimate_8bit_gb:.1f}GB")
        report_lines.append("")
        report_lines.append("This is a floor estimate, not a guarantee — actual VRAM also depends on "
                             "batch size, attention implementation, and ComfyUI's own offloading behavior.")

        report = "\n".join(report_lines)
        return (
            round(base_weights_gb, 2),
            round(inference_estimate_gb, 2),
            round(training_estimate_gb, 2) if training_estimate_gb is not None else 0.0,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "VRAMEstimatorNode": VRAMEstimatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMEstimatorNode": "VRAM / Model Size Estimator 📐",
}
