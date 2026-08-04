"""
TensorVizion ComfyUI Nodes
trigger_word_extractor_node.py — Parses a LoRA filename (CivitAI-style
naming, underscores, version tags) and extracts a best-guess trigger word,
stripping common noise tokens (v1/v2, sdxl, fp16, pruned, epoch markers,
etc). Pairs with LoRA Info Inspector and LoRA Stack when wiring up a batch
of LoRAs by filename alone.
"""

import os
import re


class TriggerWordExtractorNode:
    """
    Splits `filename_or_path` on common separators, drops known noise
    tokens and pure numeric/epoch markers, and returns both a trigger-word
    guess (underscore-joined) and a human-readable cleaned token string.
    """

    CATEGORY = "TensorVizion/Model"

    NOISE_TOKENS = {
        "v1", "v2", "v3", "v4", "v5", "sdxl", "sd15", "sd", "1.5", "xl",
        "fp16", "fp32", "pruned", "lora", "lycoris", "locon", "loha",
        "final", "last", "epoch", "e1", "e2", "e3", "e4", "e5", "step",
        "steps", "ckpt", "safetensors", "model",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_or_path": ("STRING", {"default": "my_character_v2_sdxl_fp16.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("trigger_word", "cleaned_tokens")
    FUNCTION      = "run"

    def run(self, filename_or_path):
        name = os.path.basename(filename_or_path)
        name = os.path.splitext(name)[0]

        parts = [p for p in re.split(r"[_\-\.\s]+", name) if p]

        kept = []
        for p in parts:
            lp = p.lower()
            if lp in self.NOISE_TOKENS:
                continue
            if re.fullmatch(r"e?\d+", lp):
                continue
            kept.append(p)

        cleaned = " ".join(kept) if kept else name
        trigger = "_".join(kept) if kept else name

        return (trigger, cleaned)


NODE_CLASS_MAPPINGS = {
    "TriggerWordExtractorNode": TriggerWordExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TriggerWordExtractorNode": "Trigger Word Extractor 🏹",
}
