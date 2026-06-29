"""
TensorVizion ComfyUI Nodes
clip_text_compare_node.py — Encodes two prompts and outputs both conditioning
tensors plus a cosine-similarity score so you can compare prompt directions.
"""

import torch


class CLIPTextCompareNode:
    """
    Encodes two text prompts with the same CLIP model and returns both
    conditionings along with their cosine similarity score (float, 0-1).
    Useful for prompt A/B testing and measuring semantic distance.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":      ("CLIP",),
                "prompt_a":  ("STRING", {"multiline": True, "default": "a futuristic city at night"}),
                "prompt_b":  ("STRING", {"multiline": True, "default": "a neon-lit cyberpunk street"}),
            }
        }

    RETURN_TYPES  = ("CONDITIONING", "CONDITIONING", "FLOAT",  "STRING")
    RETURN_NAMES  = ("conditioning_a", "conditioning_b", "similarity", "similarity_label")
    FUNCTION      = "compare_prompts"

    def compare_prompts(self, clip, prompt_a, prompt_b):
        # Encode prompt A
        tokens_a = clip.tokenize(prompt_a)
        cond_a, pooled_a = clip.encode_from_tokens(tokens_a, return_pooled=True)
        conditioning_a = [[cond_a, {"pooled_output": pooled_a}]]

        # Encode prompt B
        tokens_b = clip.tokenize(prompt_b)
        cond_b, pooled_b = clip.encode_from_tokens(tokens_b, return_pooled=True)
        conditioning_b = [[cond_b, {"pooled_output": pooled_b}]]

        # Cosine similarity on pooled embeddings
        a_vec = pooled_a.float().squeeze()
        b_vec = pooled_b.float().squeeze()

        dot   = torch.dot(a_vec.flatten(), b_vec.flatten())
        norm  = a_vec.norm() * b_vec.norm()
        sim   = (dot / norm.clamp(min=1e-8)).item()
        sim   = float(max(0.0, min(1.0, sim)))

        label = f"Similarity: {sim:.4f}  ({'very similar' if sim > 0.9 else 'similar' if sim > 0.75 else 'moderate' if sim > 0.5 else 'different'})"

        return (conditioning_a, conditioning_b, sim, label)


NODE_CLASS_MAPPINGS = {
    "CLIPTextCompareNode": CLIPTextCompareNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextCompareNode": "CLIP Text Compare 🔍",
}
