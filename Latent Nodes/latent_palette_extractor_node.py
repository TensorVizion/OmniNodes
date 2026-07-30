"""
TensorVizion ComfyUI Nodes
latent_palette_extractor_node.py — Extracts a compact per-channel
statistical "signature" from a latent, pre-decode, so batches or seeds
can be compared for compositional similarity without ever running them
through the VAE. Not a literal RGB color palette (latents aren't pixel
space) — a numeric fingerprint that behaves like one for comparison
purposes: similar fingerprints tend to decode to similar-looking images.
"""

import numpy as np
import torch


class LatentPaletteExtractorNode:
    """
    Computes, per channel, the mean / std / dominant-sign-ratio across the
    spatial dimensions of a latent, then reduces the whole batch to a
    single signature vector. Two latents with similar signatures are
    likely to decode to compositionally similar images (similar overall
    brightness/contrast/structure balance); very different signatures
    usually decode to very different-looking results. Use this to filter
    a batch of seeds down to "look different enough to be worth decoding"
    before spending VAE time on all of them, or to match a new seed to an
    existing look from a past signature you saved.
    """

    CATEGORY = "TensorVizion/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "compare_signature": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES  = ("STRING",     "FLOAT",              "STRING")
    RETURN_NAMES  = ("signature",  "similarity_to_input", "report")
    FUNCTION      = "run"

    @staticmethod
    def _compute_signature(samples):
        """samples: (B, C, H, W) tensor -> flat numpy signature vector."""
        combined = samples.mean(dim=0)  # (C, H, W), average over batch
        C = combined.shape[0]

        sig = []
        for c in range(C):
            ch = combined[c].detach().cpu().numpy()
            sig.append(float(ch.mean()))
            sig.append(float(ch.std()))
            sig.append(float(np.mean(ch > 0)))  # positive-value ratio

        return np.array(sig, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a, b):
        if a.shape != b.shape:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def run(self, latent, compare_signature=""):
        samples = latent["samples"]
        sig = self._compute_signature(samples)
        sig_str = ",".join(f"{v:.5f}" for v in sig)

        similarity = 0.0
        compare_note = "No comparison signature provided."

        if compare_signature.strip():
            try:
                other = np.array(
                    [float(x) for x in compare_signature.strip().split(",") if x.strip()],
                    dtype=np.float32,
                )
                similarity = self._cosine_similarity(sig, other)
                if similarity > 0.98:
                    compare_note = "Near-identical composition to the comparison signature."
                elif similarity > 0.9:
                    compare_note = "Similar overall composition/balance."
                elif similarity > 0.7:
                    compare_note = "Some structural similarity, noticeably different in places."
                else:
                    compare_note = "Substantially different composition."
            except ValueError:
                compare_note = "Could not parse compare_signature — expected comma-separated floats."

        report = (
            f"Channels analyzed: {samples.shape[1]}\n"
            f"Signature length:  {len(sig)} values (mean/std/+ratio per channel)\n"
            f"Similarity:        {similarity:.4f}\n"
            f"{compare_note}"
        )

        return (sig_str, similarity, report)


NODE_CLASS_MAPPINGS = {
    "LatentPaletteExtractorNode": LatentPaletteExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPaletteExtractorNode": "Latent Palette Extractor 🧬",
}
