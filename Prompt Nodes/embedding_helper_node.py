"""
TensorVizion ComfyUI Nodes
embedding_helper_node.py — Browses installed textual-inversion embeddings
and injects the correct `embedding:name` syntax into a prompt string,
so users don't have to remember exact embedding filenames or the syntax
ComfyUI's CLIP tokenizer expects. Not a loader in the MODEL/CLIP sense —
embeddings are resolved by CLIP Text Encode at tokenize time — this node
is a convenience/validity-check layer that sits before it.
"""

import folder_paths


class EmbeddingHelperNode:
    """
    Picks `embedding_name` from whatever's actually installed in any
    `embeddings/` folder ComfyUI knows about, and inserts
    `(embedding:name:weight)` into `text` at the position `insert_at`
    marks in your text (or appended at the end if not found). This
    guarantees the embedding name is spelled correctly and actually
    exists, which is otherwise a common source of silent "embedding not
    found, ignoring" warnings when typed by hand.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        embeddings = folder_paths.get_filename_list("embeddings")
        if not embeddings:
            embeddings = ["(no embeddings found)"]
        return {
            "required": {
                "text":           ("STRING", {"multiline": True, "default": ""}),
                "embedding_name": (embeddings,),
                "weight":         ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "insert_at":      ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("text",   "summary")
    FUNCTION      = "run"

    def run(self, text, embedding_name, weight, insert_at):
        if embedding_name == "(no embeddings found)":
            return (text, "[TensorVizion] No embeddings found in any embeddings/ folder — text unchanged.")

        # Strip any known extension for the tag itself (ComfyUI resolves by base name)
        base_name = embedding_name
        for ext in (".pt", ".safetensors", ".bin"):
            if base_name.endswith(ext):
                base_name = base_name[: -len(ext)]
                break

        if abs(weight - 1.0) < 1e-6:
            tag = f"embedding:{base_name}"
        else:
            tag = f"(embedding:{base_name}:{weight})"

        if insert_at and insert_at in text:
            new_text = text.replace(insert_at, f"{insert_at}, {tag}", 1)
        else:
            new_text = f"{text}, {tag}" if text.strip() else tag

        summary = (
            f"Embedding:  {embedding_name}\n"
            f"Weight:     {weight}\n"
            f"Inserted:   {'at marker \"' + insert_at + '\"' if insert_at and insert_at in text else 'appended at end'}"
        )

        return (new_text, summary)


NODE_CLASS_MAPPINGS = {
    "EmbeddingHelperNode": EmbeddingHelperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmbeddingHelperNode": "Embedding Helper 🧷",
}
