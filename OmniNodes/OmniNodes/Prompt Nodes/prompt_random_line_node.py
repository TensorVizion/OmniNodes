"""
TensorVizion ComfyUI Nodes
prompt_random_line_node.py — Picks one random line from a pasted multiline
STRING, seeded for reproducibility. The inline, file-free companion to
Wildcard Loader for a quick list of alternatives typed directly into a node.
"""

import random


class PromptRandomLineNode:
    """
    Splits `text` on newlines and returns one line chosen by `seed`. Blank
    lines and lines starting with `#` are ignored, so you can keep a list
    tidy with comments/spacing without them ever being selected.

    Unlike Wildcard Loader (which reads from external .txt files and
    resolves __tokens__ inside a larger prompt), this node works entirely
    on a pasted list local to the node — useful for a short one-off set of
    alternatives you don't want to manage as a separate file.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "cinematic lighting\nsoft studio lighting\ngolden hour\nneon backlight",
                    "multiline": True,
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES  = ("STRING", "INT", "INT")
    RETURN_NAMES  = ("chosen_line", "line_index", "total_lines")
    FUNCTION      = "pick"

    def pick(self, text, seed):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

        if not lines:
            return ("", -1, 0)

        rng = random.Random(seed)
        index = rng.randrange(len(lines))
        return (lines[index], index, len(lines))


NODE_CLASS_MAPPINGS = {
    "PromptRandomLineNode": PromptRandomLineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRandomLineNode": "Prompt Random Line 🎯",
}
