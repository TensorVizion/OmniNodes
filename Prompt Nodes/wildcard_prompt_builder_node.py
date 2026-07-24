"""
TensorVizion ComfyUI Nodes
wildcard_prompt_builder_node.py — Expands inline {option_a|option_b|option_c}
syntax within a prompt template, including nested groups. The compact,
file-free companion to Wildcard Loader (which resolves __tokens__ against
external .txt files); this one keeps the whole option set in the node.
"""

import re
import random


class WildcardPromptBuilderNode:
    """
    Resolves every {a|b|c} group in `template` by picking one option per
    group, seeded by `seed` for reproducibility. Groups may nest, e.g.
    {a|{b|c}}, and are resolved innermost-first until none remain.
    """

    CATEGORY = "TensorVizion/Prompt"

    _GROUP_RE = re.compile(r"\{([^{}]*)\}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "multiline": True,
                    "default": "a {red|blue|green} {car|bicycle}, {sunny|rainy} day",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION      = "run"

    def run(self, template, seed):
        rng = random.Random(seed)
        text = template

        guard = 0
        while "{" in text and guard < 100:
            def resolve(match):
                options = match.group(1).split("|")
                return rng.choice(options) if options else ""

            new_text = self._GROUP_RE.sub(resolve, text)
            if new_text == text:
                break
            text = new_text
            guard += 1

        return (re.sub(r"\s+", " ", text).strip(),)


NODE_CLASS_MAPPINGS = {
    "WildcardPromptBuilderNode": WildcardPromptBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardPromptBuilderNode": "Wildcard Prompt Builder 🧩",
}
