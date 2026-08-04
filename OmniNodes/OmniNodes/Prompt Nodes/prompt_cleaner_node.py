"""
TensorVizion ComfyUI Nodes
prompt_cleaner_node.py — Normalises a prompt string: strips duplicate
comma-separated tags, collapses extra whitespace/commas, and optionally
lowercases or truncates. Useful after combining/wildcard-resolving several
fragments where duplicate tags tend to creep in.
"""

import re


class PromptCleanerNode:
    """
    Cleans up a prompt after concatenation, wildcard resolution, or manual
    editing has left it messy:

      - Splits on commas, strips whitespace from each tag
      - Removes exact duplicate tags (case-insensitive when
        `case_insensitive_dedup` is on), keeping the first occurrence
      - Collapses runs of whitespace within each tag to a single space
      - Drops empty tags left behind by trailing/double commas
      - Optionally lowercases everything
      - Optionally truncates to `max_tags` tags (keeps the first N)

    Returns the cleaned string plus a count of how many duplicate/empty
    tags were removed, so you can see whether cleaning actually did
    anything on a given run.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "case_insensitive_dedup": ("BOOLEAN", {"default": True}),
                "lowercase": ("BOOLEAN", {"default": False}),
                "max_tags": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
            }
        }

    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("cleaned_text", "removed_count")
    FUNCTION      = "clean"

    def clean(self, text, case_insensitive_dedup, lowercase, max_tags):
        raw_tags = text.split(",")
        cleaned_tags = []
        seen = set()
        removed = 0

        for raw in raw_tags:
            tag = re.sub(r"\s+", " ", raw.strip())
            if not tag:
                removed += 1
                continue

            key = tag.lower() if case_insensitive_dedup else tag
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            cleaned_tags.append(tag)

        if max_tags > 0 and len(cleaned_tags) > max_tags:
            removed += len(cleaned_tags) - max_tags
            cleaned_tags = cleaned_tags[:max_tags]

        result = ", ".join(cleaned_tags)
        if lowercase:
            result = result.lower()

        return (result, removed)


NODE_CLASS_MAPPINGS = {
    "PromptCleanerNode": PromptCleanerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCleanerNode": "Prompt Cleaner 🧹",
}
