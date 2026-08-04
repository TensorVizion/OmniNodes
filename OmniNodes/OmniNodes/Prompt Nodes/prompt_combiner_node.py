"""
TensorVizion ComfyUI Nodes
prompt_combiner_node.py — Joins up to 4 prompt fragments into one string,
with per-fragment weight syntax and a configurable separator. The text
equivalent of Audio Mixer/Image Blend for combining multiple sources into one.
"""


class PromptCombinerNode:
    """
    Combines up to 4 optional text fragments into a single prompt string.
    Each connected fragment gets its own `weight_N`; a weight of 1.0 is
    passed through unchanged, anything else is wrapped in `(fragment:weight)`
    syntax before joining (the standard emphasis format most SD/SDXL
    frontends and the CLIP Text Weight node both understand).

    Empty/disconnected slots are skipped entirely — no dangling separators.
    `separator` defaults to a comma-space (`, `) but accepts any string,
    including a newline (`\\n`) for line-per-fragment prompts.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": ", "}),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "text_1": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "text_2": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "text_3": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "text_4": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("combined_text",)
    FUNCTION      = "combine"

    def combine(self, separator, weight_1, weight_2, weight_3, weight_4,
                text_1="", text_2="", text_3="", text_4=""):
        slots = [(text_1, weight_1), (text_2, weight_2), (text_3, weight_3), (text_4, weight_4)]
        parts = []
        for text, weight in slots:
            text = (text or "").strip()
            if not text:
                continue
            if abs(weight - 1.0) < 1e-6:
                parts.append(text)
            else:
                parts.append(f"({text}:{weight:.2f})")

        return (separator.join(parts),)


NODE_CLASS_MAPPINGS = {
    "PromptCombinerNode": PromptCombinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCombinerNode": "Prompt Combiner ➕",
}
