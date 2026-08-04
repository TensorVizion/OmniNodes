"""
TensorVizion ComfyUI Nodes
negative_prompt_presets_node.py — A dropdown of common negative-prompt
boilerplate (quality/anatomy/style presets), concatenated with optional
custom text, output as a ready-to-encode STRING. Saves retyping the same
"worst quality, blurry, extra limbs..." boilerplate in every workflow,
while still allowing full customization.
"""

_PRESETS = {
    "none": "",
    "general quality": (
        "worst quality, low quality, blurry, jpeg artifacts, watermark, "
        "signature, text, cropped, out of frame"
    ),
    "anatomy fixes": (
        "bad anatomy, bad hands, extra limbs, extra fingers, missing fingers, "
        "fused fingers, malformed hands, mutated hands, long neck, "
        "disfigured, deformed"
    ),
    "general quality + anatomy": (
        "worst quality, low quality, blurry, jpeg artifacts, watermark, "
        "signature, text, cropped, out of frame, bad anatomy, bad hands, "
        "extra limbs, extra fingers, missing fingers, fused fingers, "
        "malformed hands, mutated hands, long neck, disfigured, deformed"
    ),
    "photography (avoid illustration look)": (
        "illustration, painting, drawing, art, sketch, cartoon, anime, "
        "3d render, cgi, blurry, low quality, overexposed, underexposed"
    ),
    "anime/illustration (avoid photo-real look)": (
        "photo, photorealistic, realistic, 3d render, blurry, low quality, "
        "worst quality, jpeg artifacts, watermark, signature"
    ),
}


class NegativePromptPresetsNode:
    """
    Picks a `preset` of common negative-prompt boilerplate and appends
    optional `custom_text` after it (comma-separated) — output is a plain
    STRING ready to feed straight into CLIP Text Encode (Simple)'s `text`
    input with `is_positive` set to False. `none` outputs only whatever's
    in `custom_text`, letting this node also just act as a pass-through
    when you don't want any preset at all.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset":      (list(_PRESETS.keys()), {"default": "general quality + anatomy"}),
                "custom_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("negative_prompt", "summary")
    FUNCTION      = "run"

    def run(self, preset, custom_text):
        preset_text = _PRESETS.get(preset, "")
        custom = custom_text.strip()

        if preset_text and custom:
            combined = f"{preset_text}, {custom}"
        elif preset_text:
            combined = preset_text
        else:
            combined = custom

        summary = (
            f"Preset:       {preset}\n"
            f"Custom added: {'yes' if custom else 'no'}\n"
            f"Total length: {len(combined)} chars"
        )

        return (combined, summary)


NODE_CLASS_MAPPINGS = {
    "NegativePromptPresetsNode": NegativePromptPresetsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NegativePromptPresetsNode": "Negative Prompt Presets 🚫",
}
