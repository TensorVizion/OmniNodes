"""
TensorVizion ComfyUI Nodes
empty_latent_image_node.py — The basic blank-latent starting canvas every
text-to-image workflow begins from, with SDXL-aware resolution presets
layered on top of ComfyUI's own core EmptyLatentImage so users don't have
to remember or look up which pixel dimensions are actually
SDXL-native/recommended.
"""

from nodes import EmptyLatentImage as _CoreEmptyLatentImage

# SDXL's officially trained/recommended resolutions (all multiples of 64,
# each near ~1024x1024 = ~1 megapixel, the native SDXL training area).
_PRESETS = {
    "custom":              None,
    "1024x1024 (square)":  (1024, 1024),
    "896x1152 (portrait)": (896, 1152),
    "832x1216 (portrait)": (832, 1216),
    "1152x896 (landscape)":(1152, 896),
    "1216x832 (landscape)":(1216, 832),
    "1344x768 (widescreen)": (1344, 768),
    "768x1344 (tall)":     (768, 1344),
    "512x512 (SD1.5 square)": (512, 512),
}


class EmptyLatentImageNode:
    """
    Creates an empty (pure noise-ready) latent batch. Pick a `preset` for
    a known-good SDXL-native resolution, or select `custom` to use the
    `width`/`height` fields directly (still snapped to the nearest
    multiple of 8, same as ComfyUI's own core node requires). Delegates
    the actual tensor creation straight to `nodes.EmptyLatentImage`, so
    behaviour matches the standard node exactly — this is a preset
    convenience layer on top, not a reimplementation.
    """

    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset":     (list(_PRESETS.keys()), {"default": "1024x1024 (square)"}),
                "width":      ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height":     ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES  = ("LATENT", "STRING")
    RETURN_NAMES  = ("latent",  "summary")
    FUNCTION      = "run"

    def run(self, preset, width, height, batch_size):
        preset_dims = _PRESETS.get(preset)
        if preset_dims is not None:
            width, height = preset_dims

        core = _CoreEmptyLatentImage()
        result = core.generate(width, height, batch_size)
        latent = result[0]

        summary = (
            f"Preset:     {preset}\n"
            f"Resolution: {width}x{height}\n"
            f"Batch size: {batch_size}\n"
            f"Megapixels: {(width * height) / 1_000_000:.2f}"
        )

        return (latent, summary)


NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageNode": EmptyLatentImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentImageNode": "Empty Latent Image ⬜",
}
