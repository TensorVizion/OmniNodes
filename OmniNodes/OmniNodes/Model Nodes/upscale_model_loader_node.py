"""
TensorVizion ComfyUI Nodes
upscale_model_loader_node.py — Loads an ESRGAN-family upscale model file
(RealESRGAN, 4x-UltraSharp, etc.) for use with ComfyUI's Upscale Image
(using Model) node. A thin, TensorVizion-branded wrapper around
ComfyUI's own core UpscaleModelLoader.
"""
from nodes import UpscaleModelLoader as _CoreUpscaleModelLoader

class UpscaleModelLoaderNode:
    """
    Loads `model_name` from the `upscale_models/` folder using ComfyUI's
    own core upscale-model-loading logic. Output plugs directly into
    ComfyUI's Upscale Image (using Model) node — this node only handles
    loading the model file itself, not running the upscale.
    """
    CATEGORY = "TensorVizion/Model Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return _CoreUpscaleModelLoader.INPUT_TYPES()

    RETURN_TYPES  = ("UPSCALE_MODEL", "STRING")
    RETURN_NAMES  = ("upscale_model",  "summary")
    FUNCTION      = "run"

    def run(self, model_name):
        core = _CoreUpscaleModelLoader()
        result = core.load_model(model_name)
        upscale_model = result[0]
        summary = f"Upscale model loaded: {model_name}"
        return (upscale_model, summary)

# --- Define mappings for OmniNodes ---
NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoaderNode": UpscaleModelLoaderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoaderNode": "Upscale Model Loader 🔭 (TensorVizion)",
}
# ---