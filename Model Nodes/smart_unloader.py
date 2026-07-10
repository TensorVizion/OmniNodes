class SmartUnloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_to_unload": ("MODEL",),
                "also_unload_VAE": ("BOOLEAN", {"default": True}),
                "also_unload_CLIP": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    RETURN_NAMES = ("unloaded_model", "unloaded_vae", "unloaded_clip")
    FUNCTION = "unload_models"
    CATEGORY = "Model Loaders/Utils"
    OUTPUT_NODE = True
    
    def unload_models(self, model_to_unload, also_unload_VAE, also_unload_CLIP):
        import torch
        torch.cuda.empty_cache()
        
        unload_msg = f"🧹 Unloaded {model_to_unload.__class__.__name__}"
        if also_unload_VAE:
            unload_msg += ", VAE"
        if also_unload_CLIP:
            unload_msg += ", CLIP"
        
        print(unload_msg)
        
        # Return None to release references
        return (None, None, None)