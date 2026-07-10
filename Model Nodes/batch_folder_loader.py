class BatchFolderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "models/checkpoints/"}),
                "filter_extensions": ("STRING", {"default": ".safetensors,.ckpt"}),
                "load_first": ("BOOLEAN", {"default": True, "label": "Load first model"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP", "STRING")
    RETURN_NAMES = ("model", "vae", "clip", "loaded_models_list")
    FUNCTION = "load_batch"
    CATEGORY = "Model Loaders/Batch"
    
    def load_batch(self, folder_path, filter_extensions, load_first):
        import os
        extensions = [ext.strip() for ext in filter_extensions.split(",")]
        
        models = []
        for file in os.listdir(folder_path):
            if any(file.endswith(ext) for ext in extensions):
                models.append(file)
        
        model_list = "\n".join(models[:10])  # Show first 10
        if len(models) > 10:
            model_list += f"\n... and {len(models) - 10} more"
        
        model, vae, clip = None, None, None
        if load_first and models:
            model_path = os.path.join(folder_path, models[0])
            model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                model_path, output_vae=True, output_clip=True
            )
        
        return (model, vae, clip, f"📂 Found {len(models)} models:\n{model_list}")