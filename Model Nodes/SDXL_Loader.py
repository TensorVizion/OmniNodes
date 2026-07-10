class SimpleSDXLLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_auto": ("BOOLEAN", {"default": True, "label": "Auto-load VAE"}),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
            }
        }
    
    RETURN_TYPES = ("MODEL", "VAE", "CLIP")
    FUNCTION = "load_sdxl"
    CATEGORY = "Model Loaders/SDXL"
    
    def load_sdxl(self, model_name, vae_auto, vae_name):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            model_path, 
            output_vae=True,
            output_clip=True
        )
        
        if vae_auto and not vae:
            # Auto-find matching VAE
            vae_name = model_name.replace(".safetensors", ".vae.safetensors")
            if folder_paths.exists_annotated_file("vae", vae_name):
                vae = comfy.sd.load_vae(vae_name)
        
        return (model, vae, clip)