class QuickLoRAStacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "lora_1": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lora_2": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lora_3": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "stack_loras"
    CATEGORY = "Model Loaders/LoRA"
    
    def stack_loras(self, model, clip, lora_1=None, strength_1=0.5, 
                    lora_2=None, strength_2=0.5, lora_3=None, strength_3=0.5):
        lorasa = [(lora_1, strength_1), (lora_2, strength_2), (lora_3, strength_3)]
        
        for lora_name, strength in lorasa:
            if lora_name and lora_name != "None" and strength > 0:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_path, strength, strength)
        
        return (model, clip)