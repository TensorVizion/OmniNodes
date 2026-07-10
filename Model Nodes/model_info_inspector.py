class ModelInfoInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "inspect_model"
    CATEGORY = "Model Loaders/Info"
    OUTPUT_NODE = True
    
    def inspect_model(self, model_name):
        import json
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        
        info = {}
        try:
            # Read model metadata without loading
            from safetensors import safe_open
            with safe_open(model_path, framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    info = json.loads(metadata.get("ss_metadata", "{}"))
        except:
            info = {"error": "Could not read metadata"}
        
        # Format readable output
        output = f"📁 Model: {model_name}\n"
        for key, value in list(info.items())[:5]:
            output += f"  • {key}: {value}\n"
        
        return (output,)