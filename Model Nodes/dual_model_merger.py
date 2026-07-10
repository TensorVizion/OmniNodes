class DualModelMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "interpolation": (["Weighted Sum", "Add Difference", "Slerp"],),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge_models"
    CATEGORY = "Model Loaders/Merge"
    
    def merge_models(self, model_A, model_B, merge_ratio, interpolation):
        from comfy.model_merging import merge_models
        
        merged = merge_models(
            model_A, model_B, 
            merge_ratio, 
            interpolation_method=interpolation.lower().replace(" ", "_")
        )
        return (merged,)