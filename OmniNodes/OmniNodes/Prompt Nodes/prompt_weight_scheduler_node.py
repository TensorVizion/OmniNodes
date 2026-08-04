"""
TensorVizion ComfyUI Nodes
prompt_weight_scheduler_node.py — Interpolates a token's (token:weight)
emphasis across a frame/step range, with linear or eased curves. Built for
animation and gradual-emphasis batch workflows; pairs naturally with CLIP
Text Weight for a static single-value version of the same idea.
"""


class PromptWeightSchedulerNode:
    """
    Computes a weight for `token` at `current_frame` out of `total_frames`,
    interpolating between `start_weight` and `end_weight` along the chosen
    `curve`, and returns a ready-to-use `(token:weight)` prompt fragment
    alongside the raw float.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "token": ("STRING", {"default": "masterpiece"}),
                "start_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.01}),
                "end_weight": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.01}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "total_frames": ("INT", {"default": 24, "min": 1, "max": 999999}),
                "curve": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("weighted_prompt", "weight")
    FUNCTION      = "run"

    @staticmethod
    def _ease(t, curve):
        if curve == "linear":
            return t
        if curve == "ease_in":
            return t * t
        if curve == "ease_out":
            return 1 - (1 - t) * (1 - t)
        if curve == "ease_in_out":
            return 3 * t * t - 2 * t * t * t
        return t

    def run(self, token, start_weight, end_weight, current_frame, total_frames, curve):
        t = 0.0 if total_frames <= 1 else current_frame / max(1, (total_frames - 1))
        t = min(max(t, 0.0), 1.0)
        t = self._ease(t, curve)
        weight = round(start_weight + (end_weight - start_weight) * t, 3)
        return (f"({token}:{weight})", weight)


NODE_CLASS_MAPPINGS = {
    "PromptWeightSchedulerNode": PromptWeightSchedulerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptWeightSchedulerNode": "Prompt Weight Scheduler ⏳",
}
