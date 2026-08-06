import random
import json


class AestheticDriftGenerator:
    """
    Generates a sequence of prompts that walk along user-defined aesthetic
    axes, producing a coherent evolutionary series instead of random variations.
    """

    AXES = {
        "chaos_order": ["pristine orderly composition", "balanced structured layout",
                        "organic asymmetry", "controlled chaos", "fragmented explosion"],
        "organic_geometric": ["pure organic curves", "flowing natural forms",
                              "mixed organic and geometric", "clean geometric precision",
                              "rigid geometric abstraction"],
        "saturated_desaturated": ["monochrome greyscale", "muted desaturated palette",
                                  "balanced color saturation", "rich saturated tones",
                                  "hyper-saturated pop colors"],
        "intimate_epic": ["extreme intimate macro", "close personal space",
                          "human-scale framing", "wide environmental context", "vast epic scale"],
        "ancient_futuristic": ["ancient primordial aesthetic", "classical historical",
                               "contemporary modern", "near-future speculative",
                               "deep-future alien"]
    }

    @classmethod
    def INPUT_TYPES(cls):
        axes_list = list(cls.AXES.keys())
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "portrait of a wanderer", "multiline": True}),
                "active_axes": ("STRING", {"default": "chaos_order,organic_geometric", "multiline": True}),
                "start_position": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "end_position": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "step_count": ("INT", {"default": 5, "min": 2, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("drift_prompts", "drift_path")
    FUNCTION = "generate_drift"
    CATEGORY = "creative/director_toolkit"

    def generate_drift(self, base_prompt, active_axes, start_position,
                       end_position, step_count, seed):
        rng = random.Random(seed)
        axes = [a.strip() for a in active_axes.split(",") if a.strip() in self.AXES]

        if not axes:
            axes = ["chaos_order"]

        path = []
        prompts = []
        for step in range(step_count):
            t = step / max(step_count - 1, 1)
            pos = start_position + (end_position - start_position) * t
            axis_descriptors = []
            path_entry = {"step": step + 1, "t": round(t, 3), "position": round(pos, 3)}

            for axis in axes:
                bucket = self.AXES[axis]
                idx = min(int(pos * (len(bucket) - 1)), len(bucket) - 1)
                descriptor = bucket[idx]
                axis_descriptors.append(descriptor)
                path_entry[axis] = descriptor

            descriptor_str = ", ".join(axis_descriptors)
            prompt = f"{base_prompt}, {descriptor_str}, drift step {step + 1} of {step_count}"
            prompts.append(prompt)
            path.append(path_entry)

        return ("\n---\n".join(prompts), json.dumps(path, indent=2))