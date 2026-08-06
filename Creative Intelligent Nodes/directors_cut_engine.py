import random
import json


class DirectorsCutEngine:
    """
    Expands a base prompt into a batch of shot-specific conditioning prompts
    using cinematic grammar.
    """

    SHOT_TYPES = ["extreme wide establishing shot", "wide shot", "medium shot",
                  "medium close-up", "close-up", "extreme close-up detail"]
    CAMERA_MOVES = ["static locked frame", "slow dolly push-in", "orbiting tracking shot",
                    "handheld with subtle sway", "smooth lateral pan", "tilt reveal"]
    LIGHTING = ["golden hour warmth", "overcast soft diffusion", "high-contrast noir lighting",
                "neon-lit nighttime", "dappled natural light", "studio three-point"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "woman in a cafe", "multiline": True}),
                "style_keywords": ("STRING", {"default": "cinematic, shallow depth of field", "multiline": True}),
                "variation_count": ("INT", {"default": 6, "min": 2, "max": 12}),
                "shot_distribution": (["even", "ascending_intimacy", "descending_intimacy", "bookend_wides"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("combined_prompts", "shot_metadata")
    FUNCTION = "generate_shots"
    CATEGORY = "creative/director_toolkit"

    def generate_shots(self, base_prompt, style_keywords, variation_count,
                       shot_distribution, seed):
        rng = random.Random(seed)

        # Determine shot indices based on distribution
        if shot_distribution == "even":
            shots = [self.SHOT_TYPES[i % len(self.SHOT_TYPES)] for i in range(variation_count)]
        elif shot_distribution == "ascending_intimacy":
            shots = [self.SHOT_TYPES[min(i, len(self.SHOT_TYPES) - 1)]
                     for i in range(variation_count)]
        elif shot_distribution == "descending_intimacy":
            shots = [self.SHOT_TYPES[len(self.SHOT_TYPES) - 1 - min(i, len(self.SHOT_TYPES) - 1)]
                     for i in range(variation_count)]
        else:  # bookend_wides
            shots = [self.SHOT_TYPES[0] if (i == 0 or i == variation_count - 1)
                     else self.SHOT_TYPES[2 + (i % 3)]
                     for i in range(variation_count)]

        prompts = []
        metadata = []
        for i in range(variation_count):
            shot = shots[i]
            cam = rng.choice(self.CAMERA_MOVES)
            light = rng.choice(self.LIGHTING)
            prompt = (
                f"{shot} of {base_prompt}, {cam}, {light}, "
                f"{style_keywords}, shot {i + 1} of {variation_count}"
            )
            prompts.append(prompt)
            metadata.append({
                "shot_number": i + 1,
                "shot_type": shot,
                "camera": cam,
                "lighting": light
            })

        combined = "\n---\n".join(prompts)
        meta_json = json.dumps(metadata, indent=2)
        return (combined, meta_json)