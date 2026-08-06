import json


class MoodToParameterTranslator:
    """
    Maps natural-language mood descriptions to concrete sampler parameters.
    """

    MOOD_DB = {
        "melancholic": {"cfg": 7.5, "steps": 30, "denoise": 0.62, "sampler": "euler_ancestral"},
        "hopeful": {"cfg": 7.0, "steps": 25, "denoise": 0.65, "sampler": "dpmpp_2m"},
        "nostalgic": {"cfg": 8.0, "steps": 32, "denoise": 0.55, "sampler": "euler_ancestral"},
        "energetic": {"cfg": 6.5, "steps": 22, "denoise": 0.75, "sampler": "dpmpp_sde"},
        "serene": {"cfg": 7.0, "steps": 28, "denoise": 0.58, "sampler": "dpmpp_2m"},
        "ominous": {"cfg": 9.0, "steps": 35, "denoise": 0.50, "sampler": "euler"},
        "dreamy": {"cfg": 6.8, "steps": 30, "denoise": 0.70, "sampler": "euler_ancestral"},
        "chaotic": {"cfg": 8.5, "steps": 28, "denoise": 0.80, "sampler": "dpmpp_sde"},
        "clinical": {"cfg": 7.2, "steps": 26, "denoise": 0.60, "sampler": "dpmpp_2m"},
        "romantic": {"cfg": 6.9, "steps": 30, "denoise": 0.66, "sampler": "euler_ancestral"},
        "anxious": {"cfg": 8.8, "steps": 32, "denoise": 0.55, "sampler": "dpmpp_sde"},
        "triumphant": {"cfg": 7.0, "steps": 24, "denoise": 0.72, "sampler": "dpmpp_2m"},
        "mysterious": {"cfg": 8.5, "steps": 34, "denoise": 0.58, "sampler": "euler"},
        "playful": {"cfg": 6.5, "steps": 22, "denoise": 0.78, "sampler": "dpmpp_sde"},
        "somber": {"cfg": 8.2, "steps": 32, "denoise": 0.52, "sampler": "euler"},
        "vibrant": {"cfg": 6.7, "steps": 24, "denoise": 0.74, "sampler": "dpmpp_2m"},
    }

    LIGHTING_KEYWORDS = {
        "noir": "high-contrast monochrome key light",
        "golden hour": "warm low-angle rim light",
        "neon": "saturated RGB color separation",
        "overcast": "soft diffused omnidirectional",
        "studio": "controlled three-point setup",
        "moonlight": "cool blue side key",
        "candlelight": "warm flickering low-intensity",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mood_description": ("STRING", {"default": "melancholic but hopeful, like 4am in a strange city", "multiline": True}),
                "intensity": ("INT", {"default": 7, "min": 1, "max": 10}),
                "target_quality": (["draft", "production"],),
                "quality_boost_steps": ("INT", {"default": 4, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("sampler_config", "lighting_suggestion", "reasoning_log")
    FUNCTION = "translate"
    CATEGORY = "creative/director_toolkit"

    def translate(self, mood_description, intensity, target_quality, quality_boost_steps):
        desc_lower = mood_description.lower()
        detected_moods = [m for m in self.MOOD_DB if m in desc_lower]

        if not detected_moods:
            detected_moods = ["dreamy"]  # safe default

        # Average the parameters of all detected moods, weighted equally
        configs = [self.MOOD_DB[m] for m in detected_moods]
        avg_cfg = sum(c["cfg"] for c in configs) / len(configs)
        avg_steps = sum(c["steps"] for c in configs) / len(configs)
        avg_denoise = sum(c["denoise"] for c in configs) / len(configs)

        # Intensity modifier: higher intensity = lower CFG (let model breathe more)
        intensity_modifier = (intensity - 5) * 0.1
        cfg = max(1.0, avg_cfg - intensity_modifier)

        # Quality modifier
        if target_quality == "production":
            steps = int(avg_steps + quality_boost_steps)
        else:
            steps = int(avg_steps)

        # Pick sampler from first detected mood (could be smarter with voting)
        sampler = configs[0]["sampler"]

        # Detect lighting keywords
        lighting = "natural unspecified lighting"
        for kw, desc in self.LIGHTING_KEYWORDS.items():
            if kw in desc_lower:
                lighting = desc
                break

        config = {
            "sampler": sampler,
            "cfg": round(cfg, 2),
            "steps": steps,
            "denoise": round(avg_denoise, 2),
            "scheduler": "karras" if "ancestral" in sampler or "sde" in sampler else "normal"
        }

        reasoning = (
            f"Detected moods: {', '.join(detected_moods)}\n"
            f"Intensity {intensity}/10 -> CFG offset {intensity_modifier:+.1f}\n"
            f"Quality: {target_quality} (boost {quality_boost_steps} steps)\n"
            f"Selected sampler: {sampler}"
        )

        return (json.dumps(config, indent=2), lighting, reasoning)