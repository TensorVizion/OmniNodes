import torch
import json


class PromptArchaeologyEngine:
    """
    Reverse-engineers a reference image into a structured, reusable prompt
    scaffold by analyzing composition, lighting, palette, and depth cues.
    """

    COMPOSITIONS = {
        "rule_of_thirds": "subject placed at one-third intersection",
        "center_weighted": "subject anchored in central frame",
        "symmetric": "mirrored bilateral composition",
        "diagonal": "strong diagonal leading lines",
        "layered": "distinct foreground, midground, and background layers",
    }

    LIGHTING_QUALITIES = {
        "high_key": "bright, low-contrast, airy lighting",
        "low_key": "dark, high-contrast, dramatic lighting",
        "rembrandt": "Rembrandt triangle lighting with strong shadow falloff",
        "flat": "flat, even, shadowless illumination",
        "rim": "strong backlit rim lighting separating subject from background",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "analysis_depth": (["surface", "deep", "forensic"],),
                "output_format": (["natural_language", "tags", "structured"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_scaffold", "analysis_breakdown")
    FUNCTION = "excavate"
    CATEGORY = "creative/director_toolkit"

    def excavate(self, reference_image, analysis_depth, output_format):
        img = reference_image[0]  # first image in batch
        H, W, C = img.shape

        # Convert to grayscale for analysis
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

        # Brightness statistics -> key (high/low)
        mean_brightness = float(gray.mean())
        std_brightness = float(gray.std())
        key = "high_key" if mean_brightness > 0.6 else "low_key" if mean_brightness < 0.35 else "balanced"

        # Color palette extraction (mean color)
        palette_rgb = [float(img[..., c].mean()) for c in range(3)]
        warmth = palette_rgb[0] - palette_rgb[2]  # red minus blue
        palette_descriptor = "warm" if warmth > 0.05 else "cool" if warmth < -0.05 else "neutral"
        saturation = max(palette_rgb) - min(palette_rgb)
        sat_descriptor = "saturated" if saturation > 0.15 else "muted" if saturation < 0.05 else "moderate saturation"

        # Composition heuristics
        thirds_x1, thirds_y1 = W / 3, H / 3
        thirds_x2, thirds_y2 = 2 * W / 3, 2 * H / 3

        # Brightness gradient -> lighting direction
        left_brightness = float(gray[:, :W // 4].mean())
        right_brightness = float(gray[:, 3 * W // 4:].mean())
        top_brightness = float(gray[:H // 4, :].mean())
        bottom_brightness = float(gray[3 * H // 4:, :].mean())

        light_dir = "front-lit"
        if left_brightness - right_brightness > 0.08:
            light_dir = "lit from the left"
        elif right_brightness - left_brightness > 0.08:
            light_dir = "lit from the right"
        if top_brightness - bottom_brightness > 0.08:
            light_dir += " with overhead source"
        elif bottom_brightness - top_brightness > 0.08:
            light_dir += " with underlight"

        # Depth cue: contrast gradient top-to-bottom suggests depth
        top_third_contrast = float(gray[:H // 3, :].std())
        bottom_third_contrast = float(gray[2 * H // 3:, :].std())
        depth_cue = "atmospheric haze with receding depth" if bottom_third_contrast < top_third_contrast * 0.7 \
            else "sharp depth of field" if top_third_contrast > bottom_third_contrast * 1.3 \
            else "balanced depth"

        # Quality based on analysis depth
        if analysis_depth == "surface":
            descriptor_count = 4
        elif analysis_depth == "deep":
            descriptor_count = 7
        else:
            descriptor_count = 10

        # Build scaffold
        scaffold_parts = []
        scaffold_parts.append(f"[SUBJECT] (your subject here)")
        scaffold_parts.append(f"composition: {self.COMPOSITIONS['rule_of_thirds']}")
        scaffold_parts.append(f"lighting: {self.LIGHTING_QUALITIES[key]}, {light_dir}")
        scaffold_parts.append(f"palette: {palette_descriptor} {sat_descriptor}")
        scaffold_parts.append(f"depth: {depth_cue}")

        if descriptor_count >= 7:
            scaffold_parts.append(f"camera: [lens type], [focal length], [aperture]")
            scaffold_parts.append(f"texture: [surface qualities like grain, smooth, rough]")
            scaffold_parts.append(f"style: [art movement or visual reference]")

        if descriptor_count >= 10:
            scaffold_parts.append(f"mood: [emotional tone]")
            scaffold_parts.append(f"era: [time period aesthetic]")
            scaffold_parts.append(f"focal hierarchy: [primary/secondary subject priority]")

        if output_format == "natural_language":
            prompt = (
                f"A [SUBJECT] in {self.LIGHTING_QUALITIES[key]}, {light_dir}. "
                f"The composition follows {self.COMPOSITIONS['rule_of_thirds']}. "
                f"Color palette is {palette_descriptor} with {sat_descriptor}. "
                f"{depth_cue.capitalize()}."
            )
        elif output_format == "tags":
            prompt = ", ".join(scaffold_parts)
        else:
            prompt = json.dumps(dict(
                zip([p.split(":")[0] for p in scaffold_parts],
                    [":".join(p.split(":")[1:]).strip() for p in scaffold_parts])
            ), indent=2)

        breakdown = (
            f"Mean brightness: {mean_brightness:.3f}\n"
            f"Brightness std: {std_brightness:.3f}\n"
            f"Key classification: {key}\n"
            f"Palette (R,G,B): {palette_rgb[0]:.3f}, {palette_rgb[1]:.3f}, {palette_rgb[2]:.3f}\n"
            f"Warmth score: {warmth:+.3f}\n"
            f"Saturation: {saturation:.3f}\n"
            f"Light direction: {light_dir}\n"
            f"Depth cue: {depth_cue}"
        )

        return (prompt, breakdown)
