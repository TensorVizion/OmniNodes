import torch
import torch.nn.functional as F


class VisualDNAMixer:
    """
    Decouples structural DNA (composition, layout) from aesthetic DNA
    (color, texture, mood) and recombines them at user-defined ratios.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "structure_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aesthetic_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_mode": (["linear", "frequency_split", "feature_swap"],),
                "frequency_split_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("mixed_image", "dna_report")
    FUNCTION = "mix_dna"
    CATEGORY = "creative/director_toolkit"

    def mix_dna(self, image_a, image_b, structure_weight, aesthetic_weight,
                blend_mode, frequency_split_ratio):
        # Resize b to match a if needed
        if image_a.shape[-2:] != image_b.shape[-2:]:
            image_b = F.interpolate(
                image_b.permute(0, 3, 1, 2),
                size=image_a.shape[-2:],
                mode='bilinear'
            ).permute(0, 2, 3, 1)

        if blend_mode == "linear":
            result = image_a * structure_weight + image_b * aesthetic_weight
        elif blend_mode == "frequency_split":
            result = self._frequency_blend(image_a, image_b, frequency_split_ratio,
                                           structure_weight, aesthetic_weight)
        else:
            result = self._feature_swap(image_a, image_b,
                                        structure_weight, aesthetic_weight)

        result = torch.clamp(result, 0.0, 1.0)

        report = (
            f"DNA MIX REPORT\n"
            f"Mode: {blend_mode}\n"
            f"Structure from A: {structure_weight:.2f}\n"
            f"Aesthetic from B: {aesthetic_weight:.2f}\n"
            f"Output shape: {tuple(result.shape)}"
        )
        return (result, report)

    def _frequency_blend(self, a, b, split, struct_w, aes_w):
        a_gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        b_gray = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]

        a_fft = torch.fft.fft2(a_gray)
        b_fft = torch.fft.fft2(b_gray)

        H, W = a_gray.shape[-2:]
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=a.device, dtype=torch.float32)
        x = torch.arange(W, device=a.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, dtype=torch.float32))
        mask = torch.sigmoid(10 * (dist / max_dist - split))

        combined_gray = torch.fft.ifft2(a_fft * mask + b_fft * (1 - mask)).real
        ratio = combined_gray / (a_gray + 1e-8)

        result = a.clone()
        for c in range(3):
            result[..., c] = a[..., c] * ratio * aes_w + b[..., c] * struct_w
        return result

    def _feature_swap(self, a, b, struct_w, aes_w):
        result = a.clone()
        for c in range(3):
            a_mean, a_std = a[..., c].mean(), a[..., c].std() + 1e-8
            b_mean, b_std = b[..., c].mean(), b[..., c].std() + 1e-8
            transferred = (a[..., c] - a_mean) / a_std * b_std + b_mean
            result[..., c] = transferred * aes_w + a[..., c] * struct_w
        return result