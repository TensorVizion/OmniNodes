import torch
import torch.nn.functional as F


class LatentFrequencyBandSplitter:
    """
    Decomposes a latent tensor into three frequency bands (macro / mid / micro)
    and recombines user-selected bands after optional processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "process_macro": ("BOOLEAN", {"default": True}),
                "process_mid": ("BOOLEAN", {"default": True}),
                "process_micro": ("BOOLEAN", {"default": False}),
                "macro_cutoff": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.05}),
                "mid_cutoff": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 0.9, "step": 0.05}),
                "recombination": (["sum", "macro_dominant", "micro_dominant"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("processed_latent",)
    FUNCTION = "split_bands"
    CATEGORY = "creative/director_toolkit"

    def split_bands(self, latent, process_macro, process_mid, process_micro,
                    macro_cutoff, mid_cutoff, recombination):
        samples = latent["samples"]
        H, W = samples.shape[-2:]

        # Frequency-domain mask for two cutoffs
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=samples.device, dtype=torch.float32)
        x = torch.arange(W, device=samples.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, dtype=torch.float32))
        norm = dist / max_dist

        # Low-pass (macro), mid-band, high-pass (micro)
        macro_mask = (norm <= macro_cutoff).float()
        mid_mask = ((norm > macro_cutoff) & (norm <= mid_cutoff)).float()
        micro_mask = (norm > mid_cutoff).float()

        fft = torch.fft.fft2(samples)
        macro = torch.fft.ifft2(fft * macro_mask).real
        mid = torch.fft.ifft2(fft * mid_mask).real
        micro = torch.fft.ifft2(fft * micro_mask).real

        if recombination == "sum":
            result = (macro if process_macro else 0) + \
                     (mid if process_mid else 0) + \
                     (micro if process_micro else 0)
        elif recombination == "macro_dominant":
            result = (macro if process_macro else 0) + \
                     0.5 * (mid if process_mid else 0) + \
                     0.25 * (micro if process_micro else 0)
        else:  # micro_dominant
            result = 0.5 * (macro if process_macro else 0) + \
                     0.75 * (mid if process_mid else 0) + \
                     (micro if process_micro else 0)

        new_latent = {"samples": result}
        return (new_latent,)