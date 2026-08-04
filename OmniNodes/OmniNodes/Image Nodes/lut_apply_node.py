"""
TensorVizion ComfyUI Nodes
lut_apply_node.py — Loads a 3D .cube LUT file and applies it to an IMAGE
via trilinear interpolation. Image Color Grade covers manual exposure/
contrast/temperature adjustments; this is the complementary tool for
professional LUT-driven color grading workflows (film-emulation packs,
"cinematic look" LUTs, etc.), which are a different and very common
grading approach that manual sliders can't replicate.
"""

import os
import numpy as np
import torch


class LUTApplyNode:
    """
    Parses a standard Adobe/IRIDAS .cube file: reads the `LUT_3D_SIZE N`
    header, then N^3 RGB triplets with red varying fastest, then green,
    then blue (the standard .cube data ordering). `DOMAIN_MIN`/
    `DOMAIN_MAX` are read if present (default 0-1, the overwhelming
    majority of real-world LUTs); non-default domains are rescaled
    before lookup.

    Uses trilinear interpolation to sample the LUT smoothly — a LUT's
    grid points (commonly 17, 33, or 65 per axis) don't line up with an
    image's actual pixel values, so nearest-neighbor lookup would
    produce visible color-banding on smooth gradients. `strength` blends
    the graded result back with the original image (1.0 = fully graded,
    0.0 = unchanged) for adjustable grade intensity without needing a
    separate blend node.

    LUT files only ever contain RGB float data — no executable content
    is possible in this format, so loading an untrusted .cube file
    carries no code-execution risk (unlike, say, a pickle-based file).
    """

    CATEGORY = "TensorVizion/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "summary")
    FUNCTION = "process"

    # ------------------------------------------------------------------
    def _parse_cube(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LUT file not found: {path}")

        size = None
        domain_min = np.array([0.0, 0.0, 0.0])
        domain_max = np.array([1.0, 1.0, 1.0])
        data = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("TITLE"):
                    continue
                if line.upper().startswith("LUT_1D_SIZE"):
                    raise ValueError("This is a 1D LUT file (LUT_1D_SIZE) — this node only supports 3D LUTs (LUT_3D_SIZE).")
                if line.upper().startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                    continue
                if line.upper().startswith("DOMAIN_MIN"):
                    domain_min = np.array([float(x) for x in line.split()[1:4]])
                    continue
                if line.upper().startswith("DOMAIN_MAX"):
                    domain_max = np.array([float(x) for x in line.split()[1:4]])
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue  # skip any stray non-numeric line rather than hard-failing

        if size is None:
            raise ValueError(f"No LUT_3D_SIZE header found in {path} — not a valid 3D .cube file.")
        expected = size ** 3
        if len(data) != expected:
            raise ValueError(
                f"Expected {expected} data rows for LUT_3D_SIZE {size}, found {len(data)} in {path}. "
                f"File may be truncated or malformed."
            )

        # .cube ordering: red fastest, then green, then blue -> reshape as (B,G,R,3) then transpose to (R,G,B,3).
        lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
        lut = np.transpose(lut, (2, 1, 0, 3))  # now indexed [r_idx, g_idx, b_idx]
        return lut, size, domain_min, domain_max

    def _trilinear_sample(self, lut, size, rgb):
        # rgb: (N, 3) float32 in [0,1] after domain normalization.
        scaled = rgb * (size - 1)
        idx0 = np.floor(scaled).astype(np.int32)
        idx0 = np.clip(idx0, 0, size - 2)
        frac = scaled - idx0

        r0, g0, b0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
        fr, fg, fb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

        c000 = lut[r0, g0, b0]
        c100 = lut[r1, g0, b0]
        c010 = lut[r0, g1, b0]
        c110 = lut[r1, g1, b0]
        c001 = lut[r0, g0, b1]
        c101 = lut[r1, g0, b1]
        c011 = lut[r0, g1, b1]
        c111 = lut[r1, g1, b1]

        c00 = c000 * (1 - fr) + c100 * fr
        c10 = c010 * (1 - fr) + c110 * fr
        c01 = c001 * (1 - fr) + c101 * fr
        c11 = c011 * (1 - fr) + c111 * fr

        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg

        return c0 * (1 - fb) + c1 * fb

    def process(self, image, lut_path, strength):
        try:
            lut, size, domain_min, domain_max = self._parse_cube(lut_path.strip())
        except (FileNotFoundError, ValueError) as e:
            summary = f"LUT load failed: {e}"
            return (image, summary)

        img_np = image.cpu().numpy().astype(np.float32)  # (B,H,W,3) in [0,1]
        B, H, W, C = img_np.shape

        domain_range = np.maximum(domain_max - domain_min, 1e-6)
        flat = img_np.reshape(-1, C)[:, :3]
        normalized = np.clip((flat - domain_min) / domain_range, 0.0, 1.0)

        graded = self._trilinear_sample(lut, size, normalized)
        graded = graded.reshape(B, H, W, 3)

        if C == 4:
            graded = np.concatenate([graded, img_np[..., 3:4]], axis=-1)

        blended = img_np * (1.0 - strength) + graded * strength
        blended = np.clip(blended, 0.0, 1.0)

        summary = f"Applied {os.path.basename(lut_path)} (size={size}, strength={strength:.2f})"
        return (torch.from_numpy(blended.astype(np.float32)), summary)


NODE_CLASS_MAPPINGS = {
    "LUTApplyNode": LUTApplyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LUTApplyNode": "3D LUT Apply 🎞️",
}
