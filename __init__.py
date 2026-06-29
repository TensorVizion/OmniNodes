"""
OmniNodes — ComfyUI Custom Node Pack by TensorVizion
__init__.py

Registers all 16 nodes across three sub-folders:

  Audio Nodes/      (7)  — audio_beat_detect, audio_mixer, audio_normalize,
                            audio_pitch_shift, audio_reverb, audio_spectrogram,
                            audio_waveform
  Model Utilities/  (6)  — clip_text_compare, clip_text_weight, lora_info,
                            lora_stack, model_block_freeze, model_merge_weighted
  Latent/           (3)  — latent_blend, latent_mask, latent_noise_inject

Nodes appear in ComfyUI under:
  TensorVizion/Audio
  TensorVizion/Model Utilities
  TensorVizion/Latent
"""

import importlib.util
import sys
import traceback
from pathlib import Path

# ── Top-level mappings (read by ComfyUI) ─────────────────────────────────────
NODE_CLASS_MAPPINGS:        dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# ── Node manifest ─────────────────────────────────────────────────────────────
# Defines load order and lets the loader warn about missing files explicitly.
_MANIFEST: dict = {
    "Audio Nodes": [
        "audio_beat_detect_node.py",
        "audio_mixer_node.py",
        "audio_normalize_node.py",
        "audio_pitch_shift_node.py",
        "audio_reverb_node.py",
        "audio_spectrogram_node.py",
        "audio_waveform_node.py",
    ],
    "Model Utilities": [
        "clip_text_compare_node.py",
        "clip_text_weight_node.py",
        "lora_info_node.py",
        "lora_stack_node.py",
        "model_block_freeze_node.py",
        "model_merge_weighted_node.py",
    ],
    "Latent": [
        "latent_blend_node.py",
        "latent_mask_node.py",
        "latent_noise_inject_node.py",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_module(py_file: Path):
    """
    Load a .py file as an isolated module via importlib.
    Namespaced as  omninodes.<folder>.<stem>  to avoid collisions with
    other custom node packs that may share file names.
    Returns the loaded module, or None on any failure.
    """
    module_name = f"omninodes.{py_file.parent.stem}.{py_file.stem}"

    # Return cached module on hot-reload
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        print(f"[OmniNodes] ⚠  spec_from_file_location failed: {py_file.name}")
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        print(f"[OmniNodes] ✗  Import error in {py_file.name}:")
        traceback.print_exc()
        del sys.modules[module_name]
        return None

    return module


def _merge(module, filename: str) -> int:
    """
    Merge NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from a module
    into the top-level dicts.  Returns the number of node classes merged.
    """
    class_map   = getattr(module, "NODE_CLASS_MAPPINGS",        None)
    display_map = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)

    if not isinstance(class_map, dict) or not class_map:
        print(f"[OmniNodes] ⚠  {filename}: NODE_CLASS_MAPPINGS missing or empty")
        return 0

    NODE_CLASS_MAPPINGS.update(class_map)
    if isinstance(display_map, dict):
        NODE_DISPLAY_NAME_MAPPINGS.update(display_map)

    return len(class_map)


# ── Load loop ─────────────────────────────────────────────────────────────────

_root         = Path(__file__).resolve().parent
_files_ok     = 0
_files_failed = 0
_total        = sum(len(v) for v in _MANIFEST.values())

for _folder_name, _expected_files in _MANIFEST.items():
    _folder = _root / _folder_name

    if not _folder.is_dir():
        print(f"[OmniNodes] ✗  Folder missing: '{_folder_name}/'  "
              f"({len(_expected_files)} nodes skipped)")
        _files_failed += len(_expected_files)
        continue

    # Warn about any expected file that is absent from disk
    for _fname in _expected_files:
        if not (_folder / _fname).exists():
            print(f"[OmniNodes] ⚠  File not found: {_folder_name}/{_fname}")

    # Load every *_node.py present in the folder (sorted for determinism)
    for _py in sorted(_folder.glob("*_node.py")):
        _mod = _load_module(_py)
        if _mod is None:
            _files_failed += 1
            continue

        _n = _merge(_mod, _py.name)
        if _n:
            _files_ok += 1
            print(f"[OmniNodes] ✓  {_folder_name}/{_py.name}")
        else:
            _files_failed += 1

# ── Startup summary ───────────────────────────────────────────────────────────
_nodes_registered = len(NODE_CLASS_MAPPINGS)
_status = "✓" if _files_failed == 0 else "⚠"
print(
    f"\n[OmniNodes] {_status}  "
    f"{_files_ok}/{_total} files loaded  ·  "
    f"{_nodes_registered} node{'s' if _nodes_registered != 1 else ''} registered"
    + (f"  ·  {_files_failed} failed" if _files_failed else "")
    + "\n"
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
