import importlib.util
import sys
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _try_merge_mappings(module):
    """
    Merge NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from a loaded module (if present).
    """
    class_mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    display_mappings = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)

    if isinstance(class_mappings, dict):
        NODE_CLASS_MAPPINGS.update(class_mappings)
    if isinstance(display_mappings, dict):
        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)

def _load_module_from_path(py_file: Path):
    """
    Load a module from an absolute .py path (no package import required).
    """
    module_name = f"omni_nodes_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module

# Find and load every *_node.py under the two node folders.
_root = Path(__file__).resolve().parent
_candidate_dirs = [
    _root / "Audio Nodes",
    _root / "Model Utilities",
]

for d in _candidate_dirs:
    if not d.exists() or not d.is_dir():
        continue
    for py in sorted(d.glob("*_node.py")):
        try:
            mod = _load_module_from_path(py)
            if mod is not None:
                _try_merge_mappings(mod)
        except Exception:
            # If a single node fails to import, we don't want to break the entire extension.
            # ComfyUI will still load nodes that successfully imported.
            continue