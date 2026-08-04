import importlib.util
import sys
import traceback
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_root = Path(__file__).resolve().parent # Fixed typo: was 'file'

# ─────────────────────────────────────────────────────────────
# Module loader (Updated to catch syntax errors during compilation)
# ─────────────────────────────────────────────────────────────
def _load_module(py_file: Path):
    module_name = f"omninodes.{py_file.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if not spec or not spec.loader:
            print(f"[OmniNodes] ❌ Cannot load spec: {py_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Execute the module, catching syntax errors during compilation/loading
        spec.loader.exec_module(module)
        return module
    except SyntaxError as e:
        # This catches errors like invalid syntax, indentation errors, etc.
        print(f"[OmniNodes] ❌ Syntax Error in file: {py_file}")
        print(f"           Error details: {e}")
        return None
    except Exception as e:
        # This catches other runtime errors during import
        print(f"[OmniNodes] ❌ Error importing: {py_file}")
        print(f"           Error details: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

# ... (Keep the rest of your functions: _merge, and the main loading loop) ...

# ─────────────────────────────────────────────────────────────
# Safe merger
# ─────────────────────────────────────────────────────────────
def _merge(module, file_path: Path):
    class_map = getattr(module, "NODE_CLASS_MAPPINGS", None)
    display_map = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if not isinstance(class_map, dict) or len(class_map) == 0:
        print(f"[OmniNodes] ⚠️ No NODE_CLASS_MAPPINGS in {file_path.name}")
        return 0
    NODE_CLASS_MAPPINGS.update(class_map)
    if isinstance(display_map, dict):
        NODE_DISPLAY_NAME_MAPPINGS.update(display_map)
    return len(class_map)

# ─────────────────────────────────────────────────────────────
# Recursive loader (ROBUST CORE)
# ─────────────────────────────────────────────────────────────
print("\n[OmniNodes] 🚀 Starting robust scan...\n")
py_files = list(_root.rglob("*.py"))
loaded = 0
failed = 0
skipped = 0
for py_file in py_files:
    # Skip this file itself
    if py_file.name == "__init__.py": # Fixed typo: was "init.py"
        continue
    # Skip cache
    if "__pycache__" in str(py_file):
        continue
    module = _load_module(py_file)
    if module is None:
        failed += 1
        continue
    n = _merge(module, py_file)
    if n > 0:
        loaded += 1
        print(f"[OmniNodes] ✅ Loaded: {py_file.relative_to(_root)} ({n} nodes)")
    else:
        skipped += 1

# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
print("\n[OmniNodes] ===============================")
print(f"[OmniNodes] Files scanned : {len(py_files)}")
print(f"[OmniNodes] Modules loaded: {loaded}")
print(f"[OmniNodes] Skipped       : {skipped}")
print(f"[OmniNodes] Failed        : {failed}")
print(f"[OmniNodes] Nodes total   : {len(NODE_CLASS_MAPPINGS)}")
print("[OmniNodes] ===============================\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] # Fixed typo: was "all"