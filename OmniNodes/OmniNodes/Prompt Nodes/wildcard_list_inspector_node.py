"""
TensorVizion ComfyUI Nodes
wildcard_list_inspector_node.py — Lists every wildcard file Wildcard Loader
can see, with line counts and a short preview, so you don't have to open a
file browser to remember what's available or what a file is called.
"""

import os

import folder_paths


class WildcardListInspectorNode:
    """
    Scans the same wildcards/ folder Wildcard Loader reads from (ComfyUI
    root's wildcards/ if present, otherwise this pack's bundled wildcards/
    folder) and reports every .txt file found: its wildcard name (what
    goes inside __double_underscores__), how many usable lines it has, and
    a short preview of the first few.

    Run this once to see what's available, then reference the names
    directly in a Wildcard Loader prompt.
    """

    CATEGORY = "TensorVizion/Prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_lines": ("INT", {"default": 3, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES  = ("STRING", "INT")
    RETURN_NAMES  = ("listing", "file_count")
    FUNCTION      = "inspect"

    def inspect(self, preview_lines):
        # Re-implement the same directory resolution locally rather than
        # importing the sibling module by path (ComfyUI's per-file loader
        # doesn't guarantee import order between files), so this node
        # works standalone even if loaded before wildcard_loader_node.py.
        base_path = getattr(folder_paths, "base_path", None)
        wildcards_dir = None
        if base_path:
            candidate = os.path.join(base_path, "wildcards")
            if os.path.isdir(candidate):
                wildcards_dir = candidate
        if wildcards_dir is None:
            local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wildcards")
            wildcards_dir = local_dir

        if not os.path.isdir(wildcards_dir):
            return (f"No wildcards folder found yet at:\n{wildcards_dir}\n\n"
                    f"Run Wildcard Loader once — it creates this folder with a couple of starter files.", 0)

        entries = []
        for root, _dirs, files in os.walk(wildcards_dir):
            for f in sorted(files):
                if not f.lower().endswith(".txt"):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, wildcards_dir).replace("\\", "/")[:-4]
                try:
                    with open(full, "r", encoding="utf-8") as fh:
                        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
                except Exception:
                    lines = []
                entries.append((rel, lines))

        entries.sort(key=lambda e: e[0])

        if not entries:
            return (f"Wildcards folder is empty:\n{wildcards_dir}", 0)

        out = [f"Wildcards folder: {wildcards_dir}", f"Found {len(entries)} file(s):", ""]
        for name, lines in entries:
            out.append(f"__{name}__  ({len(lines)} line{'s' if len(lines) != 1 else ''})")
            for p in lines[:preview_lines]:
                out.append(f"    - {p}")
            if len(lines) > preview_lines:
                out.append(f"    ... and {len(lines) - preview_lines} more")
            out.append("")

        return ("\n".join(out).rstrip(), len(entries))


NODE_CLASS_MAPPINGS = {
    "WildcardListInspectorNode": WildcardListInspectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WildcardListInspectorNode": "Wildcard List Inspector 📋",
}
