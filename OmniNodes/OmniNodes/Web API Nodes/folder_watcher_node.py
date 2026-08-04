"""
TensorVizion ComfyUI Nodes
folder_watcher_node.py — Scans a folder and returns the next file not
yet recorded in a "processed" manifest, then (optionally) records it as
processed. Enables simple queue-style batch processing — drop files
into a folder over time, each queue run picks up the next unhandled one
— without needing a real job queue or database.
"""

import os
import json


class FolderWatcherNode:
    """
    Scans `folder_path` for files matching `extensions` (comma-separated,
    e.g. "png,jpg,jpeg"), sorted alphabetically for a stable, predictable
    order. Compares against a manifest JSON file (`manifest_path` —
    defaults to `<folder_path>/.tensorvizion_processed.json` if left
    blank) listing filenames already handled.

    Returns the first matching file NOT in the manifest. If
    `mark_processed_immediately` is True, that file is added to the
    manifest as soon as it's returned (appropriate if downstream
    processing is expected to always succeed); if False, nothing is
    marked and you should wire a separate call to mark it done only
    after downstream processing actually completes — pair with a second
    Folder Watcher call in "mark_only" mode for that pattern.

    `mode`:
      scan_and_return — find and return the next unprocessed file (does
                          NOT touch the manifest unless
                          mark_processed_immediately is True)
      mark_only        — takes `mark_filename` and adds just that one
                          filename to the manifest, without scanning.
                          Use this after downstream processing succeeds,
                          if you didn't use mark_processed_immediately.
    """

    CATEGORY = "TensorVizion/Web API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "mode": (["scan_and_return", "mark_only"],),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp"}),
                "manifest_path": ("STRING", {"default": ""}),
                "mark_processed_immediately": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mark_filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("next_file_path", "next_file_name", "remaining_count", "total_count", "summary")
    FUNCTION = "run"

    def _manifest_path(self, folder_path, manifest_path):
        if manifest_path.strip():
            return manifest_path.strip()
        return os.path.join(folder_path, ".tensorvizion_processed.json")

    def _load_manifest(self, path):
        if not os.path.isfile(path):
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed", []))
        except (json.JSONDecodeError, OSError):
            return set()  # treat a corrupt manifest as empty rather than crashing

    def _save_manifest(self, path, processed_set):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"processed": sorted(processed_set)}, f, indent=2)

    def run(self, folder_path, mode, extensions, manifest_path, mark_processed_immediately, mark_filename=""):
        folder_path = folder_path.strip()
        if not os.path.isdir(folder_path):
            msg = f"Folder not found: {folder_path}"
            return ("", "", 0, 0, msg)

        m_path = self._manifest_path(folder_path, manifest_path)
        processed = self._load_manifest(m_path)

        if mode == "mark_only":
            if not mark_filename.strip():
                return ("", "", 0, 0, "mode=mark_only but no mark_filename given.")
            processed.add(mark_filename.strip())
            self._save_manifest(m_path, processed)
            return ("", mark_filename.strip(), 0, 0, f"Marked '{mark_filename.strip()}' as processed.")

        ext_list = tuple(f".{e.strip().lower().lstrip('.')}" for e in extensions.split(",") if e.strip())
        all_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(ext_list))
        unprocessed = [f for f in all_files if f not in processed]

        if not unprocessed:
            summary = f"No unprocessed files remaining ({len(all_files)} total, all processed)."
            return ("", "", 0, len(all_files), summary)

        next_file = unprocessed[0]
        next_path = os.path.join(folder_path, next_file)

        if mark_processed_immediately:
            processed.add(next_file)
            self._save_manifest(m_path, processed)

        remaining = len(unprocessed) - (1 if mark_processed_immediately else 0)
        summary = f"Next: {next_file} ({remaining} remaining of {len(all_files)} total)"
        return (next_path, next_file, remaining, len(all_files), summary)


NODE_CLASS_MAPPINGS = {
    "FolderWatcherNode": FolderWatcherNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderWatcherNode": "Folder Watcher 👁️",
}
