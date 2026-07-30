"""
TensorVizion ComfyUI Nodes
response_saver_node.py — Exit point for Web API workflows. Writes an HTTP
response (JSON or raw text) to a file on disk, with the same
collision-avoiding numbered-filename convention used by Video Save and
Custom Folder Batch Saver. HTTP Request / RSS Feed Parser / JSON Field
Extractor all produce data in memory; nothing in the Web API category
persisted it to disk until now.
"""

import os
import json


class AlwaysEqualProxy(str):
    """Wildcard type marker — see Workflow Nodes/any_switch_node.py for
    the full rationale. Re-declared locally so this file has no
    import-order dependency on another folder under the pack's per-file
    loader."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


class ResponseSaverNode:
    """
    Writes `content` to `output_dir/filename.ext`. Accepts either a JSON
    value (dict/list — pretty-printed on save) or a plain string (saved
    verbatim) on the same `content` socket, so it can sit directly after
    HTTP Request's `response_json` or `response_text` output, or after
    JSON Field Extractor's `value_string`/`value_raw`.

    If a file with the target name already exists, appends `_001`,
    `_002`, etc. rather than overwriting — matching the numbering
    convention Video Save and Custom Folder Batch Saver use elsewhere in
    the pack, so repeat workflow runs never silently clobber a previous
    save.
    """

    CATEGORY = "TensorVizion/Web API"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": (ANY_TYPE,),
                "output_dir": ("STRING", {"default": "output/tensorvizion/web_api"}),
                "filename": ("STRING", {"default": "response"}),
                "format": (["json", "txt"], {"default": "json"}),
                "pretty_print": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "summary")
    FUNCTION = "save"

    def save(self, content, output_dir, filename, format, pretty_print):
        os.makedirs(output_dir, exist_ok=True)

        ext = "json" if format == "json" else "txt"
        path = os.path.join(output_dir, f"{filename}.{ext}")
        counter = 0
        while os.path.exists(path):
            counter += 1
            path = os.path.join(output_dir, f"{filename}_{counter:03d}.{ext}")

        # Normalize content to text. JSON-typed content (dict/list) gets
        # serialized; anything else is written as-is via str().
        if isinstance(content, (dict, list)):
            if format == "json":
                text = json.dumps(content, indent=2 if pretty_print else None, ensure_ascii=False)
            else:
                text = json.dumps(content, ensure_ascii=False)
        else:
            text = str(content)
            # If the caller asked for json format but handed us a plain
            # string, try to re-parse + pretty-print it (e.g. HTTP
            # Request's response_text is a JSON string, not a dict) —
            # fall back to writing it verbatim if it isn't valid JSON.
            if format == "json":
                try:
                    parsed = json.loads(text)
                    text = json.dumps(parsed, indent=2 if pretty_print else None, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            summary = f"Failed to write {path}: {e}"
            return ("", summary)

        summary = (
            f"Format:     {format}\n"
            f"Bytes:      {len(text.encode('utf-8'))}\n"
            f"Saved to:   {path}"
        )
        return (path, summary)


NODE_CLASS_MAPPINGS = {
    "ResponseSaverNode": ResponseSaverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResponseSaverNode": "Response Saver 💾",
}
