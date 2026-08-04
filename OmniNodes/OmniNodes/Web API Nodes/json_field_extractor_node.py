"""
TensorVizion ComfyUI Nodes
json_field_extractor_node.py — Pulls a single value out of nested JSON
(dict/list) via a dot-path, e.g. "data.items.0.title". Companion to
HTTP Request / RSS Feed Parser, which return whole JSON blobs that
downstream nodes usually only need one field from.
"""

import json


class AlwaysEqualProxy(str):
    """Wildcard type marker — see Workflow Nodes/any_switch_node.py for
    the full rationale. Re-declared locally so this file has no
    import-order dependency on another folder under the pack's per-file
    loader. Lets `json_input` accept JSON dict/list output (e.g. from
    HTTP Request's response_json) OR a plain STRING (e.g. response_text,
    or a manually typed/pasted blob) on the same socket."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


def _resolve_path(data, path: str):
    """
    Walk `data` using a dot-separated path. Numeric segments index into
    lists; anything else is treated as a dict key. Raises KeyError/
    IndexError/TypeError on a bad path — callers catch and report these
    as a clean "not found" rather than letting the traceback leak.
    """
    current = data
    if not path:
        return current
    for segment in path.split("."):
        if isinstance(current, list):
            current = current[int(segment)]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            raise TypeError(
                f"Cannot descend into segment '{segment}' — parent is "
                f"type {type(current).__name__}, not dict or list."
            )
    return current


class JSONFieldExtractorNode:
    """
    Extracts one value from a JSON object/string using a dot-path.

    Accepts either a JSON dict (e.g. straight from HTTP Request's
    `response_json` output) or a raw JSON string (e.g. `response_text`,
    or a manually pasted blob) on the same `json_input` socket. Numeric
    path segments index into arrays: "data.items.0.title" reaches the
    "title" key of the first element of the "items" array inside "data".

    If the path doesn't resolve — missing key, out-of-range index, or
    trying to descend into a non-container — `found` is False and
    `value_string` returns `fallback` instead of raising, so a broken
    path doesn't stop the whole workflow.
    """

    CATEGORY = "TensorVizion/Web API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": (ANY_TYPE,),
                "field_path": ("STRING", {"default": "data.0.name"}),
                "fallback": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", ANY_TYPE, "BOOLEAN", "STRING")
    RETURN_NAMES = ("value_string", "value_raw", "found", "summary")
    FUNCTION = "extract"

    def extract(self, json_input, field_path, fallback):
        # json_input may already be a dict/list (from HTTPRequestNode's
        # JSON output) or a raw string (from response_text or a manual
        # paste) — normalize to a Python object first.
        if isinstance(json_input, (dict, list)):
            data = json_input
        else:
            try:
                data = json.loads(json_input)
            except (json.JSONDecodeError, TypeError) as e:
                summary = f"Input is not valid JSON: {e}"
                return (fallback, fallback, False, summary)

        try:
            value = _resolve_path(data, field_path.strip())
        except (KeyError, IndexError, TypeError, ValueError) as e:
            summary = f"Path '{field_path}' not found: {e}"
            return (fallback, fallback, False, summary)

        # value_string is always a display-friendly string (dicts/lists
        # get JSON-serialized); value_raw preserves the original Python
        # type so a nested dict/list can be wired straight into another
        # node that expects structured data.
        if isinstance(value, (dict, list)):
            value_string = json.dumps(value, ensure_ascii=False)
        else:
            value_string = str(value)

        summary = f"Path: {field_path}\nType: {type(value).__name__}\nFound: True"
        return (value_string, value, True, summary)


NODE_CLASS_MAPPINGS = {
    "JSONFieldExtractorNode": JSONFieldExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONFieldExtractorNode": "JSON Field Extractor 🔎",
}
