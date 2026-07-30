"""
TensorVizion ComfyUI Nodes
json_builder_node.py — Assembles a JSON object from up to 4 key/value
pairs, with automatic type coercion (numbers/booleans/null detected from
plain-text values). Companion to HTTP Request's `body` field, which
otherwise requires hand-writing a JSON string in a text widget.
"""

import json


def _coerce(value: str):
    """
    Best-effort type coercion for a plain-text widget value:
      "true"/"false" -> bool, "null" -> None, numeric strings -> int/float,
      everything else stays a string. Matches how a human typing into a
      JSON body field would expect "5" or "true" to behave rather than
      staying quoted as "\"5\"" / "\"true\"".
    """
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() == "null":
        return None
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass
    # If it looks like the user already typed JSON (a nested object/array/
    # quoted string), parse it as such instead of treating it as plain text.
    if stripped and stripped[0] in "{[" or (stripped.startswith('"') and stripped.endswith('"')):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


class JSONBuilderNode:
    """
    Builds a JSON object from up to 4 key/value pairs, for constructing
    request bodies (e.g. HTTP Request's `body` input) without hand-typing
    JSON syntax.

    Each value is a plain-text widget with light auto-coercion: "true" /
    "false" become booleans, "null" becomes JSON null, plain numeric
    strings become int/float, and text starting with `{`, `[`, or a
    quoted string is parsed as JSON (so a value can itself be a nested
    object/array/string if needed). Leave a key blank to skip that slot.

    `extra_json` optionally merges in a larger hand-written JSON object —
    handy when you need more than 4 fields or a deeply nested structure
    the 4 key/value slots can't express; keys here take priority over the
    4 slots on conflict.
    """

    CATEGORY = "TensorVizion/Web API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_1": ("STRING", {"default": ""}),
                "value_1": ("STRING", {"default": ""}),
                "key_2": ("STRING", {"default": ""}),
                "value_2": ("STRING", {"default": ""}),
                "key_3": ("STRING", {"default": ""}),
                "value_3": ("STRING", {"default": ""}),
                "key_4": ("STRING", {"default": ""}),
                "value_4": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_json": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("JSON", "STRING", "STRING")
    RETURN_NAMES = ("json_object", "json_string", "summary")
    FUNCTION = "build"

    def build(self, key_1, value_1, key_2, value_2, key_3, value_3, key_4, value_4, extra_json=""):
        obj = {}
        skipped = []

        for key, value in ((key_1, value_1), (key_2, value_2), (key_3, value_3), (key_4, value_4)):
            key = key.strip()
            if not key:
                continue
            obj[key] = _coerce(value)

        if extra_json.strip():
            try:
                extra = json.loads(extra_json)
                if isinstance(extra, dict):
                    obj.update(extra)
                else:
                    skipped.append("extra_json (not a JSON object, ignored)")
            except json.JSONDecodeError as e:
                skipped.append(f"extra_json (invalid JSON: {e})")

        json_string = json.dumps(obj, ensure_ascii=False)
        summary = f"Keys: {list(obj.keys())}"
        if skipped:
            summary += f"\nSkipped: {', '.join(skipped)}"

        return (obj, json_string, summary)


NODE_CLASS_MAPPINGS = {
    "JSONBuilderNode": JSONBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONBuilderNode": "JSON Builder 🧱",
}
