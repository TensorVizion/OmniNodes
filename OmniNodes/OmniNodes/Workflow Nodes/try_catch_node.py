"""
TensorVizion ComfyUI Nodes
try_catch_node.py — Checks an upstream value against common "this
actually failed" signals (None, empty string, a string starting with a
configurable error prefix, or NaN/Inf for numeric types) and substitutes
a fallback value when one is detected, so a soft failure a few nodes
upstream doesn't propagate garbage through the rest of a long batch run.

IMPORTANT — real scope of what this can and can't do: ComfyUI does not
have a language-level try/except across the node graph; if an upstream
node raises an uncaught Python exception, ComfyUI halts that node and
everything downstream of it regardless of anything wired in after it —
no node in any custom pack can intercept that. What THIS node actually
does is check a VALUE that already came out of an upstream node's own
internal error handling (the pattern this whole pack's Web API category
already uses: catch the exception internally, return an error-marker
string/None instead of raising). This node is the reusable "if that
error-marker shows up, substitute something safe" step, so each upstream
node doesn't need its own bespoke fallback-routing logic wired around it.
"""

import math


class TryCatchNode:
    """
    `value` is checked in order:
      1. None -> treated as failed
      2. If `treat_empty_string_as_failure` and value == "" -> treated as failed
      3. If value is a string starting with `error_prefix` (default "ERROR")
         -> treated as failed
      4. If value is a float/int and is NaN or +/-Inf -> treated as failed

    If none of the above match, `value` passes through unchanged and
    `caught` is False. If a failure is detected, `fallback` is returned
    instead and `caught` is True — wire `caught` into a Conditional Gate
    or a log/notify node if you want to know when a fallback fired
    rather than just silently continuing with the fallback.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "", "forceInput": True}),
                "fallback": ("STRING", {"default": ""}),
                "error_prefix": ("STRING", {"default": "ERROR"}),
                "treat_empty_string_as_failure": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("value", "caught", "summary")
    FUNCTION = "check"

    def _is_failure(self, value, error_prefix, treat_empty_as_failure):
        if value is None:
            return True, "value was None"

        if isinstance(value, str):
            if treat_empty_as_failure and value == "":
                return True, "value was an empty string"
            if error_prefix and value.startswith(error_prefix):
                return True, f"value started with error prefix '{error_prefix}'"
            return False, ""

        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return True, "value was NaN/Inf"
            return False, ""

        return False, ""

    def check(self, value, fallback, error_prefix, treat_empty_string_as_failure):
        failed, reason = self._is_failure(value, error_prefix.strip(), treat_empty_string_as_failure)

        if failed:
            summary = f"Caught: {reason}. Substituted fallback."
            return (fallback, True, summary)

        summary = "No failure detected — value passed through unchanged."
        return (value, False, summary)


NODE_CLASS_MAPPINGS = {
    "TryCatchNode": TryCatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryCatchNode": "Try/Catch (Value Guard) 🛟",
}
