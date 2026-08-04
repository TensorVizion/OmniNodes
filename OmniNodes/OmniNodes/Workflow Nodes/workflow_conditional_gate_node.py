"""
TensorVizion ComfyUI Nodes
workflow_conditional_gate_node.py — Evaluates a comparison between two
numeric or string values and routes an ANY-type payload down a "true" or
"false" output socket accordingly. Any Switch takes a pre-computed
BOOLEAN and picks between two payloads; this node computes the boolean
itself from a comparison so branching logic doesn't need a separate
math/compare node wired in front of it.
"""


class AlwaysEqualProxy(str):
    """Wildcard type marker — see smart_unloader.py / any_switch_node.py for
    the full rationale. Re-declared locally so this file has no
    import-order dependency under the pack's per-file loader."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


class WorkflowConditionalGateNode:
    """
    Compares `value_a` against `value_b` using `comparator` and routes
    `payload` to `output_true` when the condition holds, `output_false`
    otherwise (the socket not taken receives None). `condition_met`
    reports the boolean result directly, so it can also drive a Workflow
    End run_label or a log node without needing a second gate.

    Numeric comparators (==, !=, >, >=, <, <=) attempt to parse both values
    as floats first; if either fails to parse, falls back to string
    comparison (only == and != are meaningful for strings in that case).
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_a": ("STRING", {"default": "0"}),
                "comparator": (["==", "!=", ">", ">=", "<", "<="],),
                "value_b": ("STRING", {"default": "0"}),
            },
            "optional": {
                "payload": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE, "BOOLEAN")
    RETURN_NAMES = ("output_true", "output_false", "condition_met")
    FUNCTION = "gate"

    def _try_float(self, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def gate(self, value_a, comparator, value_b, payload=None):
        fa = self._try_float(value_a)
        fb = self._try_float(value_b)

        if fa is not None and fb is not None:
            a, b = fa, fb
        else:
            a, b = str(value_a), str(value_b)

        if comparator == "==":
            result = a == b
        elif comparator == "!=":
            result = a != b
        elif comparator == ">":
            result = a > b if not isinstance(a, str) else False
        elif comparator == ">=":
            result = a >= b if not isinstance(a, str) else False
        elif comparator == "<":
            result = a < b if not isinstance(a, str) else False
        elif comparator == "<=":
            result = a <= b if not isinstance(a, str) else False
        else:
            result = False

        if result:
            return (payload, None, True)
        return (None, payload, False)


NODE_CLASS_MAPPINGS = {
    "WorkflowConditionalGateNode": WorkflowConditionalGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowConditionalGateNode": "Conditional Gate 🚦",
}
