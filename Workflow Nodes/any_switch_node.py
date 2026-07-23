"""
TensorVizion ComfyUI Nodes
any_switch_node.py — Boolean-gated router for any ComfyUI type. Picks
between two inputs without needing a type-specific switch node per data
type (one for MODEL, one for IMAGE, one for LATENT, etc.).
"""


class AlwaysEqualProxy(str):
    """Wildcard type marker — see smart_unloader.py for the full rationale.
    Re-declared locally so this file has no import-order dependency on
    Model Nodes/smart_unloader.py under the pack's per-file loader."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


class AnySwitchNode:
    """
    Routes `input_a` or `input_b` to `output` based on `condition` — a
    single reusable switch for any ComfyUI type (MODEL, IMAGE, LATENT,
    CONDITIONING, STRING, etc.) instead of needing a separate switch node
    per data type.

    `condition` True → passes `input_a` through.
    `condition` False → passes `input_b` through.

    `selected_label` reports which slot was chosen ("a" or "b") as a
    STRING, handy for feeding into a summary/log node or a Workflow End
    run_label so you can see which branch actually ran.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "input_a": (ANY_TYPE,),
                "input_b": (ANY_TYPE,),
            }
        }

    RETURN_TYPES  = (ANY_TYPE, "STRING")
    RETURN_NAMES  = ("output", "selected_label")
    FUNCTION      = "switch"

    def switch(self, condition, input_a=None, input_b=None):
        if condition:
            return (input_a, "a")
        return (input_b, "b")


NODE_CLASS_MAPPINGS = {
    "AnySwitchNode": AnySwitchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnySwitchNode": "Any Switch 🔀",
}
