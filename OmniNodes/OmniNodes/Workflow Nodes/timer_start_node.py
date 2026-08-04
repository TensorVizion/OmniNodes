"""
TensorVizion ComfyUI Nodes
timer_start_node.py — Records a start timestamp under a shared timer_id, to
be read back by Timer Stop later in the same graph. Use around any section
of a workflow you want to benchmark (e.g. before/after a heavy upscale or
model-merge chain).
"""

import time

import comfy.model_management as _cmm


class AlwaysEqualProxy(str):
    """Wildcard type marker — see smart_unloader.py for the full rationale.
    Re-declared locally so this file has no import-order dependency on
    Model Nodes/smart_unloader.py under the pack's per-file loader."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


def _get_timer_store():
    """
    Timer Start and Timer Stop live in separate files, each loaded as an
    independent module by OmniNodes' per-file loader (no shared package
    namespace, so a plain `from .timer_start_node import ...` between them
    won't resolve). Stashing the shared dict as an attribute on an
    already-imported, always-present ComfyUI module gives both files a
    single common place to read/write from regardless of load order.
    """
    if not hasattr(_cmm, "_omninodes_timer_starts"):
        _cmm._omninodes_timer_starts = {}
    return _cmm._omninodes_timer_starts


class TimerStartNode:
    """
    Marks the start of a timed section. Connect `passthrough` to whatever
    output starts the section you want to measure (forces this node to run
    at the right point in the graph, the same dependency trick Smart
    Unloader uses) and give it a `timer_id`. Pair with a Timer Stop node
    later in the graph using the *same* `timer_id` to get an elapsed-time
    reading.

    Multiple independent timers can run at once — just use different
    `timer_id` values for each pair.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timer_id": ("STRING", {"default": "default"}),
            },
            "optional": {
                "passthrough": (ANY_TYPE,),
            }
        }

    RETURN_TYPES  = (ANY_TYPE,)
    RETURN_NAMES  = ("output",)
    FUNCTION      = "start"

    @classmethod
    def IS_CHANGED(cls, timer_id, passthrough=None):
        # Always re-run so a repeated queue records a fresh start time
        # rather than reusing a cached timestamp from an earlier run.
        return float("nan")

    def start(self, timer_id, passthrough=None):
        _get_timer_store()[timer_id] = time.time()
        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "TimerStartNode": TimerStartNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimerStartNode": "Timer Start ⏱️▶️",
}
