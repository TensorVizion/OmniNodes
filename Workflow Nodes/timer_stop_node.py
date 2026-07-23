"""
TensorVizion ComfyUI Nodes
timer_stop_node.py — Reads the start timestamp recorded by a Timer Start
node sharing the same timer_id, and reports elapsed time. See
timer_start_node.py for the pairing pattern.
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
    Mirrors the same lookup in timer_start_node.py — see that file for why
    this uses an attribute on comfy.model_management rather than a direct
    cross-file import. Both files converge on the same dict as long as
    each computes the same attribute name on the same shared module.
    """
    if not hasattr(_cmm, "_omninodes_timer_starts"):
        _cmm._omninodes_timer_starts = {}
    return _cmm._omninodes_timer_starts


class TimerStopNode:
    """
    Marks the end of a timed section started by a Timer Start node with the
    same `timer_id`. Connect `passthrough` to whatever output ends the
    section you're measuring (forces correct graph ordering, same as
    Timer Start).

    If no matching Timer Start has run yet this queue (wrong/misspelled
    `timer_id`, or Timer Start simply hasn't executed), `elapsed_seconds`
    returns -1 and `summary` says so explicitly rather than silently
    reporting 0.
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

    RETURN_TYPES  = (ANY_TYPE, "FLOAT", "STRING")
    RETURN_NAMES  = ("output", "elapsed_seconds", "summary")
    FUNCTION      = "stop"

    @classmethod
    def IS_CHANGED(cls, timer_id, passthrough=None):
        return float("nan")

    def stop(self, timer_id, passthrough=None):
        start = _get_timer_store().get(timer_id)

        if start is None:
            summary = f"⚠️ No Timer Start found for timer_id '{timer_id}' this run."
            return (passthrough, -1.0, summary)

        elapsed = time.time() - start
        summary = f"⏱️ '{timer_id}' elapsed: {elapsed:.2f}s"
        return (passthrough, elapsed, summary)


NODE_CLASS_MAPPINGS = {
    "TimerStopNode": TimerStopNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimerStopNode": "Timer Stop ⏱️⏹️",
}
