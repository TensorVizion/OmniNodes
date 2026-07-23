"""
TensorVizion ComfyUI Nodes
workflow_end_node.py — Explicit terminal node for a workflow. Consumes up
to 4 dangling outputs of any type, optionally writes a run-log entry and a
"done" marker file, so external scripts/queues watching the output folder
can detect completion without polling ComfyUI's own API.
"""

import os
import time
import datetime

import folder_paths


class AlwaysEqualProxy(str):
    """Wildcard type marker — see smart_unloader.py for the full rationale.
    Re-declared locally so this file has no import-order dependency on
    Model Nodes/smart_unloader.py under the pack's per-file loader."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


ANY_TYPE = AlwaysEqualProxy("*")


class WorkflowEndNode:
    """
    Marks the end of a workflow. Connect any final outputs you want to make
    sure actually execute (an image that's already been saved, a summary
    string, a model you want unloaded first, etc.) into the four optional
    `input_N` sockets — none of them need to be used for anything else
    afterward, so this node is a clean place to let dangling branches
    terminate instead of leaving them unconnected.

    With `write_log` on, appends one line to `omninodes_run_log.txt` in
    ComfyUI's output/ folder: a timestamp and your `run_label`. With
    `write_done_marker` on, also (re)writes a small `<run_label>.done` file
    to output/ containing the completion timestamp — useful as a simple
    signal file for external automation (batch scripts, queue managers)
    watching the output folder rather than polling ComfyUI's HTTP API.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_label": ("STRING", {"default": "workflow"}),
                "write_log": ("BOOLEAN", {"default": True}),
                "write_done_marker": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_1": (ANY_TYPE,),
                "input_2": (ANY_TYPE,),
                "input_3": (ANY_TYPE,),
                "input_4": (ANY_TYPE,),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("summary",)
    FUNCTION      = "finish"
    OUTPUT_NODE   = True

    def finish(self, run_label, write_log, write_done_marker,
               input_1=None, input_2=None, input_3=None, input_4=None):
        connected = sum(1 for i in (input_1, input_2, input_3, input_4) if i is not None)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out_dir = folder_paths.get_output_directory()

        actions = []

        if write_log:
            try:
                log_path = os.path.join(out_dir, "omninodes_run_log.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {run_label} — {connected} input(s) reached workflow end\n")
                actions.append(f"logged to {os.path.basename(log_path)}")
            except Exception as e:
                actions.append(f"log write failed ({e})")

        if write_done_marker:
            try:
                safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_label) or "workflow"
                marker_path = os.path.join(out_dir, f"{safe_label}.done")
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write(f"done at {timestamp}\nepoch {time.time():.3f}\n")
                actions.append(f"wrote {os.path.basename(marker_path)}")
            except Exception as e:
                actions.append(f"marker write failed ({e})")

        summary = f"✅ Workflow '{run_label}' finished at {timestamp} ({connected}/4 inputs connected)"
        if actions:
            summary += "\n" + "\n".join(actions)

        print(f"[OmniNodes] {summary}")

        return {"ui": {"text": [summary]}, "result": (summary,)}


NODE_CLASS_MAPPINGS = {
    "WorkflowEndNode": WorkflowEndNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowEndNode": "Workflow End 🏁",
}
