"""
TensorVizion ComfyUI Nodes
batch_counter_node.py — Persistent, file-backed counter that increments
once per queued run. Useful for deriving a fresh seed per run in a queue,
or numbering outputs across separate executions (not just within a single
batch, which ComfyUI's own batch index already covers).
"""

import os

import folder_paths


class BatchCounterNode:
    """
    Tracks how many times this node has executed across separate queue
    runs, using a small counter file stored under ComfyUI's output/
    folder — so the count survives between runs and even across ComfyUI
    restarts (unlike an in-memory counter, which Smart Unloader-style
    state would lose on restart).

    `counter_id` names the counter file (`<counter_id>.counter` under
    output/), so you can run several independent counters side by side —
    one per workflow, for example.

    `step` controls how much the counter advances each run. `reset` zeroes
    the counter back to 0 on this execution (and then still counts this
    run, ending at `step` rather than 0, so a reset run still produces a
    usable seed instead of always handing out 0).

    `derived_seed` = `base_seed + count * seed_step` — plug this into any
    seed input downstream to get a new, reproducible seed every queued
    run without manually incrementing anything by hand.
    """

    CATEGORY = "TensorVizion/Workflow"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "counter_id": ("STRING", {"default": "default"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "reset": ("BOOLEAN", {"default": False}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed_step": ("INT", {"default": 1, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES  = ("INT", "INT")
    RETURN_NAMES  = ("count", "derived_seed")
    FUNCTION      = "count"

    @classmethod
    def IS_CHANGED(cls, counter_id, step, reset, base_seed, seed_step):
        # Widget inputs alone would look identical to ComfyUI's cache on
        # every queue run, so the counter would only ever execute once.
        # Returning NaN marks this node as always-changed, forcing a real
        # execution (and a real file read/increment) every time it's queued.
        return float("nan")

    def _counter_path(self, counter_id):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in counter_id) or "default"
        out_dir = folder_paths.get_output_directory()
        return os.path.join(out_dir, f"{safe}.counter")

    def count(self, counter_id, step, reset, base_seed, seed_step):
        path = self._counter_path(counter_id)

        current = 0
        if not reset and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    current = int(f.read().strip() or "0")
            except Exception:
                current = 0

        new_count = current + step if not reset else step

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(new_count))
        except Exception as e:
            print(f"[OmniNodes] Batch Counter — could not write counter file: {e}")

        derived_seed = (base_seed + new_count * seed_step) & 0xffffffffffffffff
        return (new_count, derived_seed)


NODE_CLASS_MAPPINGS = {
    "BatchCounterNode": BatchCounterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchCounterNode": "Batch Counter 🔢",
}
