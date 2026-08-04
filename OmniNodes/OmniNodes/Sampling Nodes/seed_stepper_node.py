"""
TensorVizion ComfyUI Nodes
seed_stepper_node.py — Tracks a history of seeds actually used across
queue runs and picks the next one according to `mode`. Batch Counter
already derives a fresh seed per run via base_seed + count*seed_step,
but has no memory of which specific seeds were used — this node adds
real history tracking, enabling "randomize but never repeat" and "cycle
through this exact list of seeds" modes that a pure counter can't do.
"""

import os
import json
import random

import folder_paths


class SeedStepperNode:
    """
    mode:
      increment          — same idea as Batch Counter's derived_seed, but
                             also logs every seed used to the history file
                             (useful if you want a readable audit trail of
                             exactly which seeds produced which run, not
                             just the current count).
      random_no_repeat   — picks a random seed in [0, max_seed] that does
                             NOT appear in this history file, retrying up
                             to 1000 times if a collision occurs. Once
                             `history_limit` seeds have been recorded, the
                             oldest entries are dropped so the file
                             doesn't grow forever and collisions stay rare.
      cycle_list         — steps through `seed_list` (comma-separated
                             integers) in order, wrapping back to the
                             start once exhausted. Ignores history/
                             randomness entirely — deterministic playback
                             of a specific, hand-picked seed sequence.

    `history_id` names the history file (`<history_id>.seedhistory.json`
    under ComfyUI's output/ folder), so multiple independent histories
    can run side by side.
    """

    CATEGORY = "TensorVizion/Sampling"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["increment", "random_no_repeat", "cycle_list"],),
                "history_id": ("STRING", {"default": "default"}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed_step": ("INT", {"default": 1, "min": 0, "max": 1000000}),
                "max_seed": ("INT", {"default": 0xffffffff, "min": 1, "max": 0xffffffffffffffff}),
                "history_limit": ("INT", {"default": 1000, "min": 10, "max": 100000}),
                "seed_list": ("STRING", {"default": "1,2,3,4,5"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("seed", "history_count", "summary")
    FUNCTION = "step"

    @classmethod
    def IS_CHANGED(cls, mode, history_id, base_seed, seed_step, max_seed, history_limit, seed_list):
        # Same reasoning as Batch Counter: this node's real state lives in
        # a file, not in its widget values, so it must be forced to
        # re-execute every queue run rather than being cache-skipped.
        return float("nan")

    def _history_path(self, history_id):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in history_id) or "default"
        out_dir = folder_paths.get_output_directory()
        return os.path.join(out_dir, f"{safe}.seedhistory.json")

    def _load_history(self, path):
        if not os.path.isfile(path):
            return {"seeds": [], "cycle_index": 0}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("seeds", [])
            data.setdefault("cycle_index", 0)
            return data
        except (json.JSONDecodeError, OSError):
            return {"seeds": [], "cycle_index": 0}

    def _save_history(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"[OmniNodes] Seed Stepper — could not write history file: {e}")

    def step(self, mode, history_id, base_seed, seed_step, max_seed, history_limit, seed_list):
        path = self._history_path(history_id)
        data = self._load_history(path)
        seeds = data["seeds"]

        if mode == "increment":
            seed = (base_seed + len(seeds) * seed_step) & 0xffffffffffffffff
            seeds.append(seed)
            if len(seeds) > history_limit:
                seeds = seeds[-history_limit:]
            data["seeds"] = seeds
            self._save_history(path, data)
            summary = f"increment: seed={seed} (run #{len(seeds)})"
            return (seed, len(seeds), summary)

        elif mode == "random_no_repeat":
            existing = set(seeds)
            seed = None
            for _ in range(1000):
                candidate = random.randint(0, max_seed)
                if candidate not in existing:
                    seed = candidate
                    break
            if seed is None:
                # Extremely unlikely at any reasonable max_seed, but fall
                # back to a fresh random value rather than erroring out —
                # a rare repeat is better than halting the queue.
                seed = random.randint(0, max_seed)
                note = " (WARNING: could not find a non-repeating seed after 1000 tries, allowed a repeat)"
            else:
                note = ""
            seeds.append(seed)
            if len(seeds) > history_limit:
                seeds = seeds[-history_limit:]
            data["seeds"] = seeds
            self._save_history(path, data)
            summary = f"random_no_repeat: seed={seed}{note} ({len(seeds)} in history)"
            return (seed, len(seeds), summary)

        else:  # cycle_list
            try:
                parsed = [int(s.strip()) for s in seed_list.split(",") if s.strip()]
            except ValueError:
                return (base_seed, len(seeds), f"cycle_list: could not parse seed_list '{seed_list}', using base_seed")

            if not parsed:
                return (base_seed, len(seeds), "cycle_list: seed_list is empty, using base_seed")

            idx = data["cycle_index"] % len(parsed)
            seed = parsed[idx]
            data["cycle_index"] = idx + 1
            self._save_history(path, data)
            summary = f"cycle_list: seed={seed} (position {idx + 1}/{len(parsed)})"
            return (seed, len(seeds), summary)


NODE_CLASS_MAPPINGS = {
    "SeedStepperNode": SeedStepperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedStepperNode": "Seed Stepper 🌱",
}
